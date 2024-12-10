import os
import torch
import tensorrt as trt
import numpy as np
from pathlib import Path
import yaml
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TRTEngineBuilder:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.model_name = self.config["model"]["name"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.device.type != 'cuda':
            raise RuntimeError("CUDA device is required for TensorRT engine building")

        self.trt_logger = trt.Logger(trt.Logger.VERBOSE)
        self.precision = self.config["tensorrt"]["precision"].lower()

        # Initialize paths
        self.setup_paths()

    def setup_paths(self):
        """Setup necessary paths for model and data"""
        base_dir = Path.cwd()
        self.models_dir = base_dir / self.config["paths"]["models"]
        self.speed_test_dir = base_dir / self.config["paths"]["speed_test"]

        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def load_input_data(self):
        """Load input test data"""
        input_path = self.speed_test_dir / self.config["data"]["input_file"]
        data = np.loadtxt(str(input_path))
        return torch.tensor(data, dtype=torch.float32).to(self.device)

    def get_engine_path(self):
        """Generate engine path based on configuration"""
        precision_suffix = "_FP16" if self.precision == "fp16" else ""
        dataset_size = self.config["data"]["dataset_size"]
        size_suffix = f"_{dataset_size}"

        engine_name = f"{self.model_name}{size_suffix}{precision_suffix}.engine"
        return self.models_dir / engine_name

    def get_onnx_path(self):
        """Generate ONNX model path"""
        dataset_size = self.config["data"]["dataset_size"]
        size_suffix = f"_{dataset_size}"
        return self.models_dir / f"{self.model_name}{size_suffix}.onnx"

    def export_to_onnx(self, inputs):
        """Export PyTorch model to ONNX format"""
        model_path = self.models_dir / f"{self.model_name}.pth"
        model = torch.jit.load(str(model_path), map_location=self.device)
        model.eval()

        onnx_path = self.get_onnx_path()
        torch.onnx.export(model, inputs, str(onnx_path), verbose=True)
        return onnx_path

    def build_engine(self):
        """Build TensorRT engine"""
        try:
            # Load input data
            inputs = self.load_input_data()

            # Export to ONNX
            onnx_path = self.export_to_onnx(inputs)

            # Create builder and network
            builder = trt.Builder(self.trt_logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, self.trt_logger)

            # Parse ONNX model
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"ONNX parsing error: {parser.get_error(error)}")
                    raise RuntimeError("Failed to parse ONNX model")

            # Create builder config
            config = builder.create_builder_config()
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE,
                self.config["tensorrt"]["workspace_size"] << 30
            )

            # Set FP16 mode if requested
            if self.precision == "fp16":
                if not builder.platform_has_fast_fp16:
                    logger.warning("Platform doesn't support fast FP16")
                config.set_flag(trt.BuilderFlag.FP16)

            # Set optimization profile
            profile = builder.create_optimization_profile()

            if self.config["tensorrt"]["batch_size"]["use_fixed"]:
                min_batch = self.config["tensorrt"]["batch_size"]["fixed"]["min"]
                opt_batch = self.config["tensorrt"]["batch_size"]["fixed"]["optimal"]
                max_batch = self.config["tensorrt"]["batch_size"]["fixed"]["max"]
            else:
                base_size = inputs.shape[0]
                min_batch = int(base_size * self.config["tensorrt"]["batch_size"]["min_factor"])
                opt_batch = int(base_size * self.config["tensorrt"]["batch_size"]["optimal_factor"])
                max_batch = int(base_size * self.config["tensorrt"]["batch_size"]["max_factor"])

            dim = inputs.shape[1]
            profile.set_shape(
                "inputs_test_tensor",
                trt.Dims((min_batch, dim)),
                trt.Dims((opt_batch, dim)),
                trt.Dims((max_batch, dim))
            )
            config.add_optimization_profile(profile)

            # Build and save engine
            logger.info("Building TensorRT engine...")
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")

            engine_path = self.get_engine_path()
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)

            logger.info(f"Successfully built and saved engine to {engine_path}")

def main():
    parser = argparse.ArgumentParser(description='Build TensorRT engine')
    parser.add_argument('--config', type=str, default='configs/engine_builder_config.yaml',
                      help='Path to configuration file')
    args = parser.parse_args()

    builder = TRTEngineBuilder(args.config)
    builder.build_engine()

if __name__ == "__main__":
    main()
