#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <sstream>

// Include TensorRT headers
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "NvOnnxParser.h"

// Up-to-date Stats for Small w Pen150 Medium w Pen350
// You need to update these with your own training label statistics
#define MEAN 0.00083617732161656022
#define STD_DEV 0.00070963409962132573

// Up-to-date Stats for Tabulated Model
// #define MEAN 30.08158874511718750
// #define STD_DEV 4.34816455841064453

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {  // Add noexcept
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_file> <data_file>" << std::endl;
        return -1;
    }

    std::string engine_file_name = argv[1];
    std::string input_file_name = argv[2];

    // TensorRT initialization
    Logger logger;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    std::ifstream engine_file(engine_file_name, std::ios::binary);

    if (!engine_file) {
        std::cerr << "Error opening TensorRT engine file: " << engine_file_name << std::endl;
        return -1;
    }
    engine_file.seekg(0, engine_file.end);
    size_t engine_size = engine_file.tellg();
    engine_file.seekg(0, engine_file.beg);
    std::vector<char> engine_data(engine_size);
    engine_file.read(engine_data.data(), engine_size);
    engine_file.close();

    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_size, nullptr);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    std::ifstream input_file(input_file_name);
    if (!input_file.is_open()) {
        std::cerr << "Error opening the input file\n";
        return -1;
    }

    std::vector<torch::Tensor> inputs;
    std::string line;
    while (std::getline(input_file, line)) {
        std::istringstream iss(line);
        std::vector<float> data((std::istream_iterator<float>(iss)), std::istream_iterator<float>());
        inputs.push_back(torch::tensor(data).to(torch::kCUDA));
    }
    input_file.close();

    // Start measuring time
    auto start = std::chrono::high_resolution_clock::now();

    // Inference in batches
    std::vector<torch::Tensor> outputs;
    size_t batch_size = 25000;
    for (size_t i = 0; i < inputs.size(); i += batch_size) {
        std::vector<torch::Tensor> batch_inputs(inputs.begin() + i, inputs.begin() + std::min(i + batch_size, inputs.size()));
        torch::Tensor batch = torch::stack(batch_inputs).to(torch::kCUDA);

        // Inference setup
        std::vector<void*> buffers(2);
        buffers[0] = batch.data_ptr<float>();
        torch::Tensor output = torch::empty({batch.size(0), 1}, torch::kCUDA);
        buffers[1] = output.data_ptr<float>();

        // Perform inference
        context->executeV2(buffers.data());

        outputs.push_back(output);
    }

    // Apply inverse standard scaling to the outputs on GPU
    for (auto& batch_output : outputs) {
        batch_output = batch_output * STD_DEV + MEAN;
    }

    // Stop measuring time
    auto finish = std::chrono::high_resolution_clock::now();

    for (auto& batch_output : outputs) {
        batch_output = batch_output.to(torch::kCPU);
    }

    std::ofstream output_file("preds_tensorrt.txt");
    if (!output_file.is_open()) {
        std::cerr << "Error opening the output file\n";
       return -1;
    }
    output_file << std::fixed << std::setprecision(11);

    for (const auto& batch_output : outputs) {
       for (int i = 0; i < batch_output.size(0); ++i) {
           for (int j = 0; j < batch_output.size(1); ++j) {
               output_file << batch_output[i][j].item<float>() << " ";
           }
           output_file << "\n";
       }
    }
    output_file.close();

    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";

    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}

// // mean_std_dev trained on H100s on DTAI on October 10 24-- Legacy as of Nov 1
// #define MEAN 0.00083666382124647498
// #define STD_DEV 0.00071014417335391045
