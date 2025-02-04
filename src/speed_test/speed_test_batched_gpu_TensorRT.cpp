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

    // Pre-process all batches
    std::vector<torch::Tensor> outputs;
    std::vector<torch::Tensor> input_batches;
    std::vector<std::vector<void*>> batch_buffers;
    const size_t batch_size = 100000;  // Fixed batch size
    const size_t num_batches = (inputs.size() + batch_size - 1) / batch_size;  // Ceiling division

    // Prepare input batches with fixed size
    for (size_t i = 0; i < num_batches; i++) {
        size_t start_idx = i * batch_size;
        size_t end_idx = std::min(start_idx + batch_size, inputs.size());
        size_t current_batch_size = end_idx - start_idx;

        std::vector<torch::Tensor> batch_inputs(inputs.begin() + start_idx,
                                              inputs.begin() + end_idx);

        if (current_batch_size < batch_size) {
            // Pad last batch if needed
            batch_inputs.resize(batch_size, batch_inputs.back());
        }

        torch::Tensor input_batch = torch::stack(batch_inputs).to(torch::kCUDA);
        input_batches.push_back(input_batch);

        std::vector<void*> buffers(2);
        buffers[0] = input_batch.data_ptr<float>();
        batch_buffers.push_back(buffers);
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    cudaEventRecord(start);

    // Pure inference loop with fixed batch size
    for (size_t i = 0; i < num_batches; i++) {
        torch::Tensor output = torch::empty({batch_size, 1}, torch::kCUDA);
        batch_buffers[i][1] = output.data_ptr<float>();

        context->executeV2(batch_buffers[i].data());

        // Only store the valid outputs for the last batch
        if (i == num_batches - 1 && (inputs.size() % batch_size) != 0) {
            output = output.slice(0, 0, inputs.size() % batch_size);
        }

        output = output * STD_DEV + MEAN;
        outputs.push_back(output);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Pure inference time: " << milliseconds / 1000.0 << " s\n";

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


    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}

// // mean_std_dev trained on H100s on DTAI on October 10 24-- Legacy as of Nov 1
// #define MEAN 0.00083666382124647498
// #define STD_DEV 0.00071014417335391045
