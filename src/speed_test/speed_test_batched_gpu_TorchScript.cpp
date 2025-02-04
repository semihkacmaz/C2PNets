#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <sstream>
#include "cuda_runtime_api.h"

// You need to update these with your own training label statistics
#define MEAN 0.00083617732161656022
#define STD_DEV 0.00070963409962132573

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model> <data_file>" << std::endl;
        return -1;
    }

    std::string model_file_name = argv[1];
    std::string input_file_name = argv[2];

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_file_name);
        module.to(torch::kCUDA);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }

    std::ifstream input_file(input_file_name);
    if (!input_file.is_open()) {
        std::cerr << "Error opening the input file\n";
        return -1;
    }

    // Read and parse the input file
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
    std::vector<torch::Tensor> batches;
    std::vector<std::vector<torch::jit::IValue>> batch_inputs;
    size_t batch_size = 100000;

    // Prepare all batches before timing
    for (size_t i = 0; i < inputs.size(); i += batch_size) {
        std::vector<torch::Tensor> batch_tensors(inputs.begin() + i,
            inputs.begin() + std::min(i + batch_size, inputs.size()));
        torch::Tensor batch = torch::stack(batch_tensors);
        batch = batch.to(torch::kCUDA);

        std::vector<torch::jit::IValue> inputs_ivalue;
        inputs_ivalue.push_back(batch);

        batches.push_back(batch);
        batch_inputs.push_back(inputs_ivalue);
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Ensure all previous operations are complete
    cudaDeviceSynchronize();

    // Start timing
    cudaEventRecord(start);

    // Pure inference loop
    for (size_t i = 0; i < batch_inputs.size(); i++) {
        torch::Tensor output = module.forward(batch_inputs[i]).toTensor();
        outputs.push_back(output);
    }

    // Post-process results
    for (auto& batch_output : outputs) {
        batch_output = batch_output * STD_DEV + MEAN;
    }

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Pure inference time: " << milliseconds / 1000.0 << " s\n";

    for (auto& batch_output : outputs) {
        batch_output = batch_output.to(torch::kCPU);
    }

    std::ofstream output_file("preds_libtorch.txt");
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

    return 0;
}

// // LEGACY STATS
// #define MEAN 0.00083111191634088755
// #define STD_DEV 0.0007100614020600915

// #define MEAN 0.00084167707245796920
// #define STD_DEV 0.00071704556467011571

// // mean_std_dev trained on H100s on DTAI on October 10 24-- Legacy as of Nov 1
// #define MEAN 0.00083666382124647498
// #define STD_DEV 0.00071014417335391045
