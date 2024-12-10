#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <sstream>

// You need to update these with your own training label statistics
#define MEAN 0.00083617732161656022
#define STD_DEV 0.00070963409962132573

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <data_file>" << std::endl;
        return -1;
    }

    std::string input_file_name = argv[1];

    torch::jit::script::Module module;
    try {
        module = torch::jit::load("../../../models/NNC2PL.pth");
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

    // Start measuring time
    auto start = std::chrono::high_resolution_clock::now();

    // Perform inference in batches
    std::vector<torch::Tensor> outputs;
    size_t batch_size = 25000;
    for (size_t i = 0; i < inputs.size(); i += batch_size) {
        std::vector<torch::Tensor> batch_inputs(inputs.begin() + i, inputs.begin() + std::min(i + batch_size, inputs.size()));
        torch::Tensor batch = torch::stack(batch_inputs);
        batch = batch.to(torch::kCUDA);
        std::vector<torch::jit::IValue> inputs_ivalue;
        inputs_ivalue.push_back(batch);
        torch::Tensor output = module.forward(inputs_ivalue).toTensor();
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

    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";

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
