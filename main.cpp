#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>

using namespace std;

// ---------------------------
// 3D Tensor Structure
// ---------------------------
struct Tensor3D {
    int channels, height, width;
    vector<double> data;
    Tensor3D(int c, int h, int w) : channels(c), height(h), width(w) {
        data.resize(c * h * w, 0.0);
    }
    Tensor3D() : channels(0), height(0), width(0) {}
    double& operator()(int c, int h, int w) {
        return data[c * height * width + h * width + w];
    }
    const double& operator()(int c, int h, int w) const {
        return data[c * height * width + h * width + w];
    }
};

// ---------------------------
// Convolution Layer
// ---------------------------
class ConvLayer {
public:
    int in_channels, out_channels, kernel_size, stride;
    vector<double> weights; // size: out_channels * in_channels * kernel_size * kernel_size
    vector<double> biases;  // size: out_channels
    Tensor3D input_cache;   // store input for backprop

    ConvLayer(int in_channels, int out_channels, int kernel_size, int stride)
        : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride),
          input_cache(in_channels, 0, 0)
    {
        weights.resize(out_channels * in_channels * kernel_size * kernel_size);
        biases.resize(out_channels, 0.0);
        // Randomly initialize weights
        for (int i = 0; i < weights.size(); i++) {
            weights[i] = ((double)rand() / RAND_MAX - 0.5) * 0.2;
        }
    }

    // Forward pass: computes convolution output.
    Tensor3D forward(const Tensor3D &input) {
        input_cache = input;  // save input for backprop
        int out_height = (input.height - kernel_size) / stride + 1;
        int out_width  = (input.width - kernel_size) / stride + 1;
        Tensor3D output(out_channels, out_height, out_width);
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    double sum = biases[oc];
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int ih = oh * stride + kh;
                                int iw = ow * stride + kw;
                                int w_index = oc * (in_channels * kernel_size * kernel_size)
                                              + ic * (kernel_size * kernel_size)
                                              + kh * kernel_size + kw;
                                sum += input(ic, ih, iw) * weights[w_index];
                            }
                        }
                    }
                    output(oc, oh, ow) = sum;
                }
            }
        }
        return output;
    }

    // Backward pass: given gradient from next layer, compute gradient wrt input and update weights.
    Tensor3D backward(const Tensor3D &d_out, double learning_rate) {
        int out_height = d_out.height;
        int out_width = d_out.width;
        vector<double> d_weights(weights.size(), 0.0);
        vector<double> d_biases(biases.size(), 0.0);
        Tensor3D d_input(input_cache.channels, input_cache.height, input_cache.width);
        fill(d_input.data.begin(), d_input.data.end(), 0.0);

        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    double grad_out = d_out(oc, oh, ow);
                    d_biases[oc] += grad_out;
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int ih = oh * stride + kh;
                                int iw = ow * stride + kw;
                                int w_index = oc * (in_channels * kernel_size * kernel_size)
                                              + ic * (kernel_size * kernel_size)
                                              + kh * kernel_size + kw;
                                d_weights[w_index] += input_cache(ic, ih, iw) * grad_out;
                                d_input(ic, ih, iw) += weights[w_index] * grad_out;
                            }
                        }
                    }
                }
            }
        }
        // Update weights and biases.
        for (int i = 0; i < weights.size(); i++) {
            weights[i] -= learning_rate * d_weights[i];
        }
        for (int oc = 0; oc < out_channels; oc++) {
            biases[oc] -= learning_rate * d_biases[oc];
        }
        return d_input;
    }

    // Save the layer parameters to a file.
    void save(const string &filename) {
        ofstream out(filename);
        if (!out.is_open()) {
            cerr << "Failed to open " << filename << " for saving ConvLayer." << endl;
            return;
        }
        // Save dimensions.
        out << in_channels << " " << out_channels << " " << kernel_size << " " << stride << "\n";
        // Save weights.
        for (double w : weights)
            out << w << " ";
        out << "\n";
        // Save biases.
        for (double b : biases)
            out << b << " ";
        out << "\n";
        out.close();
    }
};

// ---------------------------
// ReLU Activation Layer
// ---------------------------
class ReLULayer {
public:
    Tensor3D input_cache;
    Tensor3D forward(const Tensor3D &input) {
        input_cache = input;
        Tensor3D output = input;
        for (int i = 0; i < output.data.size(); i++) {
            if (output.data[i] < 0)
                output.data[i] = 0;
        }
        return output;
    }
    Tensor3D backward(const Tensor3D &d_out) {
        Tensor3D d_input = d_out;
        for (int i = 0; i < input_cache.data.size(); i++) {
            if (input_cache.data[i] <= 0)
                d_input.data[i] = 0;
        }
        return d_input;
    }
};

// ---------------------------
// Max Pooling Layer
// ---------------------------
class MaxPoolLayer {
public:
    int pool_size, stride;
    vector<int> max_indices; // stores index in input.data of the max element for each pooling window
    int input_channels, input_height, input_width;
    MaxPoolLayer(int pool_size, int stride) : pool_size(pool_size), stride(stride) {}

    Tensor3D forward(const Tensor3D &input) {
        input_channels = input.channels;
        input_height = input.height;
        input_width = input.width;
        int out_height = (input.height - pool_size) / stride + 1;
        int out_width = (input.width - pool_size) / stride + 1;
        Tensor3D output(input.channels, out_height, out_width);
        max_indices.resize(input.channels * out_height * out_width, -1);
        for (int c = 0; c < input.channels; c++) {
            for (int h = 0; h < out_height; h++) {
                for (int w = 0; w < out_width; w++) {
                    double max_val = -1e9;
                    int max_index = -1;
                    for (int ph = 0; ph < pool_size; ph++) {
                        for (int pw = 0; pw < pool_size; pw++) {
                            int ih = h * stride + ph;
                            int iw = w * stride + pw;
                            int index = c * (input.height * input.width) + ih * input.width + iw;
                            if (input.data[index] > max_val) {
                                max_val = input.data[index];
                                max_index = index;
                            }
                        }
                    }
                    output(c, h, w) = max_val;
                    int out_index = c * (out_height * out_width) + h * out_width + w;
                    max_indices[out_index] = max_index;
                }
            }
        }
        return output;
    }

    Tensor3D backward(const Tensor3D &d_out) {
        int out_height = d_out.height;
        int out_width = d_out.width;
        Tensor3D d_input(input_channels, input_height, input_width);
        fill(d_input.data.begin(), d_input.data.end(), 0.0);
        for (int c = 0; c < input_channels; c++) {
            for (int h = 0; h < out_height; h++) {
                for (int w = 0; w < out_width; w++) {
                    int out_index = c * (out_height * out_width) + h * out_width + w;
                    int max_index = max_indices[out_index];
                    d_input.data[max_index] += d_out(c, h, w);
                }
            }
        }
        return d_input;
    }
};

// ---------------------------
// Fully Connected (Dense) Layer
// ---------------------------
class FCLayer {
public:
    int input_size, output_size;
    vector<vector<double>> weights; // dimensions: [output_size][input_size]
    vector<double> biases;          // size: output_size
    vector<double> input_cache;     // store input vector for backprop

    FCLayer(int input_size, int output_size) : input_size(input_size), output_size(output_size) {
        weights.resize(output_size, vector<double>(input_size));
        biases.resize(output_size, 0.0);
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                weights[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.2;
            }
        }
    }

    vector<double> forward(const vector<double> &input) {
        input_cache = input;
        vector<double> output(output_size, 0.0);
        for (int i = 0; i < output_size; i++) {
            double sum = biases[i];
            for (int j = 0; j < input_size; j++) {
                sum += weights[i][j] * input[j];
            }
            output[i] = sum;
        }
        return output;
    }

    // Backward pass: updates weights and returns gradient with respect to input.
    vector<double> backward(const vector<double> &d_out, double learning_rate) {
        vector<double> d_input(input_size, 0.0);
        for (int j = 0; j < input_size; j++) {
            double sum = 0.0;
            for (int i = 0; i < output_size; i++) {
                sum += weights[i][j] * d_out[i];
            }
            d_input[j] = sum;
        }
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                double grad = d_out[i] * input_cache[j];
                weights[i][j] -= learning_rate * grad;
            }
            biases[i] -= learning_rate * d_out[i];
        }
        return d_input;
    }

    // Save the fully connected layer parameters.
    void save(const string &filename) {
        ofstream out(filename);
        if (!out.is_open()) {
            cerr << "Failed to open " << filename << " for saving FCLayer." << endl;
            return;
        }
        // Save dimensions.
        out << input_size << " " << output_size << "\n";
        // Save weights.
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                out << weights[i][j] << " ";
            }
            out << "\n";
        }
        // Save biases.
        for (int i = 0; i < output_size; i++) {
            out << biases[i] << " ";
        }
        out << "\n";
        out.close();
    }
};

// ---------------------------
// Utility Functions
// ---------------------------
vector<double> softmax(const vector<double>& x) {
    vector<double> exp_x(x.size());
    double sum = 0.0;
    for (int i = 0; i < x.size(); i++) {
        exp_x[i] = exp(x[i]);
        sum += exp_x[i];
    }
    for (int i = 0; i < x.size(); i++) {
        exp_x[i] /= sum;
    }
    return exp_x;
}

double cross_entropy_loss(const vector<double>& prob, int label) {
    return -log(prob[label] + 1e-9);
}

vector<double> flatten(const Tensor3D &tensor) {
    return tensor.data;
}

// ---------------------------
// Data Loading Structures & Functions
// ---------------------------
struct TrainingSample {
    Tensor3D input; // image tensor (1 x 28 x 28)
    int label;
    TrainingSample(const Tensor3D &img, int lbl) : input(img), label(lbl) {}
};

// Assumes CSV file with a header, first column as label, next 784 columns as pixel values.
vector<TrainingSample> load_dataset(const string &filename) {
    vector<TrainingSample> dataset;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return dataset;
    }
    string line;
    // Skip header.
    getline(file, line);
    while (getline(file, line)) {
        stringstream ss(line);
        string token;
        getline(ss, token, ',');
        int label = stoi(token);
        Tensor3D image(1, 28, 28);
        for (int i = 0; i < 28 * 28; i++) {
            if (!getline(ss, token, ',')) break;
            image.data[i] = stod(token) / 255.0; // Normalize pixels.
        }
        dataset.push_back(TrainingSample(image, label));
    }
    file.close();
    return dataset;
}

// ---------------------------
// Evaluation Function
// ---------------------------
double evaluate_model(const vector<TrainingSample>& dataset, ConvLayer &conv, ReLULayer &relu, MaxPoolLayer &pool, FCLayer &fc) {
    int correct = 0;
    for (const auto &sample : dataset) {
        Tensor3D conv_out = conv.forward(sample.input);
        Tensor3D relu_out = relu.forward(conv_out);
        Tensor3D pool_out = pool.forward(relu_out);
        vector<double> fc_input = flatten(pool_out);
        vector<double> fc_out = fc.forward(fc_input);
        vector<double> probabilities = softmax(fc_out);
        int predicted = distance(probabilities.begin(), max_element(probabilities.begin(), probabilities.end()));
        if (predicted == sample.label)
            correct++;
    }
    return 100.0 * correct / dataset.size();
}

// ---------------------------
// Main Training Routine
// ---------------------------
int main() {
    srand(time(0));

    // Load training data.
    string train_filename = "/Users/vedanthaldia/Desktop/Study/3rd Year/3-2/Machine Learning/cnn/test/train.csv";
    vector<TrainingSample> dataset = load_dataset(train_filename);
    if (dataset.empty()) {
        cerr << "No training data loaded." << endl;
        return 1;
    }
    
    // Shuffle dataset.
    random_device rd;
    mt19937 g(rd());
    shuffle(dataset.begin(), dataset.end(), g);
    
    // Split dataset: 80% training, 20% validation.
    size_t train_size = dataset.size() * 0.8;
    vector<TrainingSample> training_set(dataset.begin(), dataset.begin() + train_size);
    vector<TrainingSample> validation_set(dataset.begin() + train_size, dataset.end());
    
    // Hyperparameters.
    double learning_rate = 0.01;
    int epochs = 10;
    
    // Build CNN architecture.
    ConvLayer conv(1, 8, 3, 1);
    ReLULayer relu;
    MaxPoolLayer pool(2, 2);
    // After conv->ReLU->pool: image 28x28 becomes 8 x 13 x 13.
    int flattened_size = 8 * 13 * 13;
    FCLayer fc(flattened_size, 10);
    
    // Training loop.
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        int sample_count = 0;
        cout << "Epoch " << epoch + 1 << "/" << epochs << endl;
        for (size_t i = 0; i < training_set.size(); i++) {
            const auto &sample = training_set[i];
            // Forward pass.
            Tensor3D conv_out = conv.forward(sample.input);
            Tensor3D relu_out = relu.forward(conv_out);
            Tensor3D pool_out = pool.forward(relu_out);
            vector<double> fc_input = flatten(pool_out);
            vector<double> fc_out = fc.forward(fc_input);
            vector<double> probabilities = softmax(fc_out);
            double loss = cross_entropy_loss(probabilities, sample.label);
            total_loss += loss;
            sample_count++;
            
            // Compute gradient for softmax cross-entropy.
            vector<double> d_fc_out = probabilities;
            d_fc_out[sample.label] -= 1.0;
            
            // Backpropagation.
            vector<double> d_fc_input = fc.backward(d_fc_out, learning_rate);
            Tensor3D d_pool_out(pool_out.channels, pool_out.height, pool_out.width);
            for (int j = 0; j < d_fc_input.size(); j++) {
                d_pool_out.data[j] = d_fc_input[j];
            }
            Tensor3D d_relu_out = pool.backward(d_pool_out);
            Tensor3D d_conv_out = relu.backward(d_relu_out);
            conv.backward(d_conv_out, learning_rate);
            
            // Track progress every 100 samples.
            if ((i + 1) % 100 == 0 || i + 1 == training_set.size()) {
                cout << "  Processed " << i + 1 << " / " << training_set.size() << " samples\r" << flush;
            }
        }
        cout << endl;
        cout << "  Average Loss: " << total_loss / sample_count << endl;
        
        // Evaluate on training and validation sets.
        double train_accuracy = evaluate_model(training_set, conv, relu, pool, fc);
        double val_accuracy = evaluate_model(validation_set, conv, relu, pool, fc);
        cout << "  Training Accuracy: " << train_accuracy << "%" << endl;
        cout << "  Validation Accuracy: " << val_accuracy << "%" << endl;
        cout << "----------------------------------------" << endl;
    }
    
    // Save the trained model parameters.
    conv.save("conv_model.txt");
    fc.save("fc_model.txt");
    cout << "Model parameters saved to conv_model.txt and fc_model.txt." << endl;
    
    // Load and evaluate on test dataset.
    string test_filename = "/Users/vedanthaldia/Desktop/Study/3rd Year/3-2/Machine Learning/cnn/test/test.csv";
    vector<TrainingSample> test_set = load_dataset(test_filename);
    if (test_set.empty()) {
        cerr << "No test data loaded." << endl;
        return 1;
    }
    double test_accuracy = evaluate_model(test_set, conv, relu, pool, fc);
    cout << "Test Accuracy: " << test_accuracy << "%" << endl;
    
    return 0;
}