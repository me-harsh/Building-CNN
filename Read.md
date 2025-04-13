Made a Data file to store the csv file

src folder is the main code source

Inside src-> data we will make all the necessary classes to read csv

The final form I want:- 1.**Input Layer**: 28x28 grayscale images (MNIST format) 2. **Convolutional Layer 1**: 16 filters, 3x3 kernel, stride 1, padding 1 3. **ReLU Activation** 4. **Max Pooling Layer 1**: 2x2 pool size, stride 2 5. **Convolutional Layer 2**: 32 filters, 3x3 kernel, stride 1, padding 1 6. **ReLU Activation** 7. **Max Pooling Layer 2**: 2x2 pool size, stride 2 8. **Flatten Layer**: Converts 2D feature maps to 1D vector 9. **Dense Layer 1**: 128 units (the hidden layer of CNN) 10. **ReLU Activation** 11. **Dense Layer 2 (Output)**: 10 units (one per digit) 12. **Softmax Activation**: Converts outputs to probabilities

Here is a helpful way to understand this:https://chatgpt.com/share/67fb68f0-38f0-8004-9691-716d3d5c4114

Now if we want colorfull images just the initial layer will also have 3 channels ie. 28x28x3 from the start.
