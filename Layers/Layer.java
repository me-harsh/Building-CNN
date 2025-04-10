package CNN_project.Layers;

import java.util.List;

public abstract class Layer {
    // we want to make this to make code modular and also to keep track of previous layer and next layer
    protected Layer _nextLayer;
    protected Layer _prevLayer;

    // Convolutional layer, Maxpool layer - takes list of 2d matrix
    // FCL and last softmax layer - takes only vector input
    // + Therefore we will use polymorphism
    public abstract double[] getOutput(List<double[][]> input);

    public abstract double[] getOutput(double[] input);

    // backpropogation is only for updating all the layers 
    public abstract void backpropogation(double[] dLdO);
}
