package CNN_project.Layers;

import java.util.ArrayList;
import java.util.List;

public abstract class Layer {
    // we want to make this to make code modular and also to keep track of previous layer and next layer
    protected Layer _nextLayer;
    protected Layer _prevLayer;
    protected int channel;

    public Layer getNextLayer() {
        return _nextLayer;
    }

    public void setNextLayer(Layer _nextLayer) {
        this._nextLayer = _nextLayer;
    }

    public Layer getPrevLayer() {
        return _prevLayer;
    }

    public void setPrevLayer(Layer _prevLayer) {
        this._prevLayer = _prevLayer;
    }

    public int getChannel() {
        return channel;
    }


    // TODO: each layer should also get prevLayer's no. of channels 
    // TODO: each layer should also provide its no. of channels to the nextLayer

    // Convolutional layer, Maxpool layer - takes list of 2d matrix
    // FCL and last softmax layer - takes only vector input
    // + Therefore we will use polymorphism
    public abstract double[] getOutput(List<double[][]> input);

    public abstract double[] getOutput(double[] input);

    // backpropogation is only for updating all the layers 
    public abstract void backpropogation(double[] dLdO);

    public abstract void backpropogation(List<double[][]> dLdO);

    // now we will have a conversion function which takes the matrix and convert it to vector
    public double[] matrixToVector(List<double[][]> input) {
        int length = input.size();
        int rows = input.get(0).length;
        int cols = input.get(0)[0].length;

        double[] vector = new double[length * rows * cols];
        int i = 0;
        for (int p = 0; p < length; p++) {
            for (int q = 0; q < rows; q++) {
                for (int r = 0; r < cols; r++) {
                    vector[i] = input.get(p)[q][r];
                    i++;
                }
            }
        }
        return vector;
    }

    // now for the reverse action
    public List<double[][]> vectorToMatrix(double[] input, int length, int rows, int cols) {
        List<double[][]> out = new ArrayList<>();
        int i = 0;
        // TODO: see if here can we just work with l variable. Like why use 'i'.
        for (int l = 0; l < length; l++) {
            double[][] image = new double[rows][cols];
            for (int j = 0; j < rows; j++) {
                for (int k = 0; k < cols; k++) {
                    image[j][k] = input[i];
                    i++;
                }
            }
            out.add(image);
        }
        return out;
    }

    // One more thing we want is that given the size of this layer what is its expected outputsize
    // so that when we build a layer after that we can check the last layer outputsize
}
