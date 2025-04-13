package CNN_project.Layers;

import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer {
    private double[][] _weights;
    private int _inLength;
    private int _outLength;
    private long SEED;
    private double[] lastX;
    private double[] lastZ;

    public FullyConnectedLayer(int _inLength, int _outLength, long SEED) {
        this.SEED = SEED;
        this._inLength = _inLength;
        this._outLength = _outLength;
        _weights = new double[_inLength][_outLength];
        setRandomWeights();
    }

    // forward pass for FCL
    public double[] forwardPass(double[] input) {
        lastX = input;
        double[] z = new double[_outLength];
        double[] out = new double[_outLength];
        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                z[j] += input[i] * _weights[i][j];
            }
        }
        lastZ = z;
        // making it in two pass coz we need to save the output of each layer
        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                out[j] = ReLU(z[j]);
            }
        }
        // now we will give this value to activation function
        return out;
    }

    // writing the getOutput function
    public double[] getOutput(List<double[][]> input) {
        // if we are getting someother input like a matrix then we will convert it to vector
        double[] vector = matrixToVector(input);
        return getOutput(vector);
    }

    public double[] getOutput(double[] input) {
        // so we are passing our output to next layers and if there is some input missmatch then we have polymorphism for that
        double[] forward_pass = forwardPass(input);
        if (_nextLayer != null) {
            return _nextLayer.getOutput(forward_pass);
        } else {
            return forward_pass;
        }
    }


    // activation function
    public double ReLU(double input) {
        return input > 0 ? input : 0;
    }

    // to initialise random weights to weight matrix
    private void setRandomWeights() {
        Random random = new Random(SEED);
        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                _weights[i][j] = random.nextGaussian();
            }
        }
    }
}
