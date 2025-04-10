package CNN_project.src.data;

public class Image {
    private double[][] data;
    private int label;

    // labelling the image is not needed here as the image name is the digits itself
    public Image(double[][] data, int label) {
        this.data = data;
        this.label = label;
    }

    public double[][] getData() {
        return data;
    }

    public int getLabel() {
        return label;
    }

    @Override
    public String toString() {
        String s = label + ", \n";
        for (int row = 0; row < data.length; row++) {
            for (int col = 0; col < data[0].length; col++) {
                s += data[row][col] + ", ";
            }
            s += "\n";
        }
        return s;
    }
}
