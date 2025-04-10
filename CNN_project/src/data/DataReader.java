package CNN_project.src.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class DataReader {
    // this class is to keep the image as a file

    private final int height = 28;//image height 
    private final int width = 28; // image width
    // since both won't change so made them final

    //  we will have a Reader function which will read the csv, takes path as input
    public List<Image> readData(String path) {
        List<Image> images = new ArrayList<>();
        // now read the values
        try {
            BufferedReader readLine = new BufferedReader(new FileReader(path));
            // we are reading the train data 
            String line = readLine.readLine();//1st line is columns name
            while ((line = readLine.readLine()) != null) {
                String[] lineItems = line.split(",");
                // so we will extract the feature of each Image and feed it to the Image class to make make by reading each row and columns from the train dataset

                double data[][] = new double[height][width];
                int labels = Integer.parseInt(lineItems[0]);
                int i = 1;
                for (int row = 0; row < height; row++) {
                    for (int col = 0; col < width; col++) {
                        data[row][col] = Integer.parseInt(lineItems[i++]);
                    }
                }
                // now creating image
                Image img = new Image(data, labels);
                images.add(img);
            }

        } catch (IOException e) {
            System.err.println(e.getMessage());
        }
        return images;
    }
}
