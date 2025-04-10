package Projects.CNN_project.src;

import java.util.List;
import pythonf.ML_course.CNN_project.src.data.DataReader;
import pythonf.ML_course.CNN_project.src.data.Image;

public class Main {
    public static void main(String args[]) {
        List<Image> images = new DataReader().readData(
                "/Users/harshraj/Documents/my coding/pythonf/ML_course/CNN_project/Data/train.csv");

        System.out.println(images.get(0));

    }
}
