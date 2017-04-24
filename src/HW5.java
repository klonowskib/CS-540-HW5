/***************************************************************************************
 CS540 - Section 2
 Homework Assignment 5: Naive Bayes

 NBClassifierImpl.java
 This is the main class that implements entry functions!
 ---------
 *Free to modify anything in this file, except the class name
 You are required:
 - To keep the class name as HW5 for testing
 - Not to import any external libraries
 - Not to include any packages
 *Notice: To use this file, you should implement 1 method below.

 @author: TA
 @date: April 2017
 *****************************************************************************************/


import java.io.*;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

public class
HW5 {
    /**
     * Creates a fresh instance of the classifier.
     *
     * @return	a classifier
     */
    private static NBClassifier getNewClassifier(int[] iFeatures) {
        NBClassifier nbc = new NBClassifierImpl(iFeatures);
        return nbc;
    }

    /**
     * Main method reads command-line flags and outputs the classifications
     * of the test file.
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        // Check args
        if (args.length < 2) {
            System.out.println("usage: java HW5 <trainingFilename> <testFilename>");
        }

        // File names
        String trainFilename = new String(args[0]).trim();
        String testFilename = new String(args[1]).trim();

        // Training data
        List<String[]> trainStrData = loadFile(trainFilename);

        // Get feature field of data (1st row)
        String[] strFeatures = trainStrData.get(0);
        int[] iFeatures = tranformInstance(strFeatures);

        // Convert data into an 2D umeric array
        int[][] trainData = createInstances(trainStrData);

        // Test data
        List<String[]> testStrData = loadFile(testFilename);
        int[][] testData = createInstances(testStrData);

        // Training
        NBClassifier nbc = getNewClassifier(iFeatures);
        nbc.fit(trainData);

        // Classify test set
        Label[] yPred = nbc.classify(testData);

        // Ground truth
        int nrows = testData.length;
        int ncols = testData[0].length;
        Label[] yTrue = new Label[nrows];
        for (int i = 0; i < nrows; ++i) {
            if (testData[i][ncols - 1] == 0)
                yTrue[i] = Label.Negative;
            else
                yTrue[i] = Label.Positive;
        }

        // Evaluate
        double[] scores = evaluate(yTrue, yPred);

        // Print to the console
        System.out.printf("Correctly Classified:\t%.3f\n", scores[0]);
        System.out.printf("Precision:\t%.3f\n", scores[1]);
        System.out.printf("Recall: \t%.3f\n", scores[2]);
    }

    /**
     * Remove the first row (feature) and convert data into numeric features.
     *
     * @param  data List<String[]>
     * @return int[][] transformed data
     * @throws IOException
     */
    private static int[][] createInstances(List<String[]> data) throws IOException {
        data.remove(0);
        String[][] raw = data.toArray(new String[data.size()][data.get(0).length]);
        return processData(raw);
    }

    /**
     * Read input file and return a list of string arrays
     *
     * @param fileName String
     * @return List<String[]> dataset
     */
    private static List<String[]> loadFile(String fileName) {

        List<String[]> data = new ArrayList<String[]>();

        try {
            // FileReader reads text files in the default encoding.
            FileReader fileReader =
                    new FileReader(fileName);

            // Always wrap FileReader in BufferedReader.
            BufferedReader bufferedReader =
                    new BufferedReader(fileReader);

            // bufferedReader.readLine(); // this will read the first line

            // This will reference one line at a time
            String line = null;
            while((line = bufferedReader.readLine()) != null) {
                // Split
                String[] row = (new String(line)).trim().split(",");
                data.add(row);
            }

            // Always close files.
            bufferedReader.close();
        }
        catch(FileNotFoundException ex) {
            System.out.println(
                    "Unable to open file '" +
                            fileName + "'");
        }

        catch(IOException ex) {
            System.out.println(
                    "Error reading file '"
                            + fileName + "'");
        }

        return data;
    }

    /**
     * Transform String dataset into the numeric one
     *
     * @param rawData String[][] data
     * @return int[][] new dataset
     */
    private static int[][] processData(String[][] rawData) {
        int nrows = rawData.length;
        int ncols = rawData[0].length;

        int[][] vectors = new int[nrows][ncols];

        for (int i = 0; i < nrows; ++i) {
            vectors[i] = (tranformInstance(rawData[i])).clone();
        }

        return vectors;
    }

    /**
     * Convert a row of strings into integers
     *
     * @param row String[] row of dataset
     * @return int[] numeric data
     */
    private static int[] tranformInstance(String[] row) {

        int[] vector = new int[row.length];
        for (int i = 0; i < row.length; ++i) {
            vector[i] = Integer.valueOf(row[i]);
        }

        return vector;
    }


    /**
     * Print out the classification
     */
    private static void printOut(int idx, Label yTrue, Label yPred) {
        System.out.printf("%d. Actual: %s - Classified: %s\n", idx, yTrue, yPred);
    }

    /**
     * Round up to the 3rd decimal point
     *
     * @param x double raw
     * @return double rounded number
     */
    private static double round3(double x) {
        return Math.round(x * 1000) / 1000.0;
    }

    /**
     * Evaluate the classification
     *
     * @param {Label[], Label[]}: ground truth and predicted list
     * @return {accuracy, precision, recall}
     */
    private static double[] evaluate(Label[] yTrue, Label[] yPred) {
        double[] scores = new double[3];
        double accuracy, precision, recall;
        double true_positive = 0, false_positive = 0, true_negative = 0, false_negative = 0;
        //precision  = #true_positive/ (#true_positive + #false_positive)
        //recall =  #true_positive/ (#true_positive + #false_negative)
        int pred_index = 0;
        for(Label actual : yTrue) {
            Label predicted = yPred[pred_index];
            switch(actual) {
                case Positive: //if the actual instance is positive
                    switch (predicted) {
                        case Positive:
                            true_positive++;
                        case Negative:
                            false_negative++;
                    }
                case Negative: //if the actual instance is negative
                    switch (predicted) {
                        case Positive:
                            false_positive++;
                        case Negative:
                            true_negative++;
                    }
            }
            pred_index++;
        }
        accuracy = (true_negative + true_positive)/yPred.length;
        precision = true_positive/(true_positive + false_positive);
        recall = true_positive/(true_positive+false_negative);
        scores[0] = accuracy;
        scores[1] = precision;
        scores[2] = recall;
        return scores;
    }

}
