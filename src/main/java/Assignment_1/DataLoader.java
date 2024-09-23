package Assignment_1;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataLoader {

    // Method to load training data from a CSV file in the resources folder
    public static NeuralNet loadNeuralNetFromCSV(String filename) throws IOException {

        BufferedReader br = new BufferedReader(new FileReader(filename));

        // Parse file data (this part remains unchanged)
        String[] hiddenNeuronsStr = br.readLine().split(",");
        List<Integer> hiddenNeurons = new ArrayList<>();
        for (String s : hiddenNeuronsStr) {
            hiddenNeurons.add(Integer.parseInt(s.trim()));
        }

        int numInputs = Integer.parseInt(br.readLine().trim());
        int numOutputs = Integer.parseInt(br.readLine().trim());

        List<double[]> inputs = new ArrayList<>();
        List<double[]> outputs = new ArrayList<>();
        String line;
        while ((line = br.readLine()) != null) {
            String[] tokens = line.split(",");
            double[] input = new double[numInputs];
            double[] output = new double[numOutputs];

            for (int i = 0; i < numInputs; i++) {
                input[i] = Double.parseDouble(tokens[i].trim());
            }
            for (int i = 0; i < numOutputs; i++) {
                output[i] = Double.parseDouble(tokens[numInputs + i].trim());
            }

            inputs.add(input);
            outputs.add(output);
        }
        br.close();

        NeuralNet neuralNet = new NeuralNet(numInputs, hiddenNeurons, numOutputs, -0.5, 0.5, 0.2);
        neuralNet.setTrainingData(inputs, outputs);
        return neuralNet;
    }
}
