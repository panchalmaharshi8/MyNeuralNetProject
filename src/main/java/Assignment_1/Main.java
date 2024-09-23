package Assignment_1;

import java.io.IOException;

public class Main {

    public static void main(String[] args) {
        try {
            // Load the neural network and data from CSV
            NeuralNet nn = DataLoader.loadNeuralNetFromCSV("C:\\Users\\panch\\IdeaProjects\\CPEN502A1\\src\\main\\resources\\binary.csv");

            // Train the network until the error is below 0.05
            System.out.println("Training...");
            nn.trainUntilConverged(new double[]{0.0, 1.0}, new double[]{1.0}, 0.05);

            // Plot the error over epochs using the Plotter class
            Plotter.plotEpochErrors(nn.getEpochErrors());

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
