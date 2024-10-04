package Assignment_1;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {

    public static void main(String[] args) {
        try {
            int totalTrials = 50;
//            String fileName = "C:\\Users\\panch\\IdeaProjects\\CPEN502A1\\src\\main\\resources\\binary.csv";
            String fileName = "C:\\Users\\panch\\IdeaProjects\\CPEN502A1\\src\\main\\resources\\bipolar.csv";
            List<Integer> epochCounts = new ArrayList<>();
            NeuralNet lastNN = null;
            for (int trial = 1; trial <= totalTrials; trial++){
//                System.out.println("Trial" + trial);
                NeuralNet nn = DataLoader.loadNeuralNetFromCSV(fileName);

                int epochs = nn.trainUntilConverged(0.05);
                epochCounts.add(epochs);

//                System.out.println("Trial " + trial + ": Converged in " + epochs + " epochs.");

                lastNN = nn;
            }

            // Calculate the average number of epochs across all trials
            double averageEpochs = epochCounts.stream().mapToInt(Integer::intValue).average().orElse(0);
            System.out.println("Average epochs to converge: " + averageEpochs);

            if (lastNN != null) {
                Plotter.plotEpochErrors(lastNN.getEpochErrors());
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
