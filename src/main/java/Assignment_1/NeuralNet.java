package Assignment_1;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNet {
    private int numInputs;                   // Number of input neurons
    private int numHiddenLayers;             // Number of hidden layers
    private List<Integer> hiddenNeurons;     // List containing number of neurons in each hidden layer
    private int numOutputs;                  // Number of output neurons
    private double lowerBound;               // Lower bound for random weight initialization
    private double upperBound;               // Upper bound for random weight initialization
    private List<double[][]> weights;        // List of weight matrices between layers

    public NeuralNet(int numInputs, List<Integer> hiddenNeurons, int numOutputs, double lowerBound, double upperBound) {
        this.numInputs = numInputs;
        this.numHiddenLayers = hiddenNeurons.size();
        this.hiddenNeurons = hiddenNeurons;
        this.numOutputs = numOutputs;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.weights = new ArrayList<>();

        // Initialize weights and biases
        initializeWeights();
    }

    // Method to initialize weights and biases
    private void initializeWeights() {
        Random random = new Random();

        // Step 1: Initialize weights and biases between input layer and first hidden layer
        int previousLayerNeurons = numInputs + 1;

        // Loop through each hidden layer
        for (int i = 0; i < numHiddenLayers; i++) {
            int currentLayerNeurons = hiddenNeurons.get(i);

            // Initialize weight matrix between previous layer and current hidden layer
            double[][] layerWeights = new double[currentLayerNeurons][previousLayerNeurons];
            for (int j = 0; j < currentLayerNeurons; j++) {
                for (int k = 0; k < previousLayerNeurons; k++) {
                    layerWeights[j][k] = random.nextDouble() * (upperBound - lowerBound) + lowerBound;
                }
            }
            weights.add(layerWeights);

            // Update previousLayerNeurons for next layer
            previousLayerNeurons = currentLayerNeurons + 1;
        }

        // Step 2: Initialize weights and biases between last hidden layer and output layer
        double[][] outputWeights = new double[numOutputs][previousLayerNeurons];
        for (int j = 0; j < numOutputs; j++) {
            for (int k = 0; k < previousLayerNeurons; k++) {
                outputWeights[j][k] = random.nextDouble() * (upperBound - lowerBound) + lowerBound;
            }
        }
        weights.add(outputWeights);

    }

    public List<double[][]> getWeights (){
        return weights;
    }

    // A method to simulate the forward pass (for completeness)
    public List<Double> forward(List<Double> inputs) {
        // Placeholder: Implement the forward pass logic if not done yet
        return new ArrayList<>();  // Return dummy output for now
    }

}


