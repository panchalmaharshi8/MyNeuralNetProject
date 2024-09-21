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
    private List<double[]> biases;           // List of bias vectors for each layer

    // Constructor
    public NeuralNet(int numInputs, List<Integer> hiddenNeurons, int numOutputs, double lowerBound, double upperBound) {
        this.numInputs = numInputs;
        this.numHiddenLayers = hiddenNeurons.size();
        this.hiddenNeurons = hiddenNeurons;
        this.numOutputs = numOutputs;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.weights = new ArrayList<>();
        this.biases = new ArrayList<>();

        // Initialize weights and biases
        initializeWeightsAndBiases();
    }

    // Method to initialize weights and biases
    private void initializeWeightsAndBiases() {
        Random random = new Random();

        // Step 1: Initialize weights and biases between input layer and first hidden layer
        int previousLayerNeurons = numInputs;

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

            // Initialize bias vector for current hidden layer
            double[] layerBiases = new double[currentLayerNeurons];
            for (int j = 0; j < currentLayerNeurons; j++) {
                layerBiases[j] = random.nextDouble() * (upperBound - lowerBound) + lowerBound;
            }
            biases.add(layerBiases);

            // Update previousLayerNeurons for next layer
            previousLayerNeurons = currentLayerNeurons;
        }

        // Step 2: Initialize weights and biases between last hidden layer and output layer
        double[][] outputWeights = new double[numOutputs][previousLayerNeurons];
        for (int j = 0; j < numOutputs; j++) {
            for (int k = 0; k < previousLayerNeurons; k++) {
                outputWeights[j][k] = random.nextDouble() * (upperBound - lowerBound) + lowerBound;
            }
        }
        weights.add(outputWeights);

        // Initialize biases for output layer
        double[] outputBiases = new double[numOutputs];
        for (int j = 0; j < numOutputs; j++) {
            outputBiases[j] = random.nextDouble() * (upperBound - lowerBound) + lowerBound;
        }
        biases.add(outputBiases);
    }

    // Method to print weights and biases (for debugging)
    public void printWeightsAndBiases() {
        System.out.println("Weights:");
        for (int i = 0; i < weights.size(); i++) {
            double[][] layerWeights = weights.get(i);
            System.out.println("Layer " + i + " Weights:");
            for (double[] neuronWeights : layerWeights) {
                for (double weight : neuronWeights) {
                    System.out.printf("%.4f ", weight);
                }
                System.out.println();
            }
        }

        System.out.println("\nBiases:");
        for (int i = 0; i < biases.size(); i++) {
            double[] layerBiases = biases.get(i);
            System.out.println("Layer " + i + " Biases:");
            for (double bias : layerBiases) {
                System.out.printf("%.4f ", bias);
            }
            System.out.println();
        }
    }

    // Additional methods like forward pass, backpropagation, etc. would go here
}
