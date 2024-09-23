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
    private double[] biases;                 // **One bias per layer** (adjusted)
    private double learningRate;
    private List<double[]> trainingInputs;
    private List<double[]> trainingOutputs;
    private List<Double> epochErrors;

    public NeuralNet(int numInputs, List<Integer> hiddenNeurons, int numOutputs, double lowerBound, double upperBound,
                     double learningRate) {
        this.numInputs = numInputs;
        this.numHiddenLayers = hiddenNeurons.size();
        this.hiddenNeurons = hiddenNeurons;
        this.numOutputs = numOutputs;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.weights = new ArrayList<>();
        this.biases = new double[hiddenNeurons.size() + 1];  // One bias per hidden layer + one for the output layer
        this.learningRate = learningRate;
        this.epochErrors = new ArrayList<>();

        // Initialize weights and biases
        initializeBiases();
        initializeWeights();
    }

    private void initializeBiases() {
        for (int i = 0; i < biases.length; i++) {
            biases[i] = 1.0;  // Set each bias (one per layer) to 1
        }
    }

    // Method to initialize weights and biases
    private void initializeWeights() {
        Random random = new Random();

        // Step 1: Initialize weights between input layer and first hidden layer
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

            // Update previousLayerNeurons for the next layer
            previousLayerNeurons = currentLayerNeurons;
        }

        // Step 2: Initialize weights between the last hidden layer and output layer
        double[][] outputWeights = new double[numOutputs][previousLayerNeurons];
        for (int j = 0; j < numOutputs; j++) {
            for (int k = 0; k < previousLayerNeurons; k++) {
                outputWeights[j][k] = random.nextDouble() * (upperBound - lowerBound) + lowerBound;
            }
        }
        weights.add(outputWeights);
    }

    // Set the training data
    public void setTrainingData(List<double[]> inputs, List<double[]> outputs) {
        this.trainingInputs = inputs;
        this.trainingOutputs = outputs;
    }

    // Sigmoid activation function
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    private double calculateTotalError(double[] output, double[] expectedOutput) {
        double totalError = 0;
        for (int i = 0; i < output.length; i++) {
            double error = expectedOutput[i] - output[i];  // Difference between expected and predicted
            totalError += error * error;  // Square the difference
        }
        return totalError;  // Sum of squared errors
    }

    public double[] forwardPropagation(double[] input) {
        double[] currentInput = input; // The input for the first layer is the initial input

        // Iterate through each layer
        for (int layer = 0; layer < weights.size(); layer++) {
            double[][] layerWeights = weights.get(layer);  // Get the weights for the current layer
            double[] newInput = new double[layerWeights.length]; // This will store the outputs for this layer

            // Calculate the output for each neuron in the layer
            for (int neuron = 0; neuron < layerWeights.length; neuron++) {
                double weightedSum = 0;

                // Calculate weighted sum: w1*x1 + w2*x2 + ... + bias (bias added once per layer)
                for (int previousNeuron = 0; previousNeuron < currentInput.length; previousNeuron++) {
                    weightedSum += currentInput[previousNeuron] * layerWeights[neuron][previousNeuron];
                }

                // Apply the bias for this layer uniformly to all neurons
                weightedSum += biases[layer];

                // Apply sigmoid activation
                newInput[neuron] = sigmoid(weightedSum);
            }

            // The output of this layer becomes the input for the next layer
            currentInput = newInput;
        }

        // Return the final output (from the last layer, i.e., the output layer)
        return currentInput;
    }

    public void backPropagation(double[] input, double[] expectedOutput) {
        double[] currentInput = input;  // Store the input to start propagation
        List<double[]> layerOutputs = new ArrayList<>(); // Store all layer outputs during forward pass

        // 1. Forward pass
        layerOutputs.add(currentInput); // Add the initial input as the first "layer output"
        for (int layer = 0; layer < weights.size(); layer++) {
            double[][] layerWeights = weights.get(layer);  // Get the weights for the current layer
            double[] newInput = new double[layerWeights.length]; // This will store the outputs for this layer

            // Calculate the output for each neuron in the layer
            for (int neuron = 0; neuron < layerWeights.length; neuron++) {
                double weightedSum = 0;
                for (int previousNeuron = 0; previousNeuron < currentInput.length; previousNeuron++) {
                    weightedSum += currentInput[previousNeuron] * layerWeights[neuron][previousNeuron];
                }

                // Apply the bias for this layer uniformly to all neurons
                weightedSum += biases[layer];

                // Apply sigmoid activation
                newInput[neuron] = sigmoid(weightedSum);
            }

            // Store the output of this layer and move to the next
            layerOutputs.add(newInput);
            currentInput = newInput;
        }

        // 2. Compute the error (Total Difference Error)
        double[] outputError = new double[expectedOutput.length];
        double[] finalOutput = layerOutputs.get(layerOutputs.size() - 1); // Last layer output
        for (int i = 0; i < expectedOutput.length; i++) {
            outputError[i] = expectedOutput[i] - finalOutput[i]; // Total Difference Error
        }

        // 3. Backpropagation: Update weights and biases layer by layer
        for (int layer = weights.size() - 1; layer >= 0; layer--) {
            double[] currentLayerOutput = layerOutputs.get(layer + 1); // Output of current layer
            double[] previousLayerOutput = layerOutputs.get(layer); // Output of previous layer (or input for layer 0)
            double[] layerDelta = new double[currentLayerOutput.length]; // Store deltas for the current layer

            // Calculate layer delta (how much to adjust each neuron's weights and biases)
            for (int neuron = 0; neuron < currentLayerOutput.length; neuron++) {
                double delta = outputError[neuron] * sigmoidDerivative(currentLayerOutput[neuron]); // delta = error * sigmoid'(output)
                layerDelta[neuron] = delta;

                // Update weights for the current neuron
                for (int prevNeuron = 0; prevNeuron < previousLayerOutput.length; prevNeuron++) {
                    weights.get(layer)[neuron][prevNeuron] += learningRate * delta * previousLayerOutput[prevNeuron]; // w = w + η * delta * input
                }
            }

            // Update the bias for the entire layer uniformly
            biases[layer] += learningRate * sum(layerDelta); // Sum the deltas for the bias update

            // Compute the error for the previous layer to propagate backward
            if (layer > 0) { // No need to propagate error beyond input layer
                double[] previousError = new double[previousLayerOutput.length];
                for (int prevNeuron = 0; prevNeuron < previousLayerOutput.length; prevNeuron++) {
                    double errorSum = 0;
                    for (int neuron = 0; neuron < currentLayerOutput.length; neuron++) {
                        errorSum += layerDelta[neuron] * weights.get(layer)[neuron][prevNeuron];
                    }
                    previousError[prevNeuron] = errorSum;
                }
                outputError = previousError; // Propagate this error to the next layer
            }
        }
    }

    private double sum(double[] values) {
        double total = 0;
        for (double v : values) {
            total += v;
        }
        return total;
    }

    public void trainUntilConverged(double[] input, double[] expectedOutput, double threshold) {
        double totalError;
        int iterations = 0;

        do {
            forwardPropagation(input);  // Forward propagate to get current output
            backPropagation(input, expectedOutput);  // Perform backpropagation and update weights
            totalError = calculateTotalError(forwardPropagation(input), expectedOutput);  // Recalculate error
            iterations++;
            System.out.println("Iteration " + iterations + ": Total Error = " + totalError);

            // Track error at this epoch
            epochErrors.add(totalError);

        } while (totalError >= threshold);  // Continue until total error is below the threshold
    }

    // Method to return the list of errors for plotting
    public List<Double> getEpochErrors() {
        return epochErrors;
    }

    public List<double[][]> getWeights() {
        return weights;
    }

    public double[] getBiases() {
        return biases;
    }
}
