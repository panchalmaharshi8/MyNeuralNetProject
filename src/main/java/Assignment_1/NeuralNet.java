package Assignment_1;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNet {
    private int numInputs;
    private int numHiddenLayers;
    private List<Integer> hiddenNeurons;
    private int numOutputs;
    private double lowerBound;
    private double upperBound;
    private List<double[][]> weights;
    private double[] biases;
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

        initializeBiases();
        initializeWeights();
    }

    private void initializeBiases() {
        for (int i = 0; i < biases.length; i++) {
            biases[i] = 1.0;  // Set each bias (one per layer) to 1
        }
    }

    private void initializeWeights() {
        Random random = new Random();

        int previousLayerNeurons = numInputs;

        for (int i = 0; i < numHiddenLayers; i++) {
            int currentLayerNeurons = hiddenNeurons.get(i);

            // weight matrix num curr layer neurons x num prev layer neurons
            double[][] layerWeights = new double[currentLayerNeurons][previousLayerNeurons];
            for (int j = 0; j < currentLayerNeurons; j++) {
                for (int k = 0; k < previousLayerNeurons; k++) {
                    layerWeights[j][k] = random.nextDouble() * (upperBound - lowerBound) + lowerBound;
                }
            }
            weights.add(layerWeights);

            previousLayerNeurons = currentLayerNeurons;
        }

        // For last layer and output:
        double[][] outputWeights = new double[numOutputs][previousLayerNeurons];
        for (int j = 0; j < numOutputs; j++) {
            for (int k = 0; k < previousLayerNeurons; k++) {
                outputWeights[j][k] = random.nextDouble() * (upperBound - lowerBound) + lowerBound;
            }
        }
        weights.add(outputWeights);
    }

    public void setTrainingData(List<double[]> inputs, List<double[]> outputs) {
        this.trainingInputs = inputs;
        this.trainingOutputs = outputs;
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    private double calculateTotalError(double[] output, double[] expectedOutput) {
        double totalError = 0;
        for (int i = 0; i < output.length; i++) {
            double error = expectedOutput[i] - output[i];
            totalError += error * error;
        }
        return totalError;
    }

    public double[] forwardPropagation(double[] input) {
        double[] currentInput = input;

        for (int layer = 0; layer < weights.size(); layer++) {
            double[][] layerWeights = weights.get(layer);
            double[] newInput = new double[layerWeights.length];

            for (int neuron = 0; neuron < layerWeights.length; neuron++) {
                double weightedSum = 0;

                for (int previousNeuron = 0; previousNeuron < currentInput.length; previousNeuron++) {
                    weightedSum += currentInput[previousNeuron] * layerWeights[neuron][previousNeuron];
                }

                weightedSum += biases[layer];

                newInput[neuron] = sigmoid(weightedSum);
            }

            currentInput = newInput;
        }

        return currentInput;
    }

    public void backPropagation(double[] input, double[] expectedOutput) {
        double[] currentInput = input;
        List<double[]> layerOutputs = new ArrayList<>();

        // 1. Forward pass
        layerOutputs.add(currentInput);
        for (int layer = 0; layer < weights.size(); layer++) {
            double[][] layerWeights = weights.get(layer);
            double[] newInput = new double[layerWeights.length];

            for (int neuron = 0; neuron < layerWeights.length; neuron++) {
                double weightedSum = 0;
                for (int previousNeuron = 0; previousNeuron < currentInput.length; previousNeuron++) {
                    weightedSum += currentInput[previousNeuron] * layerWeights[neuron][previousNeuron];
                }

                weightedSum += biases[layer];

                newInput[neuron] = sigmoid(weightedSum);
            }

            layerOutputs.add(newInput);
            currentInput = newInput;
        }

        // 2. Compute the error (Total Difference Error)
        double[] outputError = new double[expectedOutput.length];
        double[] finalOutput = layerOutputs.get(layerOutputs.size() - 1); // Last layer output
        for (int i = 0; i < expectedOutput.length; i++) {
            outputError[i] = expectedOutput[i] - finalOutput[i]; // Total Difference Error
        }

        // 3. Backpropagation:
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
                    weights.get(layer)[neuron][prevNeuron] += learningRate * delta * previousLayerOutput[prevNeuron];
                }
            }

            // Update the bias for the entire layer uniformly
            biases[layer] += learningRate * sum(layerDelta);

            if (layer > 0) {
                double[] previousError = new double[previousLayerOutput.length];
                for (int prevNeuron = 0; prevNeuron < previousLayerOutput.length; prevNeuron++) {
                    double errorSum = 0;
                    for (int neuron = 0; neuron < currentLayerOutput.length; neuron++) {
                        errorSum += layerDelta[neuron] * weights.get(layer)[neuron][prevNeuron];
                    }
                    previousError[prevNeuron] = errorSum;
                }
                outputError = previousError;
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

    public int trainUntilConverged(double threshold) {
        double totalError;
        int iterations = 0;

        do {
            totalError = 0;
            for (int i = 0; i < trainingInputs.size(); i++) {
                double[] input = trainingInputs.get(i);
                double[] expectedOutput = trainingOutputs.get(i);

                forwardPropagation(input);
                backPropagation(input, expectedOutput);
                totalError += calculateTotalError(forwardPropagation(input), expectedOutput);
            }

            epochErrors.add(totalError);
            iterations++;
            System.out.println("Epoch " + iterations + ": Total Error = " + totalError);

        } while (totalError >= threshold && iterations <=6000); //REMEMBER TO GET RID OF THE ITERATIONS THINGY JUST HERE FOR DEBUG

        return iterations;
    }

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
