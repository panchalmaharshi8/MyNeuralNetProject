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
    private double learningRate;
    private List<double[]> trainingInputs;
    private List<double[]> trainingOutputs;
    private List<Double> epochErrors;
    private List<double[][]> previousWeightUpdates;  // Adding momentum

    public NeuralNet(int numInputs, List<Integer> hiddenNeurons, int numOutputs, double lowerBound, double upperBound,
                     double learningRate) {
        this.numInputs = numInputs + 1;  // +1 for bias input (added to every layer)
        this.numHiddenLayers = hiddenNeurons.size();
        this.hiddenNeurons = hiddenNeurons;
        this.numOutputs = numOutputs;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.weights = new ArrayList<>();
        this.learningRate = learningRate;
        this.epochErrors = new ArrayList<>();
        this.previousWeightUpdates = new ArrayList<>();

        initializeWeights();
    }

    private void initializeWeights() {
        Random random = new Random();

        int previousLayerNeurons = numInputs;

        for (int i = 0; i < numHiddenLayers; i++) {
            int currentLayerNeurons = hiddenNeurons.get(i);

            // Initialize weight matrix and previous weight update matrix
            double[][] layerWeights = new double[currentLayerNeurons][previousLayerNeurons];
            double[][] layerPreviousUpdates = new double[currentLayerNeurons][previousLayerNeurons];  // For momentum
            for (int j = 0; j < currentLayerNeurons; j++) {
                for (int k = 0; k < previousLayerNeurons; k++) {
                    layerWeights[j][k] = random.nextDouble() * (upperBound - lowerBound) + lowerBound;
                    layerPreviousUpdates[j][k] = 0.0;  // Initialize previous updates to 0
                }
            }
            weights.add(layerWeights);
            previousWeightUpdates.add(layerPreviousUpdates);  // Add the previous weight updates respectively
            previousLayerNeurons = currentLayerNeurons + 1;  // +1 for the bias input in the next layer
        }

        // For the last layer and output:
        double[][] outputWeights = new double[numOutputs][previousLayerNeurons];
        double[][] outputPreviousUpdates = new double[numOutputs][previousLayerNeurons];  // Adding for momentum
        for (int j = 0; j < numOutputs; j++) {
            for (int k = 0; k < previousLayerNeurons; k++) {
                outputWeights[j][k] = random.nextDouble() * (upperBound - lowerBound) + lowerBound;
                outputPreviousUpdates[j][k] = 0.0;  // Initialize previous updates to 0
            }
        }
        weights.add(outputWeights);
        previousWeightUpdates.add(outputPreviousUpdates);  // Add previous weight updates for the output layer
    }

    public void setTrainingData(List<double[]> inputs, List<double[]> outputs) {
        this.trainingInputs = inputs;
        this.trainingOutputs = outputs;
    }

    //Unused at the moment, tanh for bipolar
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    //Always feeding in the sigmoid(x) as x here!
    private double sigmoidDerivative(double x) {
        return x*(1-x);
    }

    //Use this for bipolar!
    private double tanh(double x) {
        return Math.tanh(x);
    }

    private double tanhDerivative(double x) {
        return 1 - Math.pow(Math.tanh(x), 2);
    }

    private double calculateTotalError(double[] output, double[] expectedOutput) {
        double totalError = 0;
        for (int i = 0; i < output.length; i++) {
            double error = expectedOutput[i] - output[i];
            totalError += error * error;
        }
        return totalError / 2;
    }

    public double[] forwardPropagation(double[] input) {
        // Add the bias to the input (hardcoded to 1)
        double[] currentInput = new double[input.length + 1];
        System.arraycopy(input, 0, currentInput, 0, input.length);
        currentInput[input.length] = 1.0;  // Bias input

        // Iterate through each hidden layer
        for (int layer = 0; layer < weights.size() - 1; layer++) {
            double[][] layerWeights = weights.get(layer);
            double[] newInput = new double[layerWeights.length];

            // Calculate the output for each neuron in the layer
            for (int neuron = 0; neuron < layerWeights.length; neuron++) {
                double weightedSum = 0;

                // Calculate weighted sum: w1*x1 + w2*x2 + ... + bias weight
                for (int previousNeuron = 0; previousNeuron < currentInput.length; previousNeuron++) {
                    weightedSum += currentInput[previousNeuron] * layerWeights[neuron][previousNeuron];
                }

                // Apply sigmoid activation
                newInput[neuron] = tanh(weightedSum);

                // Check if neurons are too saturated
                if (newInput[neuron] < 0.1 || newInput[neuron] > 0.9) {
//                    System.out.println("Neuron " + neuron + " in Layer " + layer + " is saturated with activation: " + newInput[neuron]);
                }
            }

            // Add bias input for the next layer
            currentInput = new double[newInput.length + 1];
            System.arraycopy(newInput, 0, currentInput, 0, newInput.length);
            currentInput[newInput.length] = 1.0;  // Bias input for the next layer
        }

        // Output layer (no bias input added here)
        double[][] outputWeights = weights.get(weights.size() - 1);
        double[] output = new double[outputWeights.length];

        // Calculate output for each neuron in the output layer
        for (int neuron = 0; neuron < outputWeights.length; neuron++) {
            double weightedSum = 0;

            for (int previousNeuron = 0; previousNeuron < currentInput.length; previousNeuron++) {
                weightedSum += currentInput[previousNeuron] * outputWeights[neuron][previousNeuron];
            }

            output[neuron] = tanh(weightedSum);  // Final output
        }

        return output;  // Return the output without bias
    }

    public void backPropagationMomentum(double[] input, double[] expectedOutput) {
        double momentum = 0.9;  // Momentum coefficient

        // Forward propagation to store the outputs of each layer
        double[] currentInput = new double[input.length + 1];
        System.arraycopy(input, 0, currentInput, 0, input.length);
        currentInput[input.length] = 1.0;  // Bias input

        List<double[]> layerOutputs = new ArrayList<>();
        layerOutputs.add(currentInput);

        // Forward pass to get the outputs at each layer
        for (int layer = 0; layer < weights.size(); layer++) {
            double[][] layerWeights = weights.get(layer);
            double[] newInput = new double[layerWeights.length];

            for (int neuron = 0; neuron < layerWeights.length; neuron++) {
                double weightedSum = 0;
                for (int previousNeuron = 0; previousNeuron < currentInput.length; previousNeuron++) {
                    weightedSum += currentInput[previousNeuron] * layerWeights[neuron][previousNeuron];
                }

                newInput[neuron] = tanh(weightedSum);
            }

            // Bias input for the next layer
            currentInput = new double[newInput.length + 1];
            System.arraycopy(newInput, 0, currentInput, 0, newInput.length);
            currentInput[newInput.length] = 1.0;  // Bias input
            layerOutputs.add(currentInput);
        }

        // Output layer error and delta calculation
        double[] outputError = new double[expectedOutput.length];
        double[] finalOutput = layerOutputs.get(layerOutputs.size() - 1);  // Last layer output (without bias)

        // Calculate output error and delta for the output layer
        for (int i = 0; i < expectedOutput.length; i++) {
            outputError[i] = expectedOutput[i] - finalOutput[i];
        }

        // Step 1: Compute delta for the output layer first
        double[][] outputWeights = weights.get(weights.size() - 1);
        double[][] outputPreviousUpdates = previousWeightUpdates.get(previousWeightUpdates.size() - 1);  // Previous updates for momentum
        double[] outputLayerDelta = new double[outputWeights.length];

        for (int neuron = 0; neuron < outputWeights.length; neuron++) {
            outputLayerDelta[neuron] = outputError[neuron] * tanhDerivative(finalOutput[neuron]);
        }

        // Step 2: Update the weights for the hidden-to-output layer (with momentum)
        double[] hiddenLayerOutput = layerOutputs.get(layerOutputs.size() - 2);  // Second-to-last layer's output
        for (int neuron = 0; neuron < outputWeights.length; neuron++) {
            for (int prevNeuron = 0; prevNeuron < hiddenLayerOutput.length; prevNeuron++) {
                // Apply momentum to the weight update
                double deltaWeight = learningRate * outputLayerDelta[neuron] * hiddenLayerOutput[prevNeuron]
                        + momentum * outputPreviousUpdates[neuron][prevNeuron];
                outputWeights[neuron][prevNeuron] += deltaWeight;

                // Store the delta weight for use in the next epoch
                outputPreviousUpdates[neuron][prevNeuron] = deltaWeight;
            }
        }

        // Step 3: Compute delta for the hidden layer using the updated output layer weights
        double[][] hiddenWeights = weights.get(weights.size() - 2);
        double[][] hiddenPreviousUpdates = previousWeightUpdates.get(previousWeightUpdates.size() - 2);  // Previous updates for momentum
        double[] hiddenLayerDelta = new double[hiddenWeights.length];

        for (int neuron = 0; neuron < hiddenWeights.length; neuron++) {
            double deltaSum = 0;
            for (int outputNeuron = 0; outputNeuron < outputLayerDelta.length; outputNeuron++) {
                deltaSum += outputLayerDelta[outputNeuron] * outputWeights[outputNeuron][neuron];
            }
            hiddenLayerDelta[neuron] = deltaSum * tanhDerivative(hiddenLayerOutput[neuron]);
        }

        // Step 4: Update the weights for the input-to-hidden layer (with momentum)
        double[] inputLayerOutput = layerOutputs.get(0);  // First layer (input + bias)
        for (int neuron = 0; neuron < hiddenWeights.length; neuron++) {
            for (int prevNeuron = 0; prevNeuron < inputLayerOutput.length; prevNeuron++) {
                // Apply momentum to the weight update
                double deltaWeight = learningRate * hiddenLayerDelta[neuron] * inputLayerOutput[prevNeuron]
                        + momentum * hiddenPreviousUpdates[neuron][prevNeuron];
                hiddenWeights[neuron][prevNeuron] += deltaWeight;

                // Store the delta weight for use in the next epoch
                hiddenPreviousUpdates[neuron][prevNeuron] = deltaWeight;
            }
        }
    }


    public void backPropagation(double[] input, double[] expectedOutput) {
        // Forward propagation to store the outputs of each layer
        double[] currentInput = new double[input.length + 1];
        System.arraycopy(input, 0, currentInput, 0, input.length);
        currentInput[input.length] = 1.0;  // Bias input

        List<double[]> layerOutputs = new ArrayList<>();
        layerOutputs.add(currentInput);

        // Forward pass to get the outputs at each layer
        for (int layer = 0; layer < weights.size(); layer++) {
            double[][] layerWeights = weights.get(layer);
            double[] newInput = new double[layerWeights.length];

            for (int neuron = 0; neuron < layerWeights.length; neuron++) {
                double weightedSum = 0;
                for (int previousNeuron = 0; previousNeuron < currentInput.length; previousNeuron++) {
                    weightedSum += currentInput[previousNeuron] * layerWeights[neuron][previousNeuron];
                }

                newInput[neuron] = tanh(weightedSum);
            }

            // Bias input for the next layer
            currentInput = new double[newInput.length + 1];
            System.arraycopy(newInput, 0, currentInput, 0, newInput.length);
            currentInput[newInput.length] = 1.0;  // Bias input
            layerOutputs.add(currentInput);
        }

        // Output layer error and delta calculation
        double[] outputError = new double[expectedOutput.length];
        double[] finalOutput = layerOutputs.get(layerOutputs.size() - 1);  // Last layer output (without bias)

        // Calculate output error and delta for the output layer
        for (int i = 0; i < expectedOutput.length; i++) {
            outputError[i] = expectedOutput[i] - finalOutput[i];
        }

        // Step 1: Compute delta for the output layer first
        double[][] outputWeights = weights.get(weights.size() - 1);
        double[] outputLayerDelta = new double[outputWeights.length];

        for (int neuron = 0; neuron < outputWeights.length; neuron++) {
            outputLayerDelta[neuron] = outputError[neuron] * tanhDerivative(finalOutput[neuron]);
        }

        // Step 2: Update the weights for the hidden-to-output layer
        double[] hiddenLayerOutput = layerOutputs.get(layerOutputs.size() - 2);  // Second-to-last layer's output
        for (int neuron = 0; neuron < outputWeights.length; neuron++) {
            for (int prevNeuron = 0; prevNeuron < hiddenLayerOutput.length; prevNeuron++) {
                outputWeights[neuron][prevNeuron] += learningRate * outputLayerDelta[neuron] * hiddenLayerOutput[prevNeuron];
            }
        }

        // Step 3: Compute delta for the hidden layer using the updated output layer weights
        double[][] hiddenWeights = weights.get(weights.size() - 2);
        double[] hiddenLayerDelta = new double[hiddenWeights.length];

        for (int neuron = 0; neuron < hiddenWeights.length; neuron++) {
            double deltaSum = 0;
            for (int outputNeuron = 0; outputNeuron < outputLayerDelta.length; outputNeuron++) {
                deltaSum += outputLayerDelta[outputNeuron] * outputWeights[outputNeuron][neuron];
            }
            hiddenLayerDelta[neuron] = deltaSum * tanhDerivative(hiddenLayerOutput[neuron]);
        }

        // Step 4: Update the weights for the input-to-hidden layer
        double[] inputLayerOutput = layerOutputs.get(0);  // First layer (input + bias)
        for (int neuron = 0; neuron < hiddenWeights.length; neuron++) {
            for (int prevNeuron = 0; prevNeuron < inputLayerOutput.length; prevNeuron++) {
                hiddenWeights[neuron][prevNeuron] += learningRate * hiddenLayerDelta[neuron] * inputLayerOutput[prevNeuron];
            }
        }
    }

    public int trainUntilConverged(double threshold) {
        double totalError;
        int iterations = 0;

        do {
            totalError = 0;
            for (int i = 0; i < trainingInputs.size(); i++) {
                double[] input = trainingInputs.get(i);
                double[] expectedOutput = trainingOutputs.get(i);

                double[] output = forwardPropagation(input);
                backPropagationMomentum(input, expectedOutput);
                totalError += calculateTotalError(output, expectedOutput);
            }

            epochErrors.add(totalError);
            iterations++;
            System.out.println("Epoch " + iterations + ": Total Error = " + totalError);

        } while (totalError >= threshold);

        return iterations;
    }

    public List<Double> getEpochErrors() {
        return epochErrors;
    }

    public List<double[][]> getWeights() {
        return weights;
    }
}
