import static org.junit.Assert.*;
import org.junit.Test;
import Assignment_1.NeuralNet;
import java.util.Arrays;
import java.util.List;

public class NeuralNetTest {
    private NeuralNet createNeuralNet() {
        // List representing two hidden layers with 3 and 4 neurons respectively
        List<Integer> hiddenLayers = Arrays.asList(3, 4);
        return new NeuralNet(2, hiddenLayers, 1, -0.5, 0.5);
    }

    @Test
    public void testWeightInitialization() {
        // Initialize the neural network
        NeuralNet nn = createNeuralNet();

        // Validate that weights were initialized
        assertNotNull(nn);  // Ensure neural network object isn't null

        // Check that weight matrix for first hidden layer (2 inputs, 3 neurons) is correct
        assertEquals(2, nn.getWeights().get(0).size());  // Should have 2 input connections
        assertEquals(3, nn.getWeights().get(0).get(0).length);  // Each input connects to 3 neurons

        // Check that weight matrix for second hidden layer (3 neurons -> 4 neurons) is correct
        assertEquals(3, nn.getWeights().get(1).size());  // 3 neurons from previous layer
        assertEquals(4, nn.getWeights().get(1).get(0).length);  // Each connects to 4 neurons
    }

    @Test
    public void testBiasInitialization() {
        // Initialize the neural network
        NeuralNet nn = createNeuralNet();

        // Validate that biases were initialized
        assertNotNull(nn);  // Ensure neural network object isn't null

        // Check biases for hidden layers (there should be biases for each neuron in each layer)
        List<List<Double>> biases = nn.getBiases();

        // For hidden layer 1 (3 neurons)
        assertEquals(3, biases.get(0).size());

        // For hidden layer 2 (4 neurons)
        assertEquals(4, biases.get(1).size());
    }

    @Test
    public void testForwardPassOutputSize() {
        // Initialize the neural network
        NeuralNet nn = createNeuralNet();

        // Input vector with 2 elements
        List<Double> input = Arrays.asList(0.5, -0.3);

        // Forward pass through the network
        List<Double> output = nn.forward(input);

        // Check that the output has the correct size (1 output neuron)
        assertEquals(1, output.size());
    }

    @Test
    public void testRandomWeightBounds() {
        // Initialize the neural network with bounds [-0.5, 0.5]
        NeuralNet nn = createNeuralNet();

        // Check that all weights are within the specified range
        List<List<double[]>> weights = nn.getWeights();

        for (List<double[]> layerWeights : weights) {
            for (double[] neuronWeights : layerWeights) {
                for (double weight : neuronWeights) {
                    assertTrue(weight >= -0.5 && weight <= 0.5);
                }
            }
        }
    }

    @Test
    public void testRandomBiasBounds() {
        // Initialize the neural network with bounds [-0.5, 0.5]
        NeuralNet nn = createNeuralNet();

        // Check that all biases are within the specified range
        List<List<Double>> biases = nn.getBiases();

        for (List<Double> layerBiases : biases) {
            for (double bias : layerBiases) {
                assertTrue(bias >= -0.5 && bias <= 0.5);
            }
        }
    }

}
