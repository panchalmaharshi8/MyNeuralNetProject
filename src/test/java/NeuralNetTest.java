import static org.junit.Assert.*;
import org.junit.Test;
import Assignment_1.NeuralNet;

import java.sql.SQLOutput;
import java.util.Arrays;
import java.util.List;

public class NeuralNetTest {
    private NeuralNet createNeuralNet() {
        // List representing two hidden layers with 3 and 4 neurons respectively
        List<Integer> hiddenLayers = Arrays.asList(3, 5);
        return new NeuralNet(2, hiddenLayers, 1, 0, 1, 0.002);
    }

    @Test
    public void testWeightDimensions() {
        NeuralNet nn = createNeuralNet();

        List<double[][]> weights = nn.getWeights();

        // Check the first layer (2 inputs to 3 neurons)
        double[][] layer1Weights = weights.get(0);
        assertEquals(3, layer1Weights.length);
        for (double[] neuronWeights : layer1Weights) {
            assertEquals(3, neuronWeights.length); // 2 input connections + 1 bias connection
        }

        // Check the second layer (3 neurons to 4 neurons)
        double[][] layer2Weights = weights.get(1);
        assertEquals(5, layer2Weights.length);
        for (double[] neuronWeights : layer2Weights) {
            assertEquals(4, neuronWeights.length); // 3 neurons from the previous layer + 1 bias neuron
        }

        // Check the output layer (4 neurons to 1 output)
        double[][] outputLayerWeights = weights.get(2);
        assertEquals(1, outputLayerWeights.length);
        for (double[] neuronWeights : outputLayerWeights) {
            assertEquals(6, neuronWeights.length); // 5 neurons from the previous layer + 1 bias neuron
        }
    }

    @Test
    public void testBiases () {
        NeuralNet nn = createNeuralNet();

        double[] biases = nn.getBiases();
        assertEquals(3, biases.length);
        for (double bias : biases) {
            assertEquals(1.0, bias, 0.0001); // Allow a small delta for floating point comparison
        }
    }


}
