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

}
