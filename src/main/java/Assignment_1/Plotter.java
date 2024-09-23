package Assignment_1;

import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.markers.None;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Plotter {

    public static void plotEpochErrors(List<Double> epochErrors) {
        // Generate x-axis (epoch numbers)
        List<Integer> epochs = IntStream.range(0, epochErrors.size())
                .boxed()
                .collect(Collectors.toList());

        // Create Chart
        XYChart chart = new XYChartBuilder().width(800).height(600).title("Error over Epochs").xAxisTitle("Epochs").yAxisTitle("Total Error").build();

        // Customize Chart
        chart.getStyler().setMarkerSize(4);
        chart.getStyler().setXAxisMin(0.0);
        chart.getStyler().setYAxisMin(0.0);
        chart.getStyler().setLegendVisible(false);

        // Add data to the chart
        chart.addSeries("Total Error", epochs, epochErrors).setMarker(new None());

        // Show it
        new SwingWrapper<>(chart).displayChart();
    }
}


