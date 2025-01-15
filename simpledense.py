# Written with help from GPT o1.
import random

class SimpleDense:
    def __init__(self, input_size, output_size):
        # Initialize weights with small random numbers
        self.weights = [[random.uniform(-0.1, 0.1) for _ in range(output_size)] for _ in range(input_size)]
        # Initialize biases with small random numbers
        self.biases = [random.uniform(-0.1, 0.1) for _ in range(output_size)]

    def forward(self, inputs):
        """
        Forward pass through the network.

        Args:
            inputs (list of float): One-hot encoded input vector.

        Returns:
            list of float: Output vector.
        """
        outputs = []
        for j in range(len(self.biases)):
            # Compute weighted sum + bias for each output neuron
            neuron_output = self.biases[j]
            for i in range(len(inputs)):
                neuron_output += inputs[i] * self.weights[i][j]
            # Apply activation function (e.g., sigmoid for probabilities)
            neuron_output = self.sigmoid(neuron_output)
            outputs.append(neuron_output)
        return outputs

    def sigmoid(self, x):
        """Sigmoid activation function."""
        import math
        return 1 / (1 + math.exp(-x))