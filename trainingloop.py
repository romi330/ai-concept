from tokenization import tokenize
from onehot import one_hot_encode

def train_model_on_data(network, interactions, vocab, num_epochs=1000, learning_rate=0.1):
    """
    Trains the neural network using provided interactions.

    Args:
        network (SimpleDense): The neural network to train.
        interactions (list of tuples): A list of (input, response) pairs as training data.
        vocab (dict): Vocabulary mapping words to indices.
        num_epochs (int): Number of iterations over the entire dataset.
        learning_rate (float): Learning rate for weight updates.
    """
    for epoch in range(num_epochs):
        total_loss = 0
        for input_text, response_text in interactions:
            # Tokenize input and response
            input_tokens = tokenize(input_text)
            response_tokens = tokenize(response_text)

            if not input_tokens or not response_tokens:
                continue  # Skip if either is empty

            # Simplistically use the first token of input and response
            input_token = input_tokens[0]
            response_token = response_tokens[0]

            # Encode input and target tokens
            input_index = vocab.get(input_token, vocab['<UNK>'])
            target_index = vocab.get(response_token, vocab['<UNK>'])

            input_vector = one_hot_encode(input_token, len(vocab), input_index)
            target_vector = one_hot_encode(response_token, len(vocab), target_index)

            # Forward pass
            outputs = network.forward(input_vector)

            # Compute loss (Mean Squared Error)
            loss = sum((o - t) ** 2 for o, t in zip(outputs, target_vector))
            total_loss += loss

            # Backward pass: compute gradients and update weights and biases
            # Gradient of loss w.r.t outputs
            gradients = [2 * (o - t) for o, t in zip(outputs, target_vector)]

            # Update weights and biases
            for i in range(len(network.weights)):
                for j in range(len(network.weights[i])):
                    network.weights[i][j] -= learning_rate * gradients[j] * input_vector[i]
            for j in range(len(network.biases)):
                network.biases[j] -= learning_rate * gradients[j]

        # Optionally print loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}')