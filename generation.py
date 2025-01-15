# Written with help from GPT o1.
from onehot import one_hot_encode

def predict_next_word(input_token, network, vocab, reverse_vocab):
    """Predicts the next word given an input token."""
    vocab_size = len(vocab)
    if input_token not in vocab:
        raise ValueError("Input token not in vocabulary.")

    # Convert input token to one-hot vector
    input_vector = one_hot_encode(input_token, vocab_size, vocab[input_token])

    # Forward pass through the network
    output_logits = network.forward(input_vector)

    # Choose the highest scoring logit as the prediction
    predicted_index = output_logits.index(max(output_logits))
    predicted_token = reverse_vocab[predicted_index]

    return predicted_token

def generate_sequence(start_word, length, network, vocab, reverse_vocab):
    """Generates a sequence of words starting with the start_word."""
    generated = [start_word]

    current_word = start_word
    for _ in range(length - 1):
        next_word = predict_next_word(current_word, network, vocab, reverse_vocab)
        generated.append(next_word)
        current_word = next_word

    return " ".join(generated)
