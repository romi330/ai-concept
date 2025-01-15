# Written with help from GPT o1.
def one_hot_encode(token, vocab_size, token_index):
    vector = [0] * vocab_size  # Initialize a vector of zeroes with the length of the vocabulary
    if token_index is not None and 0 <= token_index < vocab_size:
        vector[token_index] = 1  # Set the index corresponding to the token to 1
    else:
        vector[-1] = 1  # Assume last index is <UNK>
    return vector