# Entire file was written with help from GPT o1.
import re
import json


def tokenize(text):
    """Splits text into words, removing punctuation and converting to lowercase."""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def build_vocab_from_texts(texts):
    """
    Builds vocabulary from the given list of texts, returning both vocab and reverse_vocab.

    Args:
        texts (list of str): The input texts to build the vocabulary from.

    Returns:
        tuple: (vocab, reverse_vocab)
            vocab (dict): A dictionary mapping each word to a unique index.
            reverse_vocab (dict): A dictionary mapping each index back to its corresponding word.
    """
    tokens = []
    for text in texts:
        tokens.extend(tokenize(text))
    unique_tokens = sorted(set(tokens))
    vocab = {token: idx for idx, token in enumerate(unique_tokens)}
    vocab['<UNK>'] = len(vocab)  # Add unknown token
    reverse_vocab = {idx: token for token, idx in vocab.items()}
    return vocab, reverse_vocab


def load_vocab_from_file(filename):
    """Loads vocab and reverse_vocab from a JSON file."""
    with open(filename, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    reverse_vocab = {int(idx): token for token, idx in vocab.items()}
    return vocab, reverse_vocab


def save_vocab_to_file(vocab, filename):
    """Saves vocab dictionary to a file in JSON format."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(vocab, f)