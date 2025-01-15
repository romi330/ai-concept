# Written by RL.
from trainingloop import train_model_on_data
from tokenization import load_vocab_from_file
import tkinter as tk
from tkinter import simpledialog, messagebox
from simpledense import SimpleDense
from onehot import one_hot_encode
from load_interactions import load_interactions

def generate_response(sequence, network, vocab, reverse_vocab, max_length=4):
    tokens = tokenize(sequence)
    vocab_size = len(vocab)
    if not tokens:
        return "<UNK>"

    current_token = tokens[-1]

    generated = []
    for _ in range(max_length):
        token_id = vocab.get(current_token, vocab['<UNK>'])
        input_vector = one_hot_encode(current_token, vocab_size, token_id)
        outputs = network.forward(input_vector)

        predicted_index = outputs.index(max(outputs))
        current_token = reverse_vocab.get(predicted_index, "<UNK>")

        generated.append(current_token)

    return ' '.join(generated)

def tokenize(text):
    import re
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

if __name__ == "__main__":
    vocab_filename = 'vocab.json'
    vocab, reverse_vocab = load_vocab_from_file(vocab_filename)
    print(f"Vocabulary loaded from {vocab_filename}")

    interactions_file = 'interactions.csv'
    known_interactions = load_interactions(interactions_file)
    print("Loaded interactions:", known_interactions)

    network = SimpleDense(input_size=len(vocab), output_size=len(vocab))

    train_model_on_data(network, known_interactions, vocab, num_epochs=1000, learning_rate=0.1)

    root = tk.Tk()
    root.withdraw()
    prompt = simpledialog.askstring(prompt="Enter a phrase:", title="GPT Interaction")

    if prompt:
        response = generate_response(prompt, network, vocab, reverse_vocab)
        print("Generated sequence:", response)
        tk.messagebox.showinfo(title="Response", message=response)
    else:
        print("No input provided.")
