# Written by RL.
from tokenization import build_vocab_from_texts, save_vocab_to_file

def main():
    # Example data
    example_data = """
    The quick brown fox jumps over the lazy dog. The dog barks loudly.
    A silent fox moves quickly and jumps again. Foxes are quick and smart.
    """

    # Interaction data to be included in the vocabulary
    interaction_data = [
        "Hello",
        "Hi",
        "How are you?",
        "I am fine",
        "Goodbye",
        "Bye",
        "What is your name",
        "My name is GPT",
        "What does the fox say",
        "Ding",
        "Gibberish",
        "Response",
        "Gyatt",
        "rizz"
    ]

    # Combine all texts to build a comprehensive vocabulary
    all_texts = [example_data] + interaction_data

    # Build vocabulary
    vocab, reverse_vocab = build_vocab_from_texts(all_texts)

    # Save vocabulary to file
    save_vocab_to_file(vocab, 'vocab.json')
    print("Vocabulary built and saved to vocab.json")

if __name__ == "__main__":
    main()