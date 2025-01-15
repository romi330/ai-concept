import csv

def load_interactions(file_path):
    interactions = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header if present
        for row in csv_reader:
            interactions.append((row[0], row[1]))
    return interactions