import csv
import pdb
from itertools import combinations

def hamming_distance(str1, str2):
    """Computes the Hamming distance between two equal-length strings."""
    if len(str1) != len(str2):
        raise ValueError("Sequences must be of equal length to compute Hamming distance.")
    return sum(el1 != el2 for el1, el2 in zip(str1, str2))

def read_csv(file_path):
    """Reads a CSV file and returns the rows as a list of strings."""
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        return [row for row in reader]  # Concatenate row elements into a single string

def compute_hamming_distances(file_path):
    """Computes and prints the Hamming distance between all pairs of lines in a CSV file."""
    lines = read_csv(file_path)
    distances = []
    for (i, line1), (j, line2) in combinations(enumerate(lines), 2):
        try:
            distance = hamming_distance(line1, line2)
            distances.append(distance)
            print(f"Hamming distance between line {i+1} and line {j+1}: {distance}")
            #pdb.set_trace()
        except ValueError as e:
            print(f"Skipping comparison between line {i+1} and line {j+1}: {e}")
    avg = sum(distances)/len(distances)
    print(f"Average Hamming distance: {avg/50}")

# Example usage
file_path = "test/results/encoded_traces/encoded_traces_03-17-15-49/ID_1_run_1_ProblemSingle.csv"  # Replace with the actual file path ID_5_run_1_single_yes_constraints.csv ID_1_run_1_single_yes_constraints
compute_hamming_distances(file_path)


