import numpy as np
import csv
from collections import defaultdict
import sys
import os

# Add the path to the directory containing your prototype network script
sys.path.insert(0, os.path.abspath('../path/to/your/script/directory'))

# Import the necessary functions from your existing script
from eventlevelprotonet import (
    ProtoNet, get_contextual_embedding, compute_prototypes, 
    create_episode, train_proto_net, validate_proto_net, 
    classify_new_events, proto_net_BC, proto_net_PS, 
    prototype_tensor_BC, prototype_tensor_PS
)

# Define the cognitive vectors (adjust as needed based on your schema)
cog_vectors = {'BP': 0, 'BS': 1, 'CP': 2, 'CS': 3}
NUM_COG_VECTORS = len(cog_vectors)

def read_csv_life_events(file_path):
    events = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            date = row.get('Date', '')
            age = row.get('Age', '')
            event = row.get('Life Event', '')
            if event:  # Only add non-empty events
                events.append({
                    'date': date,
                    'age': age,
                    'event': event
                })
    return events

def process_and_update_matrices(events, transition_matrix):
    time_series_matrix = []
    
    # Classify events using your prototype network
    event_texts = [event['event'] for event in events]
    predicted_labels_BC, predicted_labels_PS = classify_new_events(
        event_texts, proto_net_BC, proto_net_PS, prototype_tensor_BC, prototype_tensor_PS
    )
    
    combined_states = combine_predictions(predicted_labels_BC, predicted_labels_PS)
    
    for i in range(len(events) - 1):
        prev_state = combined_states[i]
        next_state = combined_states[i + 1]
        
        prev_vector = cog_vectors[prev_state]
        next_vector = cog_vectors[next_state]
        
        transition_matrix[prev_vector][next_vector] += 1
        
        time_series_matrix.append(transition_matrix.copy())
    
    return time_series_matrix, transition_matrix

def combine_predictions(predicted_labels_BC, predicted_labels_PS):
    bc_ps_to_state = {
        (0, 0): 'BP',  # (Blast, Play)
        (0, 1): 'BS',  # (Blast, Sleep)
        (1, 0): 'CP',  # (Consume, Play)
        (1, 1): 'CS',  # (Consume, Sleep)
    }
    return [bc_ps_to_state.get((bc, ps), 'Unknown') for bc, ps in zip(predicted_labels_BC, predicted_labels_PS)]

def normalize_matrix(matrix):
    row_sums = matrix.sum(axis=1)
    return matrix / row_sums[:, np.newaxis]

def find_stationary_distribution(transition_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary = eigenvectors[:, np.isclose(eigenvalues, 1)]
    stationary_distribution = stationary / stationary.sum()
    return stationary_distribution.real.flatten()

def entropy_rate(transition_matrix):
    stationary_dist = find_stationary_distribution(transition_matrix)
    entropy_rate = -np.sum(stationary_dist * np.sum(transition_matrix * np.log(transition_matrix + 1e-10), axis=1))
    return entropy_rate

def mean_first_passage_time(transition_matrix):
    n = transition_matrix.shape[0]
    Z = np.linalg.inv(np.eye(n) - transition_matrix + np.ones((n, n)) / n)
    D = np.diag(np.diag(Z))
    return (np.eye(n) - Z + np.ones((n, n)) * D) / np.diag(Z)

def analyze_killer(file_path, killer_name):
    # Initialize transition matrix
    transition_matrix = np.zeros((NUM_COG_VECTORS, NUM_COG_VECTORS))

    # Read and process events
    events = read_csv_life_events(file_path)
    time_series_matrix, final_matrix = process_and_update_matrices(events, transition_matrix)

    # Normalize the final matrix
    normalized_matrix = normalize_matrix(final_matrix)

    # Save individual killer's matrix
    np.save(f'transition_matrix_{killer_name}.npy', normalized_matrix)

    # Perform analysis
    stationary_dist = find_stationary_distribution(normalized_matrix)
    entropy = entropy_rate(normalized_matrix)
    mfpt = mean_first_passage_time(normalized_matrix)

    print(f"Analysis for {killer_name}:")
    print(f"Stationary Distribution: {stationary_dist}")
    print(f"Entropy Rate: {entropy}")
    print(f"Mean First Passage Time:\n{mfpt}")

    return normalized_matrix

def main():
    # List of CSV files for different killers
    killer_files = [
        ('Type1_Allanson_Patricia.csv', 'Allanson_Patricia'),
        # Add more files here
    ]

    aggregate_matrix = np.zeros((NUM_COG_VECTORS, NUM_COG_VECTORS))

    for file_path, killer_name in killer_files:
        killer_matrix = analyze_killer(file_path, killer_name)
        aggregate_matrix += killer_matrix

    # Normalize and save the aggregate matrix
    aggregate_matrix /= len(killer_files)
    np.save('aggregate_transition_matrix.npy', aggregate_matrix)

    print("\nAggregate Analysis:")
    print(f"Stationary Distribution: {find_stationary_distribution(aggregate_matrix)}")
    print(f"Entropy Rate: {entropy_rate(aggregate_matrix)}")
    print(f"Mean First Passage Time:\n{mean_first_passage_time(aggregate_matrix)}")

if __name__ == "__main__":
    main()
