import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from sentence_transformers import SentenceTransformer
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import sys
import glob
import argparse

# Ensure consistent behavior
torch.manual_seed(42)
np.random.seed(42)

# Step 1: Load the trained prototypical network model and associated data
def load_model_and_data(model_path='proto_net.pth', prototypes_path='prototypes.pkl', label_to_theme_path='label_to_theme.pkl', model_name='all-MiniLM-L6-v2'):
    """
    Loads the trained prototypical network model, prototypes, and label_to_theme mapping.
    """
    # Load the model architecture
    class ProtoNet(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(ProtoNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            return out

    # Load the label_to_theme mapping
    with open(label_to_theme_path, 'rb') as f:
        label_to_theme = pickle.load(f)

    # Load the prototypes
    with open(prototypes_path, 'rb') as f:
        prototypes = pickle.load(f)

    # Initialize the SentenceTransformer model to get input_size
    model_embedding = SentenceTransformer(model_name)
    input_size = model_embedding.get_sentence_embedding_dimension()
    hidden_size = 128  # Should match the size used during training

    # Initialize the model
    model = ProtoNet(input_size, hidden_size)
    # Load the trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model, prototypes, label_to_theme, model_embedding

# Step 2: Classify life events using the prototypical network
def classify_life_events(model, prototypes, life_events, label_to_theme, model_embedding):
    """
    Classifies each life event into an archetypal theme.
    """
    # Use the provided model_embedding to generate embeddings
    event_embeddings = model_embedding.encode(life_events)
    with torch.no_grad():
        event_embeddings_tensor = torch.Tensor(event_embeddings)
        event_outputs = model(event_embeddings_tensor)
        # Prepare prototype embeddings
        proto_labels = list(prototypes.keys())
        proto_embeddings = torch.stack([prototypes[label] for label in proto_labels])
        # Compute distances
        distances = torch.cdist(event_outputs, proto_embeddings)
        # Predict labels
        predicted_indices = torch.argmin(distances, dim=1).numpy()
        predicted_labels = [proto_labels[idx] for idx in predicted_indices]
        predicted_themes = [label_to_theme[label] for label in predicted_labels]
    return predicted_labels, predicted_themes

# Step 3: Build the sequence of themes
def build_theme_sequence(predicted_labels):
    """
    Builds the sequence of themes (labels) from the predicted labels.
    """
    return predicted_labels  # Already in sequence

# Step 4: Train a Markov Transition Matrix
def train_markov_chain(theme_sequence, num_states):
    """
    Trains a Markov Transition Matrix based on the sequence of themes.
    """
    transition_matrix = np.zeros((num_states, num_states))
    for (i, j) in zip(theme_sequence[:-1], theme_sequence[1:]):
        transition_matrix[i, j] += 1

    # Normalize the matrix
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums!=0)
        # Replace zero rows with uniform probabilities
        zero_row_indices = np.where(row_sums.flatten() == 0)[0]
        for idx in zero_row_indices:
            transition_matrix[idx] = np.ones(num_states) / num_states
    return transition_matrix

# Step 5: Visualize the state transition diagram
def plot_state_transition_diagram(transition_matrix, label_to_theme, output_file='state_transition_diagram.png'):
    """
    Plots and saves the state transition diagram.
    """
    G = nx.DiGraph()
    num_states = transition_matrix.shape[0]
    # Add nodes
    for i in range(num_states):
        G.add_node(i, label=f'State {i}')
    # Add edges with weights
    for i in range(num_states):
        for j in range(num_states):
            weight = transition_matrix[i, j]
            if weight > 0:
                G.add_edge(i, j, weight=weight)
    pos = nx.circular_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, labels={i: f"{i}: {label_to_theme[i][:15]}..." for i in G.nodes()}, node_size=2000, node_color='lightblue', font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): f"{w:.2f}" for (i, j), w in edge_labels.items()}, font_size=6)
    plt.title('State Transition Diagram')
    plt.savefig(output_file)
    plt.clf()
    print(f"State transition diagram saved to {output_file}")

# Step 6: Compute Mean First Passage Times and Stationary Distribution
def compute_markov_properties(transition_matrix):
    """
    Computes mean first passage times and stationary distribution.
    """
    if np.any(np.isnan(transition_matrix)) or np.any(np.isinf(transition_matrix)):
        print("Transition matrix contains NaN or Inf values. Skipping computation.")
        return None, None

    num_states = transition_matrix.shape[0]
    # Compute stationary distribution
    eigvals, eigvecs = np.linalg.eig(transition_matrix.T)
    stationary = np.array(eigvecs[:, np.isclose(eigvals, 1)])
    stationary = stationary[:,0]
    stationary = stationary / stationary.sum()
    stationary = stationary.real
    # Mean first passage times
    I = np.eye(num_states)
    Z = np.linalg.inv(I - transition_matrix + np.outer(np.ones(num_states), stationary))
    mean_first_passage = (Z - np.diag(np.diag(Z))) / np.outer(stationary, np.ones(num_states))
    return stationary, mean_first_passage

def plot_stationary_distribution(stationary, label_to_theme, output_file='stationary_distribution.png'):
    """
    Plots and saves the stationary distribution.
    """
    num_states = len(stationary)
    labels = [f"{i}: {label_to_theme[i][:15]}..." for i in range(num_states)]
    plt.figure(figsize=(10,6))
    sns.barplot(x=labels, y=stationary)
    plt.xticks(rotation=45, ha='right')
    plt.title('Stationary Distribution of States')
    plt.ylabel('Probability')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.clf()
    print(f"Stationary distribution saved to {output_file}")

def plot_mean_first_passage(mean_first_passage, label_to_theme, output_file='mean_first_passage.png'):
    """
    Plots and saves the mean first passage times heatmap.
    """
    num_states = mean_first_passage.shape[0]
    labels = [f"{i}: {label_to_theme[i][:15]}..." for i in range(num_states)]
    plt.figure(figsize=(10,8))
    sns.heatmap(mean_first_passage, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title('Mean First Passage Times')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.clf()
    print(f"Mean first passage times saved to {output_file}")

# Step 7: Main function
def main():
    parser = argparse.ArgumentParser(description='Life Event Sequence Analysis')
    parser.add_argument('--input_file', type=str, help='Path to the individual life events CSV file')
    parser.add_argument('--input_pattern', type=str, help='Glob pattern to process multiple CSV files (e.g., "Type1_*.csv")')
    args = parser.parse_args()

    if args.input_file:
        csv_files = [args.input_file]
    elif args.input_pattern:
        csv_files = glob.glob(args.input_pattern)
        if not csv_files:
            print(f"No files matched the pattern '{args.input_pattern}'")
            sys.exit(1)
    else:
        print("Please provide either --input_file or --input_pattern")
        sys.exit(1)

    # Paths to model and data
    model_path = 'proto_net.pth'
    prototypes_path = 'prototypes.pkl'
    label_to_theme_path = 'label_to_theme.pkl'
    model_name = 'all-MiniLM-L6-v2'

    # Load the model and data
    model, prototypes, label_to_theme, model_embedding = load_model_and_data(model_path, prototypes_path, label_to_theme_path, model_name)

    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"CSV file {csv_file} not found.")
            continue
        try: 
            df = pd.read_csv(csv_file, encoding='utf-8', quotechar='"', skipinitialspace=True)
            if 'Life Event' not in df.columns:
                print(f"CSV file {csv_file} must contain a 'Life Event' column.")
                continue
            life_events = df['Life Event'].astype(str).tolist()
            individual_name = os.path.splitext(os.path.basename(csv_file))[0]

            # Classify life events
            predicted_labels, predicted_themes = classify_life_events(model, prototypes, life_events, label_to_theme, model_embedding)

            # Add the predictions to the DataFrame
            df['PredictedLabel'] = predicted_labels
            df['PredictedTheme'] = predicted_themes

            # Save the annotated life events
            annotated_csv = f'{individual_name}_annotated.csv'
            df.to_csv(annotated_csv, index=False)
            print(f"Annotated life events saved to {annotated_csv}")

            # Build the theme sequence
            theme_sequence = build_theme_sequence(predicted_labels)

            # Calculate the number of states based on the maximum label
            num_states = max(theme_sequence) + 1  # Ensure the transition matrix can accommodate all labels

            # Train Markov Transition Matrix
            transition_matrix = train_markov_chain(theme_sequence, num_states)

            # Visualize the state transition diagram
            state_diagram_file = f'{individual_name}_state_transition_diagram.png'
            plot_state_transition_diagram(transition_matrix, label_to_theme, output_file=state_diagram_file)

            # Compute stationary distribution and mean first passage times
            stationary, mean_first_passage = compute_markov_properties(transition_matrix)

            # Plot stationary distribution
            stationary_distribution_file = f'{individual_name}_stationary_distribution.png'
            plot_stationary_distribution(stationary, label_to_theme, output_file=stationary_distribution_file)

            # Plot mean first passage times
            mean_first_passage_file = f'{individual_name}_mean_first_passage.png'
            plot_mean_first_passage(mean_first_passage, label_to_theme, output_file=mean_first_passage_file)

            # Optionally, save the transition matrix and other data
            np.save(f'{individual_name}_transition_matrix.npy', transition_matrix)
            np.save(f'{individual_name}_stationary_distribution.npy', stationary)
            np.save(f'{individual_name}_mean_first_passage.npy', mean_first_passage)
            print(f"Transition matrix, stationary distribution, and mean first passage times saved for {individual_name}.")
        except pd.errors.ParserError as e:
            print(f"Error reading {csv_file}: {e}")
            continue  # Skip to the next file

if __name__ == "__main__":
    main()