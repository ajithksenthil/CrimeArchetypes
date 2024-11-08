import os
import numpy as np
import glob
import pickle

def load_label_to_theme(label_to_theme_path='label_to_theme.pkl'):
    with open(label_to_theme_path, 'rb') as f:
        label_to_theme = pickle.load(f)
    return label_to_theme

def load_individual_metrics(directory):
    """
    Loads individual metrics from .npy files in the specified directory.
    """
    transition_matrices = []
    stationary_distributions = []
    mean_first_passages = []
    individual_names = []

    # Pattern to match individual metric files
    transition_files = glob.glob(os.path.join(directory, '*_transition_matrix.npy'))

    for file in transition_files:
        individual_name = os.path.basename(file).replace('_transition_matrix.npy', '')
        individual_names.append(individual_name)

        # Load metrics
        transition_matrix = np.load(file)
        stationary_distribution = np.load(f'{individual_name}_stationary_distribution.npy')
        mean_first_passage = np.load(f'{individual_name}_mean_first_passage.npy')

        transition_matrices.append(transition_matrix)
        stationary_distributions.append(stationary_distribution)
        mean_first_passages.append(mean_first_passage)

    return (transition_matrices, stationary_distributions, mean_first_passages, individual_names)


def aggregate_transition_matrices(transition_matrices):
    """
    Computes the average transition matrix across all individuals.
    """
    # Ensure all matrices are the same size
    max_size = max(matrix.shape[0] for matrix in transition_matrices)
    adjusted_matrices = []

    for matrix in transition_matrices:
        if matrix.shape[0] < max_size:
            # Pad the matrix with zeros to match the maximum size
            padded_matrix = np.zeros((max_size, max_size))
            padded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
            adjusted_matrices.append(padded_matrix)
        else:
            adjusted_matrices.append(matrix)

    # Stack matrices and compute the average
    stacked_matrices = np.stack(adjusted_matrices)
    average_transition_matrix = np.mean(stacked_matrices, axis=0)

    return average_transition_matrix

def aggregate_stationary_distributions(stationary_distributions):
    """
    Computes the average stationary distribution across all individuals.
    """
    max_size = max(dist.shape[0] for dist in stationary_distributions)
    adjusted_distributions = []

    for dist in stationary_distributions:
        if dist.shape[0] < max_size:
            # Pad the distribution with zeros to match the maximum size
            padded_dist = np.zeros(max_size)
            padded_dist[:dist.shape[0]] = dist
            adjusted_distributions.append(padded_dist)
        else:
            adjusted_distributions.append(dist)

    # Stack distributions and compute the average
    stacked_distributions = np.stack(adjusted_distributions)
    average_stationary_distribution = np.mean(stacked_distributions, axis=0)

    return average_stationary_distribution

def aggregate_mean_first_passages(mean_first_passages):
    """
    Computes the average mean first passage times across all individuals.
    """
    max_size = max(matrix.shape[0] for matrix in mean_first_passages)
    adjusted_matrices = []

    for matrix in mean_first_passages:
        if matrix.shape[0] < max_size:
            # Pad the matrix with zeros to match the maximum size
            padded_matrix = np.zeros((max_size, max_size))
            padded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
            adjusted_matrices.append(padded_matrix)
        else:
            adjusted_matrices.append(matrix)

    # Stack matrices and compute the average
    stacked_matrices = np.stack(adjusted_matrices)
    average_mean_first_passage = np.mean(stacked_matrices, axis=0)

    return average_mean_first_passage


def compute_statistics(stacked_data):
    """
    Computes the mean and standard deviation across the stacked data.
    """
    mean = np.mean(stacked_data, axis=0)
    std_dev = np.std(stacked_data, axis=0)
    return mean, std_dev

import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(matrix, title, labels, output_file):
    plt.figure(figsize=(10,8))
    sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap='viridis')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.clf()
    print(f"{title} saved to {output_file}")

def plot_aggregated_stationary_distribution(stationary_distribution, labels, output_file):
    plt.figure(figsize=(10,6))
    sns.barplot(x=labels, y=stationary_distribution)
    plt.xticks(rotation=45, ha='right')
    plt.title('Aggregated Stationary Distribution')
    plt.ylabel('Probability')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.clf()
    print(f"Aggregated stationary distribution saved to {output_file}")

def main():
    # Directory where individual metrics are stored
    metrics_directory = ''  # Adjust as needed

    # Load individual metrics
    (transition_matrices, stationary_distributions, mean_first_passages, individual_names) = load_individual_metrics(metrics_directory)
    print("loaded: ", individual_names)
    # Aggregate Transition Matrices
    average_transition_matrix = aggregate_transition_matrices(transition_matrices)

    # Aggregate Stationary Distributions
    average_stationary_distribution = aggregate_stationary_distributions(stationary_distributions)

    # Aggregate Mean First Passage Times
    average_mean_first_passage = aggregate_mean_first_passages(mean_first_passages)

    # Optionally, compute standard deviations
    # For example, compute std dev for transition matrices
    max_size = average_transition_matrix.shape[0]
    adjusted_matrices = []

    for matrix in transition_matrices:
        if matrix.shape[0] < max_size:
            padded_matrix = np.zeros((max_size, max_size))
            padded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
            adjusted_matrices.append(padded_matrix)
        else:
            adjusted_matrices.append(matrix)

    stacked_matrices = np.stack(adjusted_matrices)
    _, std_dev_transition_matrix = compute_statistics(stacked_matrices)

    # Labels for states (you may need to adjust this based on your data)
    label_to_theme = load_label_to_theme()
    labels = [f"{i}: {label_to_theme.get(i, 'Unknown')[:15]}..." for i in range(max_size)]

    # Visualize the aggregated transition matrix
    plot_heatmap(average_transition_matrix, 'Aggregated Transition Matrix', labels, 'aggregated_transition_matrix.png')

    # Visualize the aggregated stationary distribution
    plot_aggregated_stationary_distribution(average_stationary_distribution, labels, 'aggregated_stationary_distribution.png')

    # Visualize the aggregated mean first passage times
    plot_heatmap(average_mean_first_passage, 'Aggregated Mean First Passage Times', labels, 'aggregated_mean_first_passage.png')

    # Additional analyses can be added here

if __name__ == "__main__":
    main()