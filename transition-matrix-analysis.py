import numpy as np

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

# Use these functions
stationary_dist = find_stationary_distribution(normalized_matrix)
entropy = entropy_rate(normalized_matrix)
mfpt = mean_first_passage_time(normalized_matrix)

print(f"Stationary Distribution: {stationary_dist}")
print(f"Entropy Rate: {entropy}")
print(f"Mean First Passage Time:\n{mfpt}")
