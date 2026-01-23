import os
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import pickle  # Import pickle for loading and saving data

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Step 1: Load the clusters and archetypal themes from the first script
def load_clusters_and_themes(filename='clusters.pkl'):
    """
    Loads clusters and their archetypal themes from the pickle file.
    """
    with open(filename, 'rb') as f:
        clusters = pickle.load(f)
    
    cluster_embeddings = []
    label_to_theme = {}
    label_to_idx = {}
    idx = 0
    for cluster in clusters:
        label = cluster['cluster_id']
        theme = cluster['archetypal_theme']
        embeddings = cluster['representative_embeddings']
        if label not in label_to_idx:
            label_to_idx[label] = idx
            idx += 1
        cluster_embeddings.append({
            'embeddings': embeddings,
            'label': label_to_idx[label]
        })
        label_to_theme[label_to_idx[label]] = theme
    return cluster_embeddings, label_to_theme

# Step 2: Compute prototypes
def compute_prototypes(cluster_embeddings):
    """
    Computes the prototype (mean embedding) for each cluster.
    """
    prototypes = {}
    for cluster in cluster_embeddings:
        embeddings = cluster['embeddings']
        label = cluster['label']
        prototypes[label] = np.mean(embeddings, axis=0)
    return prototypes

# Step 3: Define the Prototypical Network model
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

# Step 4: Training function
def train_proto_net(model, optimizer, support_embeddings, support_labels, epochs=10):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        support_embeddings_tensor = torch.Tensor(support_embeddings)
        outputs = model(support_embeddings_tensor)
        # Compute prototypes from outputs
        prototypes = {}
        for label in np.unique(support_labels):
            class_embeddings = outputs[np.array(support_labels) == label]
            prototypes[label] = class_embeddings.mean(dim=0)
        # Compute logits
        logits = []
        for output in outputs:
            distances = [F.pairwise_distance(output.unsqueeze(0), prototypes[label].unsqueeze(0)) for label in prototypes]
            logits.append(-torch.stack(distances).squeeze())
        logits = torch.stack(logits)
        loss = criterion(logits, torch.LongTensor(support_labels))
        loss.backward()
        optimizer.step()
        if (epoch+1)%5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Step 5: Classification function
def classify_new_events(model, prototypes, new_events, label_to_theme, model_name='all-MiniLM-L6-v2'):
    """
    Classify new life events into archetypal themes.
    """
    # Use original text for embeddings
    model_embedding = SentenceTransformer(model_name)
    event_embeddings = model_embedding.encode(new_events)
    model.eval()
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
        predicted_themes = [label_to_theme[proto_labels[idx]] for idx in predicted_indices]
    return predicted_themes

# Main function
def main():
    # Load clusters and themes
    cluster_embeddings_info, label_to_theme = load_clusters_and_themes('clusters.pkl')

    # Prepare training data
    support_embeddings = []
    support_labels = []
    for cluster_info in cluster_embeddings_info:
        embeddings = cluster_info['embeddings']
        label = cluster_info['label']
        for emb in embeddings:
            support_embeddings.append(emb)
            support_labels.append(label)
    n_classes = len(np.unique(support_labels))

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(support_embeddings, support_labels, test_size=0.2, random_state=42)

    # Initialize model
    input_size = len(support_embeddings[0])
    hidden_size = 128  # Adjust as needed
    model = ProtoNet(input_size, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_proto_net(model, optimizer, X_train, y_train, epochs=20)

    # Validate the model
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.Tensor(X_val)
        outputs = model(X_val_tensor)
        # Compute prototypes based on training data
        prototypes_eval = {}
        X_train_tensor = torch.Tensor(X_train)
        outputs_train = model(X_train_tensor)
        for label in np.unique(y_train):
            class_embeddings = outputs_train[np.array(y_train) == label]
            prototypes_eval[label] = class_embeddings.mean(dim=0)
        proto_labels = list(prototypes_eval.keys())
        proto_embeddings = torch.stack([prototypes_eval[label] for label in proto_labels])
        distances = torch.cdist(outputs, proto_embeddings)
        predicted_indices = torch.argmin(distances, dim=1).numpy()
        predicted_labels = [proto_labels[idx] for idx in predicted_indices]
        accuracy = np.mean(predicted_labels == y_val)
        print(f'Validation Accuracy: {accuracy*100:.2f}%')

    # Save the trained model
    torch.save(model.state_dict(), 'proto_net.pth')
    print("Trained prototypical network saved as 'proto_net.pth'")

    # Save the prototypes
    with open('prototypes.pkl', 'wb') as f:
        pickle.dump(prototypes_eval, f)
    print("Prototypes saved as 'prototypes.pkl'")

    # Save the label_to_theme mapping
    with open('label_to_theme.pkl', 'wb') as f:
        pickle.dump(label_to_theme, f)
    print("Label to theme mapping saved as 'label_to_theme.pkl'")

    # Now, classify new life events (optional)
    new_events = [
        "He was neglected by his parents during childhood.",
        "She experienced severe bullying in school.",
        "He found solace in setting fires in abandoned buildings.",
        "She cared for her sick mother for many years."
    ]
    predicted_themes = classify_new_events(model, prototypes_eval, new_events, label_to_theme)
    for event, theme in zip(new_events, predicted_themes):
        print(f"Event: {event}\nPredicted Archetypal Theme: {theme}\n")

if __name__ == "__main__":
    main()