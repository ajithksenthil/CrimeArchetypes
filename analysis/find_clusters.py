# find_clusters.py

# Lets find some clusters

# make sure you have these on your machine, run pip install "name of the package" one each of these if you dont have them already, you'll know if you run it and it fails saying this isnt imported
import os
import csv
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load CSV files
def load_csv_files(directory):
    life_events = []
    for filename in os.listdir(directory):
        if filename.startswith("Type1_") and filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip header row
                for row in csv_reader:
                    if len(row) >= 3:  # Ensure row has at least 3 columns
                        life_event = row[2].strip()  # Get the Life Event column
                        if life_event:  # Check if it's not empty
                            life_events.append(life_event)
    return life_events

# Step 2: Preprocess text data
def preprocess_text(text):
    # Add any necessary preprocessing steps here
    return text.lower()

# Step 3: Generate sentence embeddings
def generate_embeddings(sentences, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    return model.encode(sentences)

# Step 4: Apply clustering algorithm
def cluster_embeddings(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(embeddings)

# Step 5: Analyze clusters and find representative samples
def analyze_clusters(sentences, embeddings, cluster_labels):
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    results = []
    for label, indices in clusters.items():
        cluster_embeddings = embeddings[indices]
        centroid = np.mean(cluster_embeddings, axis=0)
        distances = cosine_similarity([centroid], cluster_embeddings)[0]
        representative_index = indices[np.argmax(distances)]
        results.append({
            'cluster': label,
            'size': len(indices),
            'representative_sample': sentences[representative_index]
        })
    
    return results

# Main function to run the analysis
def analyze_serial_killer_events(directory, n_clusters=10):
    # Load and preprocess data
    life_events = load_csv_files(directory)
    preprocessed_events = [preprocess_text(event) for event in life_events]
    
    # Generate embeddings and cluster
    embeddings = generate_embeddings(preprocessed_events)
    cluster_labels = cluster_embeddings(embeddings, n_clusters=n_clusters)
    
    # Analyze results
    results = analyze_clusters(life_events, embeddings, cluster_labels)
    
    return results

# Example usage
directory = "/Users/ajithsenthil/Desktop/CrimeArchetypes/mnt/data/csv"
results = analyze_serial_killer_events(directory, n_clusters=10)

for cluster in results:
    print(f"Cluster {cluster['cluster']}:")
    print(f"  Size: {cluster['size']}")
    print(f"  Representative sample: {cluster['representative_sample']}")
    print()