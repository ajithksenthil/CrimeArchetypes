
import os
import csv
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import openai

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set up OpenAI API key
openai.api_key = "my_openai_key"  # Replace with your actual OpenAI API key

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers and dates
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return " ".join(tokens)

# Step 3: Generate sentence embeddings
def generate_embeddings(sentences, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    return model.encode(sentences)

# Step 4: Apply clustering algorithm
def cluster_embeddings(embeddings, algorithm='kmeans', n_clusters=5):
    if algorithm == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif algorithm == 'agglomerative':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    elif algorithm == 'dbscan':
        clusterer = DBSCAN(eps=0.5, min_samples=5)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
    
    return clusterer.fit_predict(embeddings)

# Step 5: Analyze clusters and find representative samples
def analyze_clusters(sentences, embeddings, cluster_labels, n_representatives=3):
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
        sorted_indices = np.argsort(distances)[::-1]  # Sort in descending order
        representative_indices = [indices[i] for i in sorted_indices[:n_representatives]]
        results.append({
            'cluster': label,
            'size': len(indices),
            'representative_samples': [sentences[i] for i in representative_indices]
        })
    
    return results

# Step 6: LLM analysis of cluster representatives
def analyze_cluster_with_llm(cluster):
    prompt_template = """
You are an expert in criminal psychology and behavioral analysis. Given the following life events of serial killers, identify the archetypal pattern or theme they represent. Be concise and specific in your analysis.

Life events:
{events}

Archetypal theme:
"""

    events = "\n".join(cluster['representative_samples'])
    prompt = prompt_template.format(events=events)

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in criminal psychology and behavioral analysis."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        temperature=0.7,
    )

    reply = response.choices[0].message['content'].strip()
    return reply

# Main function to run the analysis
def analyze_serial_killer_events(directory, algorithm='kmeans', n_clusters=10, n_representatives=5):
    # Load and preprocess data
    life_events = load_csv_files(directory)
    preprocessed_events = [preprocess_text(event) for event in life_events]
    
    # Generate embeddings and cluster
    embeddings = generate_embeddings(preprocessed_events)
    cluster_labels = cluster_embeddings(embeddings, algorithm=algorithm, n_clusters=n_clusters)
    
    # Analyze results
    results = analyze_clusters(life_events, embeddings, cluster_labels, n_representatives=n_representatives)
    
    # Sort results by cluster label
    results.sort(key=lambda x: x['cluster'])
    
    # Analyze each cluster with LLM
    for cluster in results:
        cluster['archetypal_theme'] = analyze_cluster_with_llm(cluster)
    
    return results

# Example usage
if __name__ == "__main__":
    directory = "/Users/ajithsenthil/Desktop/CrimeArchetypes/mnt/data/csv"  # Update to your directory
    results = analyze_serial_killer_events(directory, algorithm='kmeans', n_clusters=10, n_representatives=5)

    for cluster in results:
        print(f"Cluster {cluster['cluster']}:")
        print(f"  Size: {cluster['size']}")
        print("  Representative samples:")
        for i, sample in enumerate(cluster['representative_samples'], 1):
            print(f"    {i}. {sample}")
        print(f"  Archetypal Theme: {cluster['archetypal_theme']}")
        print()