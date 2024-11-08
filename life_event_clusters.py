import os
import csv
import json
import argparse
import logging
import sys

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import spacy
import getpass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading 'en_core_web_sm' model for spaCy as it was not found.")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Securely set OpenAI API Key
if not os.environ.get("OPENAI_API_KEY"):
    logger.info("OpenAI API key not found in environment variables.")
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# Initialize OpenAI LLM with LangChain
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=500,  # Adjust based on your needs
    timeout=60,      # Timeout in seconds
    max_retries=2
)

# Define the prompt template
prompt_template = """
You are an expert in criminal psychology and behavioral analysis. Given the following life events of serial killers, identify the archetypal pattern or theme they represent. Be concise and specific in your analysis.

Life events:
{events}

Archetypal theme:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["events"]
)

# Create LLMChain
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Step 1: Load CSV files
def load_csv_files(directory):
    logger.info(f"Loading CSV files from directory: {directory}")
    life_events = []
    for filename in os.listdir(directory):
        if filename.startswith("Type1_") and filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            logger.info(f"Processing file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader, None)  # Skip header row
                for row in csv_reader:
                    if len(row) >= 3:  # Ensure row has at least 3 columns
                        life_event = row[2].strip()  # Get the Life Event column
                        if life_event:  # Check if it's not empty
                            life_events.append(life_event)
    logger.info(f"Total life events loaded: {len(life_events)}")
    return life_events

# Step 2: Preprocess text data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Use spaCy for Named Entity Recognition
    doc = nlp(text)
    
    # Remove person names and replace other named entities with their types
    tokens = []
    for token in doc:
        if token.ent_type_ == 'PERSON':
            continue  # Skip person names
        elif token.ent_type_ in ['GPE', 'ORG']:
            tokens.append(token.ent_type_)  # Replace with entity type
        else:
            tokens.append(token.text)
    
    text = ' '.join(tokens)
    
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
    logger.info(f"Generating embeddings using model: {model_name}")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, show_progress_bar=True)
    logger.info(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings

# Step 4: Apply clustering algorithm
def cluster_embeddings(embeddings, algorithm='kmeans', n_clusters=5):
    logger.info(f"Clustering embeddings using {algorithm} with {n_clusters} clusters")
    if algorithm == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif algorithm == 'agglomerative':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    elif algorithm == 'dbscan':
        clusterer = DBSCAN(eps=0.5, min_samples=5)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
    
    cluster_labels = clusterer.fit_predict(embeddings)
    unique_labels = set(cluster_labels)
    logger.info(f"Number of clusters found: {len(unique_labels)}")
    return cluster_labels

# Step 5: Analyze clusters and find representative samples
def analyze_clusters(sentences, embeddings, cluster_labels, n_representatives=3):
    logger.info("Analyzing clusters to find representative samples")
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
        representative_samples = [sentences[i] for i in representative_indices]
        results.append({
            'cluster': label,
            'size': len(indices),
            'representative_samples': representative_samples
        })
        logger.info(f"Cluster {label}: size={len(indices)}")
    
    return results

# Step 6: LLM analysis of cluster representatives
def analyze_cluster_with_llm(cluster):
    events = "\n".join(cluster['representative_samples'])
    logger.info(f"Analyzing cluster {cluster['cluster']} with LLM")
    try:
        response = llm_chain.run({"events": events})
        archetypal_theme = response.strip()
    except Exception as e:
        logger.error(f"Error during LLM analysis for cluster {cluster['cluster']}: {e}")
        archetypal_theme = "Analysis failed."
    return archetypal_theme

# Main function to run the analysis
def analyze_serial_killer_events(directory, algorithm='kmeans', n_clusters=10, n_representatives=3, output_file=None):
    # Load and preprocess data
    life_events = load_csv_files(directory)
    preprocessed_events = [preprocess_text(event) for event in life_events]
    
    # Generate embeddings and cluster
    embeddings = generate_embeddings(preprocessed_events)
    cluster_labels = cluster_embeddings(embeddings, algorithm=algorithm, n_clusters=n_clusters)
    
    # Analyze clusters
    results = analyze_clusters(life_events, embeddings, cluster_labels, n_representatives=n_representatives)
    
    # Sort results by cluster label
    results.sort(key=lambda x: x['cluster'])
    
    # Analyze each cluster with LLM
    for cluster in results:
        cluster['archetypal_theme'] = analyze_cluster_with_llm(cluster)
    
    # Output results
    if output_file:
        logger.info(f"Saving results to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    else:
        for cluster in results:
            print(f"Cluster {cluster['cluster']}:")
            print(f"  Size: {cluster['size']}")
            print("  Representative samples:")
            for i, sample in enumerate(cluster['representative_samples'], 1):
                print(f"    {i}. {sample}")
            print(f"  Archetypal Theme: {cluster['archetypal_theme']}")
            print()
    
    return results

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze serial killer life events to identify archetypal themes.")
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Path to the directory containing CSV data files."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="kmeans",
        choices=["kmeans", "agglomerative", "dbscan"],
        help="Clustering algorithm to use."
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=10,
        help="Number of clusters to form."
    )
    parser.add_argument(
        "--n_representatives",
        type=int,
        default=3,
        help="Number of representative samples per cluster."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON file. If not provided, results will be printed to console."
    )
    
    args = parser.parse_args()
    
    results = analyze_serial_killer_events(
        directory=args.directory,
        algorithm=args.algorithm,
        n_clusters=args.n_clusters,
        n_representatives=args.n_representatives,
        output_file=args.output
    )