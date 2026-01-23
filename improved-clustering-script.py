import os
import sys
import subprocess
import csv
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import TruncatedSVD  # Add this import
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import spacy
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import scipy.sparse as sp

# Download NLTK data silently
for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

nlp = spacy.load("en_core_web_sm")

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-key"


# Check and upgrade threadpoolctl

# NOTE run this (pip install threadpoolctl==3.1.0) for compatibility issues, so annoying, I'll add an implementation without sklearn. 
def upgrade_threadpoolctl():
    try:
        import threadpoolctl
        version = threadpoolctl.__version__
        if version.startswith('2.'):
            print("Upgrading threadpoolctl...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "threadpoolctl"])
            print("threadpoolctl upgraded successfully.")
        else:
            print(f"threadpoolctl version {version} is already up to date.")
    except ImportError:
        print("threadpoolctl not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "threadpoolctl"])
        print("threadpoolctl installed successfully.")

# Upgrade threadpoolctl before importing sklearn
upgrade_threadpoolctl()
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

# Step 2: Preprocess text data, we can add some more processing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return " ".join(tokens)


def generate_embeddings(sentences, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    return model.encode(sentences)

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

def evaluate_clustering(features, labels):
    if len(set(labels)) < 2:
        return 0, 0  # Return 0 for both scores if there's only one cluster

    # Check if features is a sparse matrix
    if sp.issparse(features):
        features = features.toarray()
    
    return silhouette_score(features, labels), calinski_harabasz_score(features, labels)



def analyze_clusters(original_sentences, embeddings, cluster_labels, n_representatives=3):
    # Check if embeddings is a sparse matrix
    if sp.issparse(embeddings):
        embeddings = embeddings.toarray()
    
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
            'representative_samples': [original_sentences[i] for i in representative_indices]
        })
    
    return results

def analyze_cluster_with_llm(cluster):
    llm = OpenAI(temperature=0.4)
    template = """
    You are an expert in criminal psychology and behavioral analysis. Given the following life events of serial killers, take a deep breath and carefully and analytically identify the archetypal pattern or theme they represent. Be concise and specific in your analysis.

    Life events:
    {events}

    Archetypal theme:
    """
    prompt = PromptTemplate(template=template, input_variables=["events"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    events = "\n".join(cluster['representative_samples'])
    response = llm_chain.run(events)
    
    return response.strip()

def plot_cluster_sizes(cluster_labels):
    cluster_sizes = Counter(cluster_labels)
    plt.bar(cluster_sizes.keys(), cluster_sizes.values())
    plt.xlabel('Cluster')
    plt.ylabel('Number of Events')
    plt.title('Distribution of Events Across Clusters')
    plt.savefig('cluster_distribution.png')
    plt.close()

def extract_content_features(texts):
    # Generate sentence embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)

    # # Generate TF-IDF features
    # tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    # tfidf_matrix = tfidf.fit_transform(texts)

    # # Combine embeddings and TF-IDF features
    # features = hstack([tfidf_matrix, embeddings])

    # return features
    # We're not using TF-IDF here, just the embeddings
    return embeddings

def analyze_serial_killer_events(directory, algorithm='kmeans', n_clusters=10, n_representatives=3):
    original_life_events = load_csv_files(directory)
    preprocessed_events = [preprocess_text(event) for event in original_life_events]
    
    # Extract content-based features
    features = extract_content_features(preprocessed_events)
    
    # Split data into train and test sets
    train_features, test_features, train_events, test_events, train_original, test_original = train_test_split(
        features, preprocessed_events, original_life_events, test_size=0.2, random_state=42
    )
    
    # Define the pipeline
    # pipeline = Pipeline([
    #     ('scaler', StandardScaler(with_mean=False)),  # StandardScaler that works with sparse matrices
    #     ('pca', TruncatedSVD()),  # Use TruncatedSVD instead of PCA for sparse data
    #     ('clusterer', KMeans())
    # ])
    pipeline = Pipeline([
        ('clusterer', KMeans())
    ])
    
    
    # Define the parameter grid
    # param_grid = {
    #     'pca__n_components': [50, 100, 150, 200],
    #     'clusterer__n_clusters': [5, 8, 10, 12, 15],
    #     'clusterer__init': ['k-means++', 'random']
    # }
    param_grid = {
        'clusterer__n_clusters': [5, 8, 10, 12, 15],
        'clusterer__init': ['k-means++', 'random']
    }
    
    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(train_features)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Get cluster labels
    train_labels = best_model.named_steps['clusterer'].labels_
    
    # Evaluate clustering
    silhouette, calinski_harabasz = evaluate_clustering(train_features, train_labels)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Silhouette Score: {silhouette}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz}")
    
    # Plot cluster distribution
    plot_cluster_sizes(train_labels)
    
    # Analyze clusters
    results = analyze_clusters(train_original, train_features, train_labels, n_representatives=n_representatives)
    results.sort(key=lambda x: x['cluster'])
    
    # Analyze each cluster with LLM
    for cluster in results:
        cluster['archetypal_theme'] = analyze_cluster_with_llm(cluster)
    
    # Classify test data
    test_labels = best_model.predict(test_features)
    
    # Evaluate test data clustering
    test_silhouette, test_calinski_harabasz = evaluate_clustering(test_features, test_labels)
    print(f"Test Silhouette Score: {test_silhouette}")
    print(f"Test Calinski-Harabasz Score: {test_calinski_harabasz}")
    
    return results, (silhouette, calinski_harabasz), (test_silhouette, test_calinski_harabasz)

 
if __name__ == "__main__":
    directory = "/Users/ajithsenthil/Desktop/CrimeArchetypes/mnt/data/csv"
    # modify the algorithm to the type of clustering you want to use, change n_representatives to have more examples from each cluster, specify number of clusters if needed.
    results, train_scores, test_scores = analyze_serial_killer_events(directory, algorithm='kmeans', n_clusters=15, n_representatives=5)

    print("\nClustering Results:")
    for cluster in results:
        print(f"Cluster {cluster['cluster']}:")
        print(f"  Size: {cluster['size']}")
        print("  Representative samples:")
        for i, sample in enumerate(cluster['representative_samples'], 1):
            print(f"    {i}. {sample}")
        print(f"  Archetypal Theme: {cluster['archetypal_theme']}")
        print()

    print(f"Train Scores - Silhouette: {train_scores[0]:.3f}, Calinski-Harabasz: {train_scores[1]:.3f}")
    print(f"Test Scores - Silhouette: {test_scores[0]:.3f}, Calinski-Harabasz: {test_scores[1]:.3f}")


"""


Silhouette Score:

Range: -1 to 1
Interpretation:

Values near 1 indicate that samples are well matched to their own clusters and poorly matched to neighboring clusters.
Values near 0 indicate overlapping clusters.
Negative values indicate that samples might have been assigned to the wrong clusters.




Calinski-Harabasz Score:

Range: Higher values are better
Interpretation:

Higher scores relate to a model with better defined clusters.
It's a ratio of the between-cluster dispersion mean and the within-cluster dispersion.

"""