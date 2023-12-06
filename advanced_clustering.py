import argparse
import os
import numpy as np
import warnings

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import rand_score

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import normalize

warnings.filterwarnings('ignore')
nltk.download('stopwords', quiet=True)


# Weights for clustering
TEXT_REL_WEIGHT = 0.1
TEXT_IDENT_REL_WEIGHT = 0.1
TEXT_REL_ATTR_WEIGHT = 10.0
NUM_ENTITY_WEIGHT = 0.1
NUM_WEAK_ENTITY_WEIGHT = 10.0
NUM_REL_WEIGHT = 1.0
NUM_IDENT_REL_WEIGHT = 0.1
NUM_REL_ATTR_WEIGHT = 0.1


def get_args() -> argparse.Namespace:
    """Sets the CLI arguments required for the functioning of program"""

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', help="Path to directory containing all image information for the dataset"
                        , required=True)
    parser.add_argument('--output_file', help="Path to file where the output should be created", required=True)
    parser.add_argument('--num_clusters', type=int, help="Number of expected clusters", required=True)

    return parser.parse_args()


def get_image_names(dataset_dir: str) -> list[str]:
    """Get the names of all images in the dataset"""

    return os.listdir(dataset_dir)


def create_dataset_json(dataset_dir: str) -> list[dict]:
    """Creates the list of dictionaries that represents each image in the dataset"""

    img_erds = []

    for input_file in os.listdir(dataset_dir):

        curr_img = dict()

        with open(os.path.join(dataset_dir, input_file), "r") as inp:

            for line in inp:
                # get just the strings in the line
                curr_line = [w.strip().strip("'") for w in line.replace('[', '').replace(']', '').split(',')]

    return list(dict())


def preprocess_text(text):
    """Helper function to preprocess the text using porter stemmer and nltk stopwords"""
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = [stemmer.stem(word) for word in text.split() if word.lower() not in stop_words]
    return ' '.join(words)


def vectorize_text_data(text_data, weight=1.0):
    """Helper function to vectorize a list of text data"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_data).toarray()
    return tfidf_matrix * weight


def contains_non_empty_strings(arr):
    """Helper function to check if an array contains only empty strings"""
    return any(s.strip() for s in arr)


def get_clusters(img_erds: list[dict], num_clusters: int) -> list[int]:
    """Get the clustering of each image as a list where the index in the clusterings
        corresponds to an image"""

    all_text_rel = []
    all_text_ident_rel = []
    all_text_rel_attr = []
    all_text_entity = []
    all_text_weak_entity = []

    for erd in img_erds:
        all_text_rel.append(preprocess_text(' '.join(erd['rel'])))
        all_text_ident_rel.append(preprocess_text(' '.join(erd['ident_rel'])))
        all_text_rel_attr.append(preprocess_text(' '.join(erd['rel_attr'])))
        all_text_rel_attr.append(preprocess_text(' '.join(erd['entity'])))
        all_text_rel_attr.append(preprocess_text(' '.join(erd['weak_entity'])))

    # Vectorize and cluster with updated weights
    if contains_non_empty_strings(all_text_rel):
        vectorized_text_rel = vectorize_text_data(all_text_rel, TEXT_REL_WEIGHT)
    else:
        vectorized_text_rel = np.array([])
        
    if contains_non_empty_strings(all_text_ident_rel):
        vectorized_text_ident_rel = vectorize_text_data(all_text_ident_rel, TEXT_IDENT_REL_WEIGHT)
    else:
        vectorized_text_ident_rel = np.array([])
        
    if contains_non_empty_strings(all_text_rel_attr):
        vectorized_text_rel_attr = vectorize_text_data(all_text_rel_attr, TEXT_REL_ATTR_WEIGHT)
    else:
        vectorized_text_rel_attr = np.array([])

    # Combine the vectors
    combined_vectors = [v for v in
                        [vectorized_text_rel, vectorized_text_ident_rel, vectorized_text_rel_attr]
                        if v.size > 0]

    combined_text_features = np.hstack(combined_vectors) if combined_vectors else np.array([])

    numerical_features = np.array(
        [[erd['num_entities'], erd['num_weak_entities'], erd['num_rel'], erd['num_ident_rel'], erd['num_rel_attr']]
         for erd in img_erds]
    )

    # Normalize numerical features
    normalized_numerical_features = normalize(numerical_features, axis=0)

    weighted_numerical_features = normalized_numerical_features * np.array(
        [NUM_ENTITY_WEIGHT, NUM_WEAK_ENTITY_WEIGHT, NUM_REL_WEIGHT, NUM_IDENT_REL_WEIGHT, NUM_REL_ATTR_WEIGHT]
    )

    features = np.hstack((weighted_numerical_features, combined_text_features))

    # Use Agglomerative cluster assignments as initial centroids for K-Means
    kmeans = KMeans(n_clusters=num_clusters, init="k-means++", n_init=100)
    kmeans.fit(features)

    return kmeans.labels_


def main():
    print(__file__)


if __name__ == '__main__':
    main()
