import argparse
import os
import warnings

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import normalize

warnings.filterwarnings('ignore')
nltk.download('stopwords', quiet=True)

# Weights for clustering - Pretrained values

TEXT_REL_ATTR_WEIGHT = 1.4500000000000004
TEXT_ENTITY_WEIGHT = 0.7499999999999998

NUM_ENTITY_WEIGHT = -0.5499999999999999
NUM_WEAK_ENTITY_WEIGHT = 3.284449513138848
NUM_REL_WEIGHT = 4.75
NUM_IDENT_REL_WEIGHT = 1.4227992175194168
NUM_REL_ATTR_WEIGHT = 5.649999999999998


def get_args() -> argparse.Namespace:
    """Sets the CLI arguments required for the functioning of program"""

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', help="Path to directory containing all image information for the dataset"
                        , required=True)
    parser.add_argument('--output_file', help="Path to file where the output should be created", required=True)
    parser.add_argument('--num_clusters', type=int, help="Number of expected clusters", required=True)

    return parser.parse_args()


def create_dataset_json(dataset_dir: str) -> list[dict]:
    """Creates the list of dictionaries that represents each image in the dataset"""

    img_erds = []

    for input_file in os.listdir(dataset_dir):
        with open(os.path.join(dataset_dir, input_file), "r") as inp:
            all_lines = [w.strip().replace('[', '').replace(']', '') for w in inp.readlines()]

            # remove the quotes from each word and remove PK if found since that is not needed
            all_lines = [[w.strip().strip("'") for w in line.split(",") if w.strip().strip("'") != "PK"]
                         for line in all_lines]

        img_erds.append(process_input_file(all_lines, input_file))

    return img_erds


def process_input_file(all_lines: list[list[str]], name: str) -> dict:
    """Helper function to create a description of the image with its properties
        gathered from the input file that calls this function"""

    # intialize the dictionary
    img_desc = {
        'num_entity': 0,
        'num_weak_entity': 0,
        'num_rel': 0,
        'num_ident_rel': 0,
        'num_rel_attr': 0,
        'entity': [],
        'weak_entity': [],
        'rel': [],
        'ident_rel': [],
        'rel_attr': [],
        'name': name
    }

    for line in all_lines:

        # process each object

        if line[0] == "entity":
            img_desc['num_entity'] += 1
            img_desc['entity'].extend(line[1:])

        elif line[0] == "weak_entity":
            img_desc['num_weak_entity'] += 1
            img_desc['weak_entity'].extend(line[1:])

        elif line[0] == "rel":
            img_desc['num_rel'] += 1
            img_desc['rel'].extend(line[1:])

        elif line[0] == "rel_attr":
            img_desc['num_rel_attr'] += 1
            img_desc['rel_attr'].extend(line[1:])

        elif line[0] == "ident_rel":
            img_desc['num_ident_rel'] += 1
            img_desc['ident_rel'].extend(line[1:])

    return img_desc


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


def get_clusters(img_erds: list[dict], num_clusters: int) -> list[dict]:
    """Get the clustering of each image as a list where the index in the clusterings
        corresponds to an image"""

    all_text_rel_attr = []
    all_text_entity = []

    for erd in img_erds:
        all_text_rel_attr.append(preprocess_text(' '.join(erd['rel_attr'])))
        all_text_entity.append(preprocess_text(' '.join(erd['entity']) + ' '.join(erd['weak_entity'])))

    if contains_non_empty_strings(all_text_rel_attr):
        vectorized_text_rel_attr = vectorize_text_data(all_text_rel_attr, TEXT_REL_ATTR_WEIGHT)
    else:
        vectorized_text_rel_attr = np.array([])

    if contains_non_empty_strings(all_text_entity):
        vectorized_text_entity = vectorize_text_data(all_text_entity, TEXT_ENTITY_WEIGHT)
    else:
        vectorized_text_entity = np.array([])

    combined_vectors = [v for v in
                        [vectorized_text_rel_attr, vectorized_text_entity]
                        if v.size > 0]

    combined_text_features = np.hstack(combined_vectors) if combined_vectors else np.array([])

    numerical_features = np.array(
        [[erd['num_entity'], erd['num_weak_entity'], erd['num_rel'], erd['num_ident_rel'], erd['num_rel_attr']]
         for erd in img_erds]
    )

    # Normalize numerical features
    normalized_numerical_features = normalize(numerical_features, axis=0)

    weighted_numerical_features = normalized_numerical_features * np.array(
        [NUM_ENTITY_WEIGHT, NUM_WEAK_ENTITY_WEIGHT, NUM_REL_WEIGHT, NUM_IDENT_REL_WEIGHT, NUM_REL_ATTR_WEIGHT]
    )

    features = np.hstack((weighted_numerical_features, combined_text_features))

    kmeans = KMeans(n_clusters=num_clusters, init="k-means++", n_init=100)
    kmeans.fit(features)
    kmeans_cluster_assignments = kmeans.labels_

    for i, img_desc in enumerate(img_erds):
        img_desc['cluster_label'] = kmeans_cluster_assignments[i]

    print(kmeans_cluster_assignments)

    return img_erds


def write_output(output_file: str, img_names: list[str], clusters: list[int]):
    """Write to the output file the cluster assignments for each image name
        where each line in the output file corresponds to a cluster"""

    all_lines = [[] for _ in range(max(clusters) + 1)]  # create empty list for each cluster

    for i in range(len(img_names)):
        all_lines[clusters[i]].append(img_names[i].rstrip(".txt"))

    with open(output_file, 'w') as out:
        out.write("\n".join([", ".join(line) for line in all_lines]))


def main():
    # get args
    args = get_args()

    # get image dataset as dictionary
    img_erds = create_dataset_json(args.dataset_dir)

    # get clusterings
    cluster_erds = get_clusters(img_erds, args.num_clusters)

    # write output
    write_output(args.output_file, [i['name'] for i in cluster_erds], [i['cluster_label'] for i in cluster_erds])


if __name__ == '__main__':
    main()
