import argparse
import numpy as np
import pandas as pd
import os

from sklearn.cluster import KMeans

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

ps = PorterStemmer()


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


def get_word_lists(dataset_dir: str) -> list[list[str]]:
    """Get the list of lists each of which represents an image.
        Removes all stopwords and stems the words"""

    all_img_words = []

    stop_words = set(stopwords.words('english'))

    for input_file in os.listdir(dataset_dir):

        curr_img_words = []

        with open(input_file, "r") as inp:
            for lines in inp:
                # add the lowercase version of every non-stop word after stemming
                curr_img_words.extend(
                    [ps.stem(w.lower().strip().strip("'")) for w in lines.replace('[', '').replace(']', '').split(',')
                     if w.lower().strip().strip("'") not in stop_words]
                )

        all_img_words.append(curr_img_words)

    return all_img_words


def get_tf_matrix(words_list: list[list[str]]) -> pd.DataFrame:
    """Get a Pandas dataframe with the term frequency matrix of the
        entire bag of words present in words list
        where tf(term) = 0 if count(term) = 0 and tf(term) = log(count(term)+1) otherwise"""

    # Get the unique set of words in all sublists
    all_words = set(word for sublist in words_list for word in sublist)

    # Create a list of dictionaries with word counts in sublists
    # get the term frequency as log(tf + 1) if tf > 0 else 0
    bow_tf = [
        {word: np.log(sublist.count(word) + 1) if sublist.count(word) > 0 else 0 for word in all_words}
        for sublist in words_list
    ]

    return pd.DataFrame(bow_tf)


def get_clusters(tf_matrix: pd.DataFrame, num_clusters: int) -> list[int]:
    """Get the clustering of each image as a list where the index in the clusterings
        corresponds to the row index in the tf_matrix.
        NOTE: The cluster assignments get added to the dataframe so pass a copy
        if the input tf_matrix should not be modified"""

    kmeans = KMeans(n_clusters=num_clusters, init='k-means++')
    kmeans.fit(tf_matrix)
    tf_matrix['cluster'] = kmeans.labels_  # also add to the dataframe

    return kmeans.labels_


def write_output(output_file: str, img_names: list[str], clusters: list[int]):
    """Write to the output file the cluster assignments for each image name
        where each line in the output file corresponds to a cluster"""

    all_lines = [[] for _ in range(max(clusters) + 1)]  # create empty list for each cluster

    for i in range(len(img_names)):
        all_lines[clusters[i]].append(img_names[i])

    with open(output_file, 'w') as out:
        out.write("\n".join([" ".join(line) for line in all_lines]))


def main():

    # get all arguments
    args = get_args()

    # get the names of all images
    img_names = get_image_names(args.dataset_dir)

    # get all the words in the dataset, separated by image
    all_words = get_word_lists(args.dataset_dir)

    # get the tf matrix
    tf_matrix = get_tf_matrix(all_words)

    # compute clusters
    clusters = get_clusters(tf_matrix, args.num_clusters)

    # write to file
    write_output(args.output_file, img_names, clusters)


if __name__ == '__main__':
    main()
