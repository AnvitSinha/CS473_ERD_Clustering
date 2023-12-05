import argparse
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.cluster import KMeans

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

ps = PorterStemmer()


def get_args() -> argparse.Namespace:
    """Sets the CLI arguments required for the functioning of program"""

    parser = argparse.ArgumentParser()

    parser.add_argument('--words_file', help="Path to the file containing the words for the images", required=True)
    parser.add_argument('--output_file', help="Path to file where the output should be created", required=True)
    parser.add_argument('--num_clusters', type=int, help="Number of expected clusters", required=True)

    return parser.parse_args()


def get_word_lists(input_file: str) -> list[list[str]]:
    """Get the list of lists each of which represents an image.
        Removes all stopwords and stems the words"""

    all_img_words = []

    stop_words = set(stopwords.words('english'))

    with open(input_file, "r") as inp:
        for lines in inp:
            # add the lowercase version of every non-stop word after stemming
            all_img_words.append(
                [ps.stem(w.lower().strip().strip("'")) for w in lines.replace('[', '').replace(']', '').split(',')
                 if w.lower().strip().strip("'") not in stop_words]
            )

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


def main():
    print(__file__)


if __name__ == '__main__':
    main()
