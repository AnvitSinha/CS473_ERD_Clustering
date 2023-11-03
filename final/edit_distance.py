import argparse

import editdistance as ed
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)


def get_args() -> argparse.Namespace:
    """Sets the CLI arguments required for the functioning of program"""

    parser = argparse.ArgumentParser()

    parser.add_argument('--objects_file', help="Path to the file containing the objects for the image", required=True)
    parser.add_argument('--question_path', help="Path to the file containing the question", required=True)
    parser.add_argument('--output_file', help="Path to file where the output should be created", required=True)
    parser.add_argument('--threshold', help="Threshold for edi distance", required=True)

    return parser.parse_args()


def read_objects_file(objects_file: str) -> list[list[str]]:
    """Given the path to the file with the detected objects,
        read the information into an array and return it"""

    all_lines = []
    with open(objects_file, 'r') as f:

        for lines in f:
            all_lines.append(lines.replace('[', '').replace(']', '').split(','))

        for i, line in enumerate(all_lines):
            for j, word in enumerate(line):
                all_lines[i][j] = word.replace('\'', '').strip()

    return all_lines


def process_question(question_path: str) -> set[str]:
    """Given the path to the question text file,
        process it and return the vocabulary based on the question"""

    # select words from question text to serve as a library of vocabulary for edit distance
    with open(question_path, 'r') as vocab_file:
        vocabulary = vocab_file.read().lower()

        for punc in string.punctuation:
            vocabulary = vocabulary.replace(punc, '')

        vocabulary = set(vocabulary.split())

    # remove any stop words in the question vocabulary
    stop_words = set(stopwords.words('english'))
    vocabulary = set([w for w in vocabulary if w.lower() not in stop_words])

    return vocabulary


def update(all_lines: list[list[str]], vocabulary: set[str], threshold: int):
    """Given the lines and the vocabulary, determine which words to update
        based on the given threshold"""

    for i, lines in enumerate(all_lines):
        for j, word in enumerate(lines):
            for vocab in vocabulary:
                if ed.eval(word.lower(), vocab) <= threshold:
                    all_lines[i][j] = vocab
                    break


def create_txt(all_lines: list[list[str]], file_path: str):
    """Put the text representation of the list into the given file
        based on the required format"""

    with open(file_path, 'w') as output_file:
        for lines in all_lines:
            output_file.write('[')
            for i, word in enumerate(lines):
                if i == 0:
                    output_file.write('\'' + word + '\'')
                else:
                    output_file.write(', \'' + word + '\'')
            output_file.write(']\n')


def main():
    # get args
    args = get_args()

    # get all lines from the objects file
    all_lines = read_objects_file(args.objects_file)

    # get the vocab from the question
    vocabulary = process_question(args.question_path)

    # update all_lines based on edit distance threshold
    update(all_lines, vocabulary, args.threshold)

    # write result to file
    create_txt(all_lines, args.output_file)


if __name__ == '__main__':
    main()
