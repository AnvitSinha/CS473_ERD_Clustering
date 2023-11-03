import argparse

import easyocr


def get_args() -> argparse.Namespace:
    """Sets the CLI arguments required for the functioning of program"""

    parser = argparse.ArgumentParser()

    parser.add_argument('--img_path', help="Path to the image to extract text from", required=True)
    parser.add_argument('--save_dir', help="Path to the directory where all text files are to be saved", required=True)
    parser.add_argument('--object_type', help="Type of the ERD object currently being processed", required=True)

    return parser.parse_args()


def extract_lines(img_path: str) -> list:
    """Take the given path to an image and extract any text from it"""

    # Create an EasyOCR reader instance with English as the default language
    reader = easyocr.Reader(lang_list=['en'])

    # Extract text from the image
    results = reader.readtext(img_path, paragraph=False)

    # Extract and store recognized text
    extracted_lines = [detection[1] for detection in results]

    return extracted_lines


def get_text(img_path: str, object_type: str) -> list:
    """Take in the path to an image and append all texts present in it
        to a list based on the kind of ERD object it is"""

    # get all lines present in the image
    all_lines = extract_lines(img_path)

    # array to return, initialise to the type of entity
    arr = [object_type]

    # add all terms in to the array
    arr.extend(all_lines)

    if object_type in {"ident_rel", "rel", "rel_attr"} and len(arr) > 2:
        # if the words of these objects somehow got split
        return [arr[0], " ".join([x.strip() for x in arr[1:]])]

    return arr


def create_txt(arr: list, file_path: str):
    """Put the text representation of the list into the given file"""

    with open(file_path, 'a') as fp:
        fp.write(repr(arr) + '\n')


def main():
    # get all arguments
    args = get_args()

    # get the array to write
    arr = get_text(args.img_path, args.object_type)

    create_txt(arr, args.save_dir)


if __name__ == '__main__':
    main()
