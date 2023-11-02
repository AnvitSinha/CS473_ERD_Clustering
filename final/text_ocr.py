import easyocr
import argparse


def get_args() -> argparse.Namespace:
    """Sets the CLI arguments required for the functioning of program"""

    parser = argparse.ArgumentParser()

    parser.add_argument('--img_path', help="Path to the image to extract text from", required=True)
    parser.add_argument('--save_dir', help="Path to the directory where all text files are to be saved", required=True)

    return parser.parse_args()


def extract_lines(img_path: str) -> list:
    """Take the given path to an image and extract any text from it"""

    # Create an EasyOCR reader instance with English as the default language
    reader = easyocr.Reader(lang_list=['en'])

    # Extract text from the image
    results = reader.readtext(img_path, paragraph=False)
    print(results)

    # Sort results based on top-left y coordinate (for top to bottom)
    # and then on the top-left x coordinate (for left to right)
    # results.sort(key=lambda detection: (detection[0][0][1], detection[0][0][0]))

    # Extract and store recognized text
    extracted_lines = [detection[1] for detection in results]

    return extracted_lines


def get_text(img_path: str) -> list:
    """Take in the path to an image and append all texts present in it
        to a list based on the kind of ERD object it is"""

    # get all lines present in the image
    all_lines = extract_lines(img_path)

    # array to return
    arr = []

    if img_path.find("weak_entity") >= 0:
        arr.append("weak_entity")
    elif img_path.find("rel_attr") >= 0:
        arr.append("rel_attr")
    elif img_path.find("entity") >= 0:
        arr.append("entity")
    elif img_path.find("ident_rel") >= 0:
        arr.append("ident_rel")
    elif img_path.find("rel") >= 0:
        arr.append("rel")

    arr.extend(all_lines)

    val = arr.count('PK')
    if val != 1:
        return arr

    arr2 = list()
    idx = arr.index('PK')
    if idx != 1:
        substring = list(arr[1:idx])  # 1 item
        substring2 = list(arr[idx + 1:])  # everything after PK
        arr2.append(arr[0])  # title
        arr2.append(arr[1])  # adding PK
        arr2.append(arr[idx])

        for item in substring:
            arr2.append(item)

        for item in substring2:
            arr2.append(item)

        return arr2


def create_txt(arr: list, file_path: str):
    """Put the text representation of the list into the given file"""

    with open(file_path, 'a') as fp:
        for x in arr:
            fp.write(str(x) + '\n')

    # fp = open(file_path, 'w')
    #
    # for x in rarr:
    #     fp.write(str(x))
    #     fp.write('\n')


def main():
    # get all arguments
    args = get_args()

    # get the array to write
    arr = get_text(args.img_path)

    create_txt(arr, args.save_dir)


if __name__ == '__main__':
    main()
