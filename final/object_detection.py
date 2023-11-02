import cv2
import yaml
import os
import subprocess
import argparse


def get_args() -> argparse.Namespace:
    """Sets the CLI arguments required for the functioning of program"""
    parser = argparse.ArgumentParser()

    # add required arguments
    parser.add_argument('--weights', help="Path to model weights", required=True)
    parser.add_argument('--img_src', help="Path to image source", required=True)
    parser.add_argument('--yolo_path', help="Path to Yolo directory", required=True)
    parser.add_argument('--save_dir', help="Path to directory where cropped images will be saved", required=True)
    parser.add_argument('--name', help="Name of the image", required=True)
    parser.add_argument('--yaml_src', help="Path to YAML file containing the entity names")
    parser.add_argument('--class_names', help="Path to the YAML file containing class names", required=True)

    return parser.parse_args()


def detect(yolo_path: str, img_src: str, weights: str, save_dir: str, name: str):
    """Given a source image, detect objects present in it
        using YoloV5 and the given weights"""

    # run command to detect
    subprocess.run(["python",
                    f"{yolo_path}/detect.py",
                    f"--source={img_src}",
                    f"--weights={weights}",
                    "--conf-thres=0.25",
                    "--save-txt",
                    f"--project={save_dir}",
                    f"--name={name}"])


def crop_and_save(name: str, img_src: str, save_dir: str, yaml_src: str):
    """Crop the detected image and save all entities present in their own directory"""

    # Directories
    root_dir = os.path.join(save_dir, name)
    labels_dir = os.path.join(root_dir, "labels")
    images_dir = img_src
    cropped_dir = os.path.join(root_dir, "cropped")

    # Load class names from the data.yaml file
    with open(yaml_src, 'r') as f:
        data = yaml.safe_load(f)
        class_names = data['names']

    # Create a directory for cropped images if it doesn't exist
    if not os.path.exists(cropped_dir):
        os.makedirs(cropped_dir)

    # Iterate through each label file
    for filename in os.listdir(labels_dir):

        # ignore any file that's not a label map
        if not filename.endswith('.txt'):
            continue

        # get the corresponding image for each label map
        image_name = filename.replace('.txt', '.png')
        image_path = os.path.join(images_dir, image_name)
        image = cv2.imread(image_path)

        if image is None:  # Check if image is loaded correctly
            print(f"Failed to load {image_name}. Skipping...")
            continue

        label_path = os.path.join(labels_dir, filename)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                parts = line.strip().split()
                class_id, x_center, y_center, width, height = map(float, parts)

                # Create directory for the class if it doesn't exist
                class_dir = os.path.join(cropped_dir, class_names[int(class_id)])
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)

                # Convert YOLO box format to pixel values
                x_center, y_center = int(x_center * image.shape[1]), int(y_center * image.shape[0])
                width, height = int(width * image.shape[1]), int(height * image.shape[0])
                x1, y1, x2, y2 = int(x_center - width / 2), int(y_center - height / 2), int(
                    x_center + width / 2), int(y_center + height / 2)

                # Clamp the bounding box coordinates
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                # Check if the cropped region is valid
                if x1 >= x2 or y1 >= y2:
                    print(f"Invalid bounding box for image {image_name}, crop index {index}. Skipping...")
                    continue

                cropped_image = image[y1:y2, x1:x2]
                if cropped_image.size == 0:
                    print(f"Empty cropped image for {image_name}, crop index {index}. Skipping...")
                    continue

                save_path = os.path.join(class_dir, f"{os.path.splitext(image_name)[0]}_crop{index}.png")
                cv2.imwrite(save_path, cropped_image)


def main():
    # get arguments
    args = get_args()

    # detect objects in the given image
    detect(args.yolo_path, args.img_src, args.weights, args.save_dir, args.name)

    # crop image and save each present entity in its own directory
    crop_and_save(args.name, args.img_src, args.save_dir, args.yaml_src)


if __name__ == '__main__':
    main()
