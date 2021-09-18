"""

Code for extracting text data from image sources stored in the local directory. 
The code uses optical character recognition (OCR) technique from:

https://azure.microsoft.com/en-gb/services/cognitive-services/computer-vision/

To run this code, Azure subscription is needed, and its Computer Vision resource 
needs to be created. 

Then, COMPUTER_VISION_SUBSCRIPTION_KEY  and COMPUTER_VISION_ENDPOINT need to be
generated.

The sample tutorial can be found at:

https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/quickstarts/python-print-text

Finally, the text data retrieved from images by The NeurIPS 2020 Education Challenge can be found in the 
azua storage blob, under datasets/eedi_comp_tasks_3_4/images_OCR_results.

"""

import argparse
import os
import sys
import requests
import json


def get_args():
    """
    Parses command line arguments.
    Returns: namespace of command line args.
    """
    parser = argparse.ArgumentParser(
        description="Extracts text data from image sources specified in the image directory and saves the results in JSON format."
    )

    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        default="experiments/data/EEDI_question_images/Images",
        help="Directory containing image data to use.",
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="experiments/data/EEDI_question_images/ORC_results",
        help="Directory to save the OCR results from the associated images. The OCR results have the name filenames as the original images sources, with only the file extension being changed ",
    )

    args = parser.parse_args()

    # Create required directories
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = get_args()

    # Add your Computer Vision subscription key and endpoint to your environment variables.
    if "COMPUTER_VISION_SUBSCRIPTION_KEY" in os.environ:
        subscription_key = os.environ["COMPUTER_VISION_SUBSCRIPTION_KEY"]
    else:
        print(
            "\nSet the COMPUTER_VISION_SUBSCRIPTION_KEY environment variable.\n**Restart your shell or IDE for changes to take effect.**"
        )
        sys.exit()

    if "COMPUTER_VISION_ENDPOINT" in os.environ:
        endpoint = os.environ["COMPUTER_VISION_ENDPOINT"]

    ocr_url = endpoint + "vision/v3.0/ocr"

    # Set Content-Type to octet-stream
    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Content-Type": "application/octet-stream",
    }

    # Some of the imaegs detect languages wrong. Specifying a specific language (e.g., English) does not solve this issue.
    params = {"language": "unk", "detectOrientation": "true"}

    image_dir_path = args.data_dir

    fns = os.listdir(image_dir_path)

    for i, fn in enumerate(fns):
        image_dir_fn = os.path.join(image_dir_path, fn)
        # Read the image into a byte array
        image_data = open(image_dir_fn, "rb").read()
        response = requests.post(ocr_url, headers=headers, params=params, data=image_data)
        response.raise_for_status()

        analysis = response.json()

        save_dir_fn = os.path.join(args.output_dir, os.path.splitext(fn)[0] + ".json")
        with open(save_dir_fn, "w") as f:
            json.dump(analysis, f)

        if i % 100 == 0:
            print("Number of images processed so far: {}".format(i))


if __name__ == "__main__":
    main()
