"""

Code for converting the raw data of OCR json files into sentences. 

The results from images (of which the texts are extracted from) by The NeurIPS 2020 Education Challenge 
can be found in the azua storage blob, under datasets/eedi_comp_tasks_3_4/images_OCR_results.

"""

import argparse
import json
import os
import pickle
import re

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def get_args():
    """
    Parses command line arguments.
    Returns: namespace of command line args.
    """
    parser = argparse.ArgumentParser(
        description="Coverts the text data of the OCR files into fixed sized vectors using transformer-based language models."
    )

    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        default="data/eedi_task_1_2_binary_obs_split",
        help="Directory containing OCR data to use.",
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="data/eedi_task_1_2_binary_obs_split/metadata",
        help="Directory to save the OCR results from the associated images. The OCR results have the name filenames as the original images sources, with only the file extension being changed ",
    )

    args = parser.parse_args()

    # Create required directories
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def get_sentences(args):
    OCR_dir = os.path.join(args.data_dir, "metadata/images_OCR_results")
    OCR_filenames = os.listdir(OCR_dir)

    if "task_1_2" in args.data_dir:
        question_ids_to_original_ids = pd.read_csv(
            os.path.join(args.data_dir, "metadata/question_ids_to_original_ids.csv")
        )
        OCR_filenames = sorted(OCR_filenames)
    else:
        assert "task_3_4" in args.data_dir
        OCR_filenames = sorted(OCR_filenames, key=lambda x: int(x.split(".")[0]))

    save_path = os.path.join(args.output_dir, "sentences.csv")

    if os.path.exists(save_path):
        sentences = pd.read_csv(save_path)
    else:
        sentences = dict()
        for filename in tqdm(OCR_filenames):
            if "task_1_2" in args.data_dir:
                if not (filename.replace("json", "jpg") == question_ids_to_original_ids["OriginalQuestionId"]).any():
                    continue

            sentences_in_question = list()
            filepath = os.path.join(OCR_dir, filename)
            analysis = json.load(open(filepath))

            # For future sanity check
            # assert analysis['language'] == 'en'

            # Extract 'sentences' from the OCR.
            region_infos = [region["lines"] for region in analysis["regions"]]
            word_infos = []
            for region in region_infos:
                word_infos_in_region = []
                for line in region:
                    line_boundingbox_y = int(line["boundingBox"].split(",")[1])
                    if line_boundingbox_y < 50 or line_boundingbox_y > 900:
                        # Remove the banner texts, e.g., "eedi, edexcel, ..."
                        if len(line["words"]) <= 3:
                            continue
                        potential_line_tobe_removed = " ".join([v["text"] for v in line["words"]]).lower()
                        if "copyright" in potential_line_tobe_removed:
                            continue
                        if "pearson" in potential_line_tobe_removed:
                            continue
                    if 75 <= line_boundingbox_y <= 79 or 680 <= line_boundingbox_y <= 700:
                        # Further remove the banner texts"
                        potential_line_tobe_removed = " ".join([v["text"] for v in line["words"]]).lower()
                        if "realising" in potential_line_tobe_removed:
                            continue
                        if "potential" in potential_line_tobe_removed:
                            continue
                        if "copynght" in potential_line_tobe_removed:
                            continue

                    for word_info in line["words"]:
                        word_infos_in_region.append(word_info)
                word_infos.append(word_infos_in_region)

            for potential_sentence in word_infos:
                sentence_list = []
                n_words = 0
                for word in potential_sentence:
                    text = word["text"]
                    if (
                        text not in ["Eedi", "O", "o", "2018"]
                        and "eedi" not in text.lower()
                        and "rosemaths" not in text.lower()
                        and "edexcel" not in text.lower()
                    ):
                        sentence_list.append(text)
                        alphabets = re.sub("[^a-zA-Z_]+", "", text)
                        if alphabets != "":
                            n_words += 1

                if n_words > 5:
                    sentence = " ".join(sentence_list)
                    sentences_in_question.append(sentence)

            # Use . as a delimiter
            sentences[filename] = ".".join(sentences_in_question)

        sentences_df = pd.DataFrame.from_dict(sentences.items())
        sentences_df.columns = ["OriginalQuestionId", "Sentences"]

        sentences_df.to_csv(save_path, index=None)
    return sentences


def main():
    args = get_args()

    sentences = get_sentences(args)


if __name__ == "__main__":
    main()
