"""

Code for converting the natural language sentences into fixed sized vectors using 
    1) sentence transformer-based language models, or
    2) TF-IDF.

To run this code, the transformers, sentence_transformers, and gensim module needs to be installed:

https://huggingface.co/
https://www.sbert.net/index.html
https://github.com/RaRe-Technologies/gensim

The results from images (of which the texts are extracted from) by The NeurIPS 2020 Education Challenge 
can be found in the azua storage blob, under datasets/eedi_comp_tasks_3_4/images_OCR_results.

"""

import torch
import argparse
import json
import os
import pickle
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from process_raw_OCR_to_sentence import get_sentences
from transformers import BertModel
from transformers import BertTokenizer
import gensim.downloader as api


def get_args():
    """
    Parses command line arguments.
    Returns: namespace of command line args.
    """
    parser = argparse.ArgumentParser(
        description="Converts the text data of the OCR files into fixed sized vectors using transformer-based language models."
    )

    parser.add_argument(
        "--data_dir", "-d", type=str, default="data/goodreads", help="Directory containing OCR data to use.",
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="data/goodreads/metadata",
        help="Directory to save the OCR results from the associated images. The OCR results have the name filenames as the original images sources, with only the file extension being changed ",
    )

    parser.add_argument(
        "--transform_using",
        "-t",
        type=str,
        choices=["sbert", "tfidf", "bert_cls", "bert_average", "neural_bow"],
        help="Choose which method to use to transform the text into a fixed sized vector.",
    )

    args = parser.parse_args()

    # Create required directories
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def transform_sentence_csv_to_list():
    args = get_args()
    sentences = get_sentences(args)
    if "task_1_2" in args.data_dir:
        question_ids_to_original_ids = pd.read_csv(
            os.path.join(args.data_dir, "metadata/question_ids_to_original_ids.csv")
        )

        sentences = sentences.set_index("OriginalQuestionId").reindex(
            question_ids_to_original_ids.OriginalQuestionId.apply(lambda x: x.replace("jpg", "json")), fill_value=np.nan
        )
        sentences = list(sentences.Sentences.replace(np.nan, ""))
    else:
        sentences = list(sentences.Sentences.replace(np.nan, ""))

    sentences_array = np.array(sentences)
    save_path = os.path.join(args.data_dir, "metadata/sentences_array.npy")
    if not os.path.exists(save_path):
        np.save(save_path, sentences_array)

    return sentences


def get_embeddings():
    args = get_args()
    sentences = transform_sentence_csv_to_list()

    if args.transform_using == "sbert":
        # sBERT model: there are different pretrained language models to select from.
        # Because of some weird error in sentence-transformers, the model was manually downloaded
        # https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/
        model = SentenceTransformer("data/sentence-transformers/paraphrase-distilroberta-base-v1")

        with torch.no_grad():
            embeddings = model.encode(sentences)

        save_path = os.path.join(args.output_dir, "item_metadata_sbert.npy")
        np.save(save_path, embeddings)
    elif args.transform_using in ("bert_cls", "bert_average"):
        transformer_model_id = "bert-base-cased"
        max_length = 64
        tokenizer = BertTokenizer.from_pretrained(transformer_model_id)
        model = BertModel.from_pretrained(transformer_model_id)

        transformer_tokenized = tokenizer(
            sentences, return_tensors="pt", truncation=True, padding=True, max_length=max_length,
        )
        with torch.no_grad():
            transformer_output = model(**transformer_tokenized)[0]

        attention_mask = transformer_tokenized["attention_mask"]

        if args.transform_using == "bert_cls":
            embeddings = np.array(transformer_output[:, 0])
            save_path = os.path.join(args.output_dir, "item_metadata_bert_cls.npy")
            np.save(save_path, embeddings)
        else:
            embeddings = np.array(
                [np.array(torch.mean(x[: sum(y).item()], axis=0)) for (x, y) in zip(transformer_output, attention_mask)]
            )
            save_path = os.path.join(args.output_dir, "item_metadata_bert_average.npy")
            np.save(save_path, embeddings)
    elif args.transform_using == "neural_bow":
        model = api.load("word2vec-google-news-300")
        tmp = [[re.sub("[\W_]+", "", word) for word in sentence_.split()] for sentence_ in sentences]
        embeddings = np.array(
            [
                np.mean([model[word] for word in sentence_ if word in model], axis=0) if sentence_ else np.zeros(300)
                for sentence_ in tmp
            ]
        )
        save_path = os.path.join(args.output_dir, "item_metadata_neural_bow.npy")
        np.save(save_path, embeddings)

    else:
        assert args.transform_using == "tfidf"

        # There are different pre-processing steps and different parameter settings for tfidf that can be taken.
        # Here, I only filtered words to be in alphanumeric characters only, and stemmed words, and lowercased them.
        # For filtering non-alphabet characters, use the below code instead.
        # sentences = [' '.join([ps.stem(re.sub('[^a-zA-Z_]+', '', word.lower())) for word in sentence.split()]) for sentence in sentences]
        ps = PorterStemmer()
        sentences = [
            " ".join([ps.stem(re.sub("[\W_]+", "", word.lower())) for word in sentence.split()])
            for sentence in sentences
        ]

        # Also, I removed tokens that appeared in less than 2 documents (i.e., df < 2).
        vectorizer = TfidfVectorizer(stop_words="english", min_df=2)

        # For EEDI dataset, the total number of tokens after these steps becomes about 200.
        embeddings = np.array(vectorizer.fit_transform(sentences).todense(), dtype=np.float)

        tfidf_features = vectorizer.get_feature_names()
        # item_nodeids = list(item_metadata_dict.keys())
        item_nodeids = list(range(len(sentences)))

        item_metadata_dict_for_df = {"QuestionId": list(), "Texts": list()}
        for i in range(len(sentences)):
            item_nodeid = item_nodeids[i]
            texts = ", ".join([v for v in sentences[i].split() if v in tfidf_features])
            item_metadata_dict_for_df["QuestionId"].append(item_nodeid)
            item_metadata_dict_for_df["Texts"].append(texts)

        item_metadata_df = pd.DataFrame.from_dict(item_metadata_dict_for_df)

        save_path_df = os.path.join(args.output_dir, "item_metadata_text.csv")
        item_metadata_df.to_csv(save_path_df, index=None)

        save_path_array = os.path.join(args.output_dir, "item_metadata_text.npy")
        np.save(save_path_array, embeddings)


def main():
    embeddings = get_embeddings()


if __name__ == "__main__":
    main()
