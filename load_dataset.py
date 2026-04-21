import nltk
nltk.download('punkt_tab')
import re


def build_sentence_corpus(text_list):
    corpus = []

    for text in text_list:

        text = text.strip()
        # sentence tokenize
        if not text or text.startswith("="):
            continue

        sentences = nltk.sent_tokenize(text)

        for sent in sentences:
            sent = sent.strip().lower()

            # tokenize
            tokens = sent.split()

            # add start and end tokens
            tokens = ["bos"] + tokens + ["eos"]

            corpus.append(tokens)

    return corpus

from datasets import load_dataset

def load_text_corpus(dataset_name="wikitext2"):

    # ------------------ LOAD DATA ------------------
    if dataset_name == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_text = dataset["train"]["text"]
        test_text = dataset["test"]["text"]
        train_corpus = build_sentence_corpus(train_text)
        test_corpus = build_sentence_corpus(test_text)

    elif dataset_name == "wikitext103":
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
        train_text = dataset["train"]["text"]
        test_text = dataset["test"]["text"]
        train_corpus = build_sentence_corpus(train_text)[:100000]
        test_corpus = build_sentence_corpus(test_text)

    return train_corpus, test_corpus
