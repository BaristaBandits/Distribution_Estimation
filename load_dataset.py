import nltk
nltk.download('punkt_tab')
import re
from datasets import load_dataset

def preprocess_sentence(sentence):
    sentence = sentence.lower()

    # normalize numbers
    sentence = re.sub(r"\d+", "<num>", sentence)

    # remove urls
    sentence = re.sub(r"http\S+|www\S+", "<url>", sentence)

    # remove wikipedia references like [1], [23]
    sentence = re.sub(r"\[\d+\]", "", sentence)

    # separate punctuation
    sentence = re.sub(r"([.,!?;:()\"'])", r" \1 ", sentence)

    # remove non-ascii characters
    sentence = re.sub(r"[^\x00-\x7F]+", " ", sentence)

    # collapse multiple spaces
    sentence = re.sub(r"\s+", " ", sentence)

    return sentence.strip()


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

            # preprocess
            sent = preprocess_sentence(sent)

            # tokenize
            tokens = sent.split()

            # add start and end tokens
            tokens = ["<s>"] + tokens + ["</s>"]

            corpus.append(tokens)

    return corpus

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
