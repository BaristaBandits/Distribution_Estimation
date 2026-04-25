import nltk
nltk.download('punkt_tab')
import re
from datasets import load_dataset
from transformers import GPT2Tokenizer

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

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


def build_sentence_corpus(text_list, tokenizer_mode="word", max_sentences=100000):
    corpus = []

    for text in text_list:
        if not text or text.startswith("="):
            continue

        sentences = nltk.sent_tokenize(text)

        for sent in sentences:
            sent = preprocess_sentence(sent)

            if tokenizer_mode == "gpt2":
                tokens = gpt2_tokenizer.tokenize(sent)
            else:
                tokens = sent.split()

            if len(tokens) == 0:
                continue

            tokens = ["<s>"] + tokens + ["</s>"]
            corpus.append(tokens)
            
            if max_sentences and len(corpus) >= max_sentences:
                return corpus

    return corpus

def load_text_corpus(dataset_name="wikitext2", tokenizer_mode="word", max_sentences=100000):   

    # ------------------ LOAD DATA ------------------
    if dataset_name == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

        train_text = dataset["train"]["text"]
        test_text = dataset["test"]["text"]

    elif dataset_name == "wikitext103":
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

        train_text = dataset["train"]["text"]
        test_text = dataset["test"]["text"]

    elif dataset_name == "openwebtext":
        dataset = load_dataset("openwebtext")

        train_text = dataset["train"]["text"]

        # split manually (since no official test split)
        split_idx = int(0.95 * len(train_text))
        test_text = train_text[split_idx:]
        train_text = train_text[:split_idx]

    elif dataset_name == "wikipedia":
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True
        )
    
        text_data = []
        for i, example in enumerate(dataset):
            text_data.append(example["text"])
            if i >= 100000:   # control size
                break
    
        split_idx = int(0.9 * len(text_data))
    
        train_text = text_data[:split_idx]
        test_text = text_data[split_idx:]
    
        train_corpus = build_sentence_corpus(train_text)
        test_corpus = build_sentence_corpus(test_text)

    else:
        raise ValueError("Unsupported dataset")

    # ------------------ BUILD CORPUS ------------------
    print("Tokenizing train corpus...")
    train_corpus = build_sentence_corpus(train_text, tokenizer_mode, max_sentences)

    print("Tokenizing test corpus...")
    test_corpus = build_sentence_corpus(test_text, tokenizer_mode, max_sentences)

    # ------------------ OPTIONAL LIMIT ------------------

    train_corpus = train_corpus[:max_sentences]
    test_corpus = test_corpus[:max_sentences//10]
    print('train size :', len(train_corpus))
    print('test size :', len(test_corpus))

    return train_corpus, test_corpus
