from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2Model


def load_embeddings(mode="glove", checkpoint_path=None, vector_size=100):

    # =======================
    # GLOVE
    # =======================
    if mode == "glove":
        print("Loading GloVe (100d)...")
        kv = api.load("glove-wiki-gigaword-100")  # already KeyedVectors

        stoi = {word: i for i, word in enumerate(kv.index_to_key)}
        itos = {i: word for word, i in stoi.items()}

        return kv, stoi, itos


    # =======================
    # WORD2VEC
    # =======================
    elif mode == "word2vec":
        print("Loading pretrained Word2Vec (Google News 300d)...")

        kv = api.load("word2vec-google-news-300")  # already KeyedVectors

        stoi = {word: i for i, word in enumerate(kv.index_to_key)}
        itos = {i: word for word, i in stoi.items()}

        return kv, stoi, itos


    # =======================
    # GPT-2 → CONVERT TO KeyedVectors
    # =======================
    elif mode == "gpt2":
        print("Loading GPT-2 embeddings (768d)...")

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2Model.from_pretrained("gpt2")

        embedding_matrix = model.get_input_embeddings().weight.detach().cpu().numpy()

        vocab = tokenizer.get_vocab()  # token -> id

        # Sort tokens by index (important!)
        sorted_items = sorted(vocab.items(), key=lambda x: x[1])
        tokens = [token for token, _ in sorted_items]

        vectors = embedding_matrix[[idx for _, idx in sorted_items]]

        # Create KeyedVectors
        kv = KeyedVectors(vector_size=vectors.shape[1])
        kv.add_vectors(tokens, vectors)

        stoi = {word: i for i, word in enumerate(kv.index_to_key)}
        itos = {i: word for word, i in stoi.items()}

        return kv, stoi, itos
