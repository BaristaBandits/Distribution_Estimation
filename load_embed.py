import gensim.downloader as api
from sklearn.decomposition import PCA
import numpy as np
import torch
import os
from transformers import GPT2Tokenizer, GPT2Model

def load_embeddings(mode="glove", checkpoint_path=None, vector_size=100):

  # GLOVE
  if mode == "glove":
        print("Loading GloVe (100d)...")
        embeddings = api.load("glove-wiki-gigaword-100")
        stoi = {word: i for i, word in enumerate(embeddings.index_to_key)}
        itos = {i: word for word, i in stoi.items()}

        return embeddings, stoi, itos

  elif mode == "word2vec":
    print("Loading pretrained Word2Vec (Google News 300d)...")
    model = api.load("word2vec-google-news-300")

    words = model.index_to_key
    vectors = np.array([model[word] for word in words])

    # No PCA — keep original 300d vectors
    embeddings = {word: vectors[i] for i, word in enumerate(words)}
    stoi = {word: i for i, word in enumerate(words)}
    itos = {i: word for i, word in enumerate(words)}

    return embeddings, stoi, itos

  elif mode == "gpt2":
    print("Loading GPT-2 embeddings (768d)...")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")

    embedding_matrix = model.get_input_embeddings().weight.detach().cpu().numpy()

    vocab = tokenizer.get_vocab()  # token -> id

    stoi = vocab
    itos = {i: t for t, i in vocab.items()}

    # Build embeddings dict
    embeddings = {token: embedding_matrix[idx] for token, idx in vocab.items()}

    return embeddings, stoi, itos
