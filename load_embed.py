import gensim.downloader as api
from sklearn.decomposition import PCA
import numpy as np
import torch
import os


def load_embeddings(mode="glove", checkpoint_path=None, vector_size=100):

  # GLOVE
  if mode == "glove":
        print("Loading GloVe (100d)...")
        embeddings = api.load("glove-wiki-gigaword-100")
        stoi = {word: i for i, word in enumerate(embeddings.index_to_key)}
        itos = {i: word for word, i in stoi.items()}

        return embeddings, stoi, itos

  #WORD2VEC reduced to 100dims with PCA
  elif mode == "word2vec":
        print("Loading pretrained Word2Vec (Google News 300d)...")
        model = api.load("word2vec-google-news-300")

        words = model.index_to_key
        vectors = np.array([model[word] for word in words])

        print("Reducing to 100 dimensions using PCA...")
        pca = PCA(n_components=vector_size)
        reduced_vectors = pca.fit_transform(vectors)

        embeddings = {word: reduced_vectors[i] for i, word in enumerate(words)}
        stoi = {word: i for i, word in enumerate(words)}
        itos = {i: word for i, word in enumerate(words)}

        return embeddings, stoi, itos
