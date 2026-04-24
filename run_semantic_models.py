import numpy as np
import matplotlib.pyplot as plt
import tqdm
import argparse

from load_embed import load_embeddings
from load_dataset import load_text_corpus
from AddConstant import AddConstantBigram
from cache_synonyms import build_cache, Support


# =======================
# ARGUMENTS
# =======================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--embeddings",
    type=str,
    default="glove",
    choices=["glove", "word2vec"],
    help="Embedding type"
)

parser.add_argument(
    "--dataset",
    type=str,
    default="wikitext2",
    choices=["wikitext2", "wikitext103"],
    help="Dataset"
)


args = parser.parse_args()

print(f"Using embeddings: {args.embeddings}")
print(f"Using dataset: {args.dataset}")


# =======================
# LOAD EMBEDDINGS
# =======================
embeddings, stoi, itos = load_embeddings(args.embeddings)


# =======================
# LOAD DATASET
# =======================
train_corpus, test_corpus = load_text_corpus(args.dataset)


# =======================
# BUILD CACHE
# =======================
support = Support(pmin=1e-5)

print("Creating Synonym Cache...")
_, synonym_cache = build_cache(
    train_corpus,
    support,
    embeddings,
    args.k_cache
)


# =======================
# MODEL
# =======================
print("Running Add-Constant Model...")
model = AddConstantBigram(topk_cache=synonym_cache)
model.fit(train_corpus)


# =======================
# EXPERIMENT GRID
# =======================
add_consts = np.arange(0.0005, 0.0009, 0.0001)
k_values = [0, 10, 20, 30, 40, 50]

results = {}


# =======================
# RUN EXPERIMENT
# =======================
for k in tqdm.tqdm(k_values, desc="k values"):
    ppl_list = []

    for c in tqdm.tqdm(add_consts, desc="add_const", leave=False):

        model.add_k = c   

        ppl = model.perplexity(test_corpus, k_syn=k)
        ppl_list.append(ppl)

    results[k] = ppl_list


# =======================
# PLOT
# =======================
plt.figure(figsize=(12, 5))

for k in k_values:
    plt.plot(add_consts, results[k], marker='o', label=f"k={k}")

plt.xlabel("Add Constant")
plt.ylabel("Perplexity")
plt.title(f"PPL vs Add-Constant ({args.dataset}, {args.embeddings})")
plt.legend()
plt.grid(True)

plt.show()
