import numpy as np
import matplotlib.pyplot as plt
import tqdm
from load_embed import load_embeddings
from load_dataset import load_text_corpus
from AddConstant import AddConstantBigram
from cache_synonyms import build_cache, Support

# load embeddings
embeddings, stoi, itos = load_embeddings("glove")

# load dataset
train_corpus, test_corpus = load_text_corpus("wikitext2")

# create synonym cache
support = Support(pmin=1e-5)
print("Creating Synonym Cache")
_, synonym_cache = build_cache(train_corpus, Support, embeddings, 50)

# run semantic add constant
print("Running for Add Constant")
model = AddConstantBigram()
model.fit(train_corpus)


add_consts = np.arange(0.0005, 0.0009, 0.0001)
k_values = [0, 10, 20, 30, 40, 50]
perplexities = []

results = {}

for k in tqdm(k_values, desc="k values"):
    ppl_list = []
    for c in tqdm(add_consts, desc="add_const", leave=False):
        ppl = model.perplexity(test_corpus, k_syn=k)
        ppl_list.append(ppl)

    results[k] = ppl_list

# plot results
plt.figure(figsize=(12,5))

for k in k_values:
    plt.plot(add_consts, results[k], marker='o', label=f"k={k}")

plt.xlabel("Add k")
plt.ylabel("Perplexity")
plt.title("Perplexity vs Add-Constant for Different k")
plt.legend()
plt.grid(True)

plt.show()