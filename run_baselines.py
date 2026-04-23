import numpy as np
import matplotlib.pyplot as plt
from load_embed import load_embeddings
from load_dataset import load_text_corpus
from AddConstant import AddConstantBigram
from BaselineModels import JelinekMercerSmoothing, BigramKneserNeyNaive

# load embeddings
embeddings, stoi, itos = load_embeddings("glove")

# load dataset
train_corpus, test_corpus = load_text_corpus("wikitext2")

# run add constant
print("Run add constant")
model = AddConstantBigram()
model.fit(train_corpus)
add_constants = np.arange(0.001, 0.010, 0.001)
perplexities = []
for c in add_constants:
    model.add_constant = c
    ppl = model.perplexity(test_corpus)
    perplexities.append(ppl)

# plot results
plt.figure(figsize=(12,5))
plt.plot(add_constants, perplexities, marker='o')
plt.xlabel("Add Constant")
plt.ylabel("Perplexity")
plt.title("Perplexity vs Add-Constant")
plt.grid(True, which="both")
plt.show()

# run JM
print("Running Jelinek Mercer")
jm_model = JelinekMercerSmoothing(train_corpus)
lambdas = np.arange(0.1, 1.0, 0.1)
jm_perplexities = []
for lam in lambdas:
    jm_model.set_lambda(lam)
    ppl = jm_model.perplexity(test_corpus)
    jm_perplexities.append(ppl)

# plot results
plt.figure(figsize=(12,5))
plt.plot(lambdas, jm_perplexities, marker='o')
plt.xlabel("Lambda")
plt.ylabel("Perplexity")
plt.title("Perplexity vs Lambda")
plt.grid(True, which="both")
plt.show()

# run Knesser Ney
print("Running Knesser Ney")
kn_model = BigramKneserNeyNaive()
kn_model.fit(train_corpus)

discounts = np.arange(0.4, 1, 0.05)
kn_perplexities = []
for d in discounts:
    kn_model.discount = d
    ppl = kn_model.perplexity(test_corpus)
    kn_perplexities.append(ppl)


# plot results
plt.figure(figsize=(12,5))
plt.plot(discounts, kn_perplexities, marker='o')
plt.xlabel("Discount")
plt.ylabel("Perplexity")
plt.title("Perplexity vs Discount")
plt.grid(True, which="both")
plt.show()