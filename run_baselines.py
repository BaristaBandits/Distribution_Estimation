import numpy as np
import matplotlib.pyplot as plt
import argparse

from load_embed import load_embeddings
from load_dataset import load_text_corpus
from AddConstant import AddConstantBigram
from BaselineModels import JelinekMercerSmoothing, BigramKneserNeyNaive


# =======================
# ARGUMENT PARSER
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

parser.add_argument("--run_add", action="store_true", help="Run Add-Constant")
parser.add_argument("--run_jm", action="store_true", help="Run Jelinek-Mercer")
parser.add_argument("--run_kn", action="store_true", help="Run Kneser-Ney")

args = parser.parse_args()

print(f"Using embeddings: {args.embeddings}")
print(f"Using dataset: {args.dataset}")


# =======================
# LOAD DATA
# =======================
embeddings, stoi, itos = load_embeddings(args.embeddings)
train_corpus, test_corpus = load_text_corpus(args.dataset)


# =======================
# ADD-CONSTANT MODEL
# =======================
if args.run_add:
    print("Running Add-Constant")

    model = AddConstantBigram()
    model.fit(train_corpus)

    add_constants = np.arange(0.001, 0.010, 0.001)
    perplexities = []

    for c in add_constants:
        model.add_constant = c
        ppl = model.perplexity(test_corpus)
        perplexities.append(ppl)

    plt.figure(figsize=(12, 5))
    plt.plot(add_constants, perplexities, marker='o')
    plt.xlabel("Add Constant")
    plt.ylabel("Perplexity")
    plt.title(f"Add-Constant ({args.dataset})")
    plt.grid(True)
    plt.show()


# =======================
# JELINEK-MERCER
# =======================
if args.run_jm:
    print("Running Jelinek-Mercer")

    jm_model = JelinekMercerSmoothing(train_corpus)

    lambdas = np.arange(0.1, 1.0, 0.1)
    jm_perplexities = []

    for lam in lambdas:
        jm_model.set_lambda(lam)
        ppl = jm_model.perplexity(test_corpus)
        jm_perplexities.append(ppl)

    plt.figure(figsize=(12, 5))
    plt.plot(lambdas, jm_perplexities, marker='o')
    plt.xlabel("Lambda")
    plt.ylabel("Perplexity")
    plt.title(f"Jelinek-Mercer ({args.dataset})")
    plt.grid(True)
    plt.show()


# =======================
# KNESER-NEY
# =======================
if args.run_kn:
    print("Running Kneser-Ney")

    kn_model = BigramKneserNeyNaive()
    kn_model.fit(train_corpus)

    discounts = np.arange(0.4, 1, 0.05)
    kn_perplexities = []

    for d in discounts:
        kn_model.discount = d
        ppl = kn_model.perplexity(test_corpus)
        kn_perplexities.append(ppl)

    plt.figure(figsize=(12, 5))
    plt.plot(discounts, kn_perplexities, marker='o')
    plt.xlabel("Discount")
    plt.ylabel("Perplexity")
    plt.title(f"Kneser-Ney ({args.dataset})")
    plt.grid(True)
    plt.show()
