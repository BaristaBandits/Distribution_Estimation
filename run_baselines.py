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
    choices=["glove", "word2vec", "gpt2"],
    help="Embedding type"
)

parser.add_argument(
    "--dataset",
    type=str,
    default="wikitext2",
    choices=["wikitext2", "wikitext103", "openwebtext", "wikipedia"],
    help="Dataset"
)

parser.add_argument(
    "--max_sentences",
    type=int,
    default=None,
    help="Limit dataset size (for speed)"
)

parser.add_argument("--run_add", action="store_true")
parser.add_argument("--run_jm", action="store_true")
parser.add_argument("--run_kn", action="store_true")

args = parser.parse_args()

print(f"Using embeddings: {args.embeddings}")
print(f"Using dataset: {args.dataset}")


# =======================
# TOKENIZER MODE (CRITICAL)
# =======================
if args.embeddings == "gpt2":
    tokenizer_mode = "gpt2"
else:
    tokenizer_mode = "word"


# =======================
# LOAD DATA
# =======================
embeddings, stoi, itos = load_embeddings(args.embeddings)

train_corpus, test_corpus = load_text_corpus(
    args.dataset,
    tokenizer_mode=tokenizer_mode,
    max_sentences=args.max_sentences
)


# =======================
# TRACK BEST RESULTS
# =======================
best_add = None
best_jm = None
best_kn = None


# =======================
# ADD-CONSTANT MODEL
# =======================
if args.run_add:
    print("\nRunning Add-Constant")

    model = AddConstantBigram()
    model.fit(train_corpus)

    add_constants = np.arange(0.001, 0.010, 0.001)
    perplexities = []

    print("\nAdd-Constant Results:")
    for c in add_constants:
        model.add_constant = c
        ppl = model.perplexity(test_corpus)
        perplexities.append(ppl)

        print(f"c = {c:.4f} → Perplexity = {ppl:.4f}")

    best_idx = np.argmin(perplexities)
    best_add = (add_constants[best_idx], perplexities[best_idx])

    print(f"\nBest Add-Constant: c = {best_add[0]:.4f}, PPL = {best_add[1]:.4f}")

    plt.figure(figsize=(12, 5))
    plt.plot(add_constants, perplexities, marker='o')
    plt.xlabel("Add Constant")
    plt.ylabel("Perplexity")
    plt.title(f"Add-Constant ({args.dataset}, {args.embeddings})")
    plt.grid(True)
    plt.show()


# =======================
# JELINEK-MERCER
# =======================
if args.run_jm:
    print("\nRunning Jelinek-Mercer")

    jm_model = JelinekMercerSmoothing(train_corpus)

    lambdas = np.arange(0.1, 1.0, 0.1)
    jm_perplexities = []

    print("\nJelinek-Mercer Results:")
    for lam in lambdas:
        jm_model.set_lambda(lam)
        ppl = jm_model.perplexity(test_corpus)
        jm_perplexities.append(ppl)

        print(f"lambda = {lam:.2f} → Perplexity = {ppl:.4f}")

    best_idx = np.argmin(jm_perplexities)
    best_jm = (lambdas[best_idx], jm_perplexities[best_idx])

    print(f"\nBest Lambda: {best_jm[0]:.2f}, PPL = {best_jm[1]:.4f}")

    plt.figure(figsize=(12, 5))
    plt.plot(lambdas, jm_perplexities, marker='o')
    plt.xlabel("Lambda")
    plt.ylabel("Perplexity")
    plt.title(f"Jelinek-Mercer ({args.dataset}, {args.embeddings})")
    plt.grid(True)
    plt.show()


# =======================
# KNESER-NEY
# =======================
if args.run_kn:
    print("\nRunning Kneser-Ney")

    kn_model = BigramKneserNeyNaive()
    kn_model.fit(train_corpus)

    discounts = np.arange(0.4, 1, 0.05)
    kn_perplexities = []

    print("\nKneser-Ney Results:")
    for d in discounts:
        kn_model.discount = d
        ppl = kn_model.perplexity(test_corpus)
        kn_perplexities.append(ppl)

        print(f"discount = {d:.2f} → Perplexity = {ppl:.4f}")

    best_idx = np.argmin(kn_perplexities)
    best_kn = (discounts[best_idx], kn_perplexities[best_idx])

    print(f"\nBest Discount: {best_kn[0]:.2f}, PPL = {best_kn[1]:.4f}")

    plt.figure(figsize=(12, 5))
    plt.plot(discounts, kn_perplexities, marker='o')
    plt.xlabel("Discount")
    plt.ylabel("Perplexity")
    plt.title(f"Kneser-Ney ({args.dataset}, {args.embeddings})")
    plt.grid(True)
    plt.show()


# =======================
# FINAL SUMMARY
# =======================
print("\n===== BEST PERPLEXITIES =====")

if best_add is not None:
    print(f"Add-Constant   → c = {best_add[0]:.4f}, PPL = {best_add[1]:.4f}")

if best_jm is not None:
    print(f"Jelinek-Mercer → lambda = {best_jm[0]:.2f}, PPL = {best_jm[1]:.4f}")

if best_kn is not None:
    print(f"Kneser-Ney     → discount = {best_kn[0]:.2f}, PPL = {best_kn[1]:.4f}")
