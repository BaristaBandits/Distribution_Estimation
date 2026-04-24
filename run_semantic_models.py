import numpy as np
import matplotlib.pyplot as plt
import tqdm
import argparse

from load_embed import load_embeddings
from load_dataset import load_text_corpus
from AddConstant import AddConstantBigram
from cache_synonyms import build_cache, Support
from SemanticKN import SemanticBigramKneserNey


# =======================
# ARGUMENTS
# =======================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--embeddings",
    type=str,
    default="glove",
    choices=["glove", "word2vec"],
)

parser.add_argument(
    "--dataset",
    type=str,
    default="wikitext2",
    choices=["wikitext2", "wikitext103"],
)

parser.add_argument("--k_cache", type=int, default=50)
parser.add_argument("--run_add", action="store_true")
parser.add_argument("--run_kn", action="store_true")

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
d_estimate, synonym_cache = build_cache(
    train_corpus,
    support,
    embeddings,
    args.k_cache
)


# =========================================================
#  1. SEMANTIC ADD-CONSTANT
# =========================================================
if args.run_add:
    print("\nRunning Semantic Add-Constant Model...")

    model = AddConstantBigram(topk_cache=synonym_cache)
    model.fit(train_corpus)

    add_consts = np.arange(0.0005, 0.0009, 0.0001)
    k_values = [0, 10, 20, 30, 40, 50]

    results_add = {}

    for k in tqdm.tqdm(k_values, desc="k values"):
        ppl_list = []

        for c in tqdm.tqdm(add_consts, desc="add_const", leave=False):
            model.add_constant = c   # FIXED name
            ppl = model.perplexity(test_corpus, k_syn=k)
            ppl_list.append(ppl)

        results_add[k] = ppl_list

    # Plot
    plt.figure(figsize=(12, 5))
    for k in k_values:
        plt.plot(add_consts, results_add[k], marker='o', label=f"k={k}")

    plt.xlabel("Add Constant")
    plt.ylabel("Perplexity")
    plt.title(f"Semantic Add-Constant ({args.dataset}, {args.embeddings})")
    plt.legend()
    plt.grid(True)
    plt.show()


    print("\n===== BEST ADD-CONSTANT PER k =====")

    for k in k_values:
        ppl_list = results_add[k]
        best_idx = np.argmin(ppl_list)
    
        best_c = add_consts[best_idx]
        best_ppl = ppl_list[best_idx]
    
        print(f"k = {k:2d} → best c = {best_c:.6f}, PPL = {best_ppl:.4f}")


# =========================================================
# 2. SEMANTIC KNESER-NEY (NEW)
# =========================================================
if args.run_kn:
    print("\nRunning Semantic Kneser-Ney...")

    discounts = np.arange(0.5, 0.95, 0.05)
    k_values = [0, 5, 10, 20, 30, 50]

    # store as: {k: [ppl_d1, ppl_d2, ...]}
    results_kn = {k: [] for k in k_values}

    for d in discounts:
        print(f"\nTraining model with discount={d:.2f}")

        model = SemanticBigramKneserNey(
            discount=d,
            topk_cache=synonym_cache,
            d_estimate=d_estimate
        )

        model.fit(train_corpus)

        for k in k_values:
            ppl = model.perplexity(test_corpus, k_syn=k)
            print(f"  k={k} → PPL={ppl:.4f}")
            results_kn[k].append(ppl)

    # =======================
    # PLOT: PPL vs DISCOUNT
    # =======================
    plt.figure(figsize=(12, 5))

    for k in k_values:
        plt.plot(discounts, results_kn[k], marker='o', label=f"k={k}")

    plt.xlabel("Discount")
    plt.ylabel("Perplexity")
    plt.title(f"Semantic Kneser-Ney ({args.dataset}, {args.embeddings})")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n===== BEST DISCOUNT PER k (Semantic KN) =====")

    for i, k in enumerate(k_values):
        best_ppl = float("inf")
        best_d = None
    
        for d in discounts:
            ppl = results_kn[d][i]
    
            if ppl < best_ppl:
                best_ppl = ppl
                best_d = d
    
        print(f"k = {k:2d} → best discount = {best_d:.2f}, PPL = {best_ppl:.4f}")
