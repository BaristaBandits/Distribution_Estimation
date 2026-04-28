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
    choices=["glove", "word2vec", "gpt2"],
)

parser.add_argument(
    "--dataset",
    type=str,
    default="wikitext2",
    choices=["wikitext2", "wikitext103", "openwebtext", "wikipedia"],
)

parser.add_argument("--k_cache", type=int, default=50)
parser.add_argument("--max_sentences", type=int, default=100000)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--run_add", action="store_true")
parser.add_argument("--run_kn", action="store_true")

args = parser.parse_args()

print(f"Using embeddings: {args.embeddings}")
print(f"Using dataset: {args.dataset}")


# =======================
# TOKENIZER MODE
# =======================
if args.embeddings == "gpt2":
    tokenizer_mode = "gpt2"
else:
    tokenizer_mode = "word"


# =======================
# LOAD EMBEDDINGS
# =======================
embeddings, stoi, itos = load_embeddings(args.embeddings)


# =======================
# LOAD DATASET
# =======================
train_corpus, test_corpus = load_text_corpus(
    args.dataset,
    tokenizer_mode=tokenizer_mode,
    max_sentences=args.max_sentences
)


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
# 1. SEMANTIC ADD-CONSTANT
# =========================================================
if args.run_add:
    print("\nRunning Add-Constant β sweep...")

    model = AddConstantBigram(topk_cache=synonym_cache)
    model.fit(train_corpus)

    betas = np.logspace(-3, 1, 10)
    k_values = [10, 30, 50]

    FIXED_C = 0.0005
    model.add_constant = FIXED_C

    results_add_beta = {k: [] for k in k_values}

    for beta in tqdm.tqdm(betas, desc="beta sweep"):
        for k in k_values:
            ppl = model.perplexity(
                test_corpus,
                k_syn=k,
                beta=beta
            )
            results_add_beta[k].append(ppl)

    # =======================
    # PLOT
    # =======================
    plt.figure(figsize=(10, 5))

    for k in k_values:
        plt.plot(betas, results_add_beta[k], marker='o', label=f"k={k}")

    plt.xscale("log")
    plt.xlabel("Beta (softmin temperature)")
    plt.ylabel("Perplexity")
    plt.title(f"Add-Constant β Sweep ({args.dataset}, {args.embeddings})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # =======================
    # BEST β
    # =======================
    print("\n===== BEST β (Add-Constant) =====")

    for k in k_values:
        ppl_list = results_add_beta[k]
        best_idx = np.argmin(ppl_list)

        print(f"k={k} → best β={betas[best_idx]:.4f}, PPL={ppl_list[best_idx]:.2f}")

# =========================================================
# 2. SEMANTIC KNESER-NEY
# =========================================================
if args.run_kn:
    print("\nRunning Kneser-Ney β sweep...")

    betas = np.logspace(-3, 1, 10)
    k_values = [10, 30, 50]

    FIXED_D = 0.85

    model = SemanticBigramKneserNey(
        discount=FIXED_D,
        topk_cache=synonym_cache,
        d_estimate=d_estimate
    )

    model.fit(train_corpus)

    results_kn_beta = {k: [] for k in k_values}

    for beta in tqdm.tqdm(betas, desc="beta sweep"):
        for k in k_values:
            ppl = model.perplexity(
                test_corpus,
                k_syn=k,
                beta=beta
            )
            results_kn_beta[k].append(ppl)

    # =======================
    # PLOT
    # =======================
    plt.figure(figsize=(10, 5))

    for k in k_values:
        plt.plot(betas, results_kn_beta[k], marker='o', label=f"k={k}")

    plt.xscale("log")
    plt.xlabel("Beta (softmin temperature)")
    plt.ylabel("Perplexity")
    plt.title(f"Kneser-Ney β Sweep ({args.dataset}, {args.embeddings})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # =======================
    # BEST β
    # =======================
    print("\n===== BEST β (KN) =====")

    for k in k_values:
        ppl_list = results_kn_beta[k]
        best_idx = np.argmin(ppl_list)

        print(f"k={k} → best β={betas[best_idx]:.4f}, PPL={ppl_list[best_idx]:.2f}")
