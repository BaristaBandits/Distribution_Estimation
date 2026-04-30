# =======================
# FORCE PGF BACKEND
# =======================
import matplotlib
matplotlib.use("pgf")

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import argparse
import os
import pickle

from load_embed import load_embeddings
from load_dataset import load_text_corpus
from AddConstant import AddConstantBigram
from cache_synonyms import build_cache, Support
from SemanticKN import SemanticBigramKneserNey


# =======================
# HIGH-QUALITY PLOT SETTINGS
# =======================
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "lines.linewidth": 2,
    "lines.markersize": 6,
})

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "pgf.rcfonts": False,
})


# =======================
# STYLE (IEEE)
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
style_path = os.path.join(BASE_DIR, "IEEEstyle.mplstyle")
plt.style.use(style_path)

CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# =======================
# ARGUMENTS
# =======================
parser = argparse.ArgumentParser()
parser.add_argument("--max_sentences", type=int, default=100000)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--k_cache", type=int, default=50)
args = parser.parse_args()


# =======================
# SETTINGS
# =======================
k_values = [0, 5, 10, 20, 30, 40, 50]
emb_list = ["glove", "word2vec", "gpt2"]

ADD_CONST_LIST =  [0.0002, 0.0003, 0.0004]
KN_DISCOUNT_LIST = [0.6, 0.7, 0.8]


# =======================
# RESULTS STORAGE
# =======================
results = {
    "add": {emb: {c: [] for c in ADD_CONST_LIST} for emb in emb_list},
    "kn": {emb: {d: [] for d in KN_DISCOUNT_LIST} for emb in emb_list}
}


# =======================
# CACHE HELPERS
# =======================
def cache_path(emb_name):
    return os.path.join(CACHE_DIR, f"{emb_name}_cache.pkl")


def load_or_build_cache(emb_name, train_corpus, embeddings):
    path = cache_path(emb_name)

    if os.path.exists(path):
        print(f"Loading cache for {emb_name}...")
        with open(path, "rb") as f:
            d_estimate, synonym_cache = pickle.load(f)
    else:
        print(f"Building cache for {emb_name}...")
        support = Support(pmin=1e-5)

        d_estimate, synonym_cache = build_cache(
            train_corpus,
            support,
            embeddings,
            args.k_cache
        )

        with open(path, "wb") as f:
            pickle.dump((d_estimate, synonym_cache), f)

        print(f"Saved cache → {path}")

    return d_estimate, synonym_cache


# =======================
# MAIN LOOP
# =======================
for emb_name in emb_list:
    print(f"\n===== Running for {emb_name} =====")

    embeddings, stoi, itos = load_embeddings(emb_name)

    tokenizer_mode = "gpt2" if emb_name == "gpt2" else "word"

    train_corpus, test_corpus = load_text_corpus(
        "wikitext103",
        tokenizer_mode=tokenizer_mode,
        max_sentences=args.max_sentences
    )

    d_estimate, synonym_cache = load_or_build_cache(
        emb_name,
        train_corpus,
        embeddings
    )

    # =======================
    # ADD-CONSTANT
    # =======================
    print("Evaluating Add-Constant...")

    for c in ADD_CONST_LIST:
        print(f"  c = {c}")

        add_model = AddConstantBigram(topk_cache=synonym_cache)
        add_model.fit(train_corpus)
        add_model.add_constant = c

        for k in tqdm.tqdm(k_values):
            ppl = add_model.perplexity(test_corpus, k_syn=k, beta=args.beta)
            results["add"][emb_name][c].append(ppl)

    # =======================
    # KNESER-NEY
    # =======================
    print("Evaluating Kneser-Ney...")

    for d in KN_DISCOUNT_LIST:
        print(f"  d = {d}")

        kn_model = SemanticBigramKneserNey(
            discount=d,
            topk_cache=synonym_cache,
            d_estimate=d_estimate
        )

        kn_model.fit(train_corpus)

        for k in tqdm.tqdm(k_values):
            ppl = kn_model.perplexity(test_corpus, k_syn=k, beta=args.beta)
            results["kn"][emb_name][d].append(ppl)


# =======================
# PLOTTING
# =======================

# -------- ADD CONSTANT --------
for emb in emb_list:
    plt.figure(figsize=(6, 4))

    for c in ADD_CONST_LIST:
        plt.plot(
            k_values,
            results["add"][emb][c],
            marker='o',
            linewidth=2,
            markersize=6,
            label=f"c = {c}"
        )

    plt.title(f"{emb.upper()} - Add Constant")
    plt.xlabel("Number of Synonyms (k)")
    plt.ylabel("Perplexity")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout(pad=0.5)
    plt.savefig(f"{emb}_add_constants.pgf")
    plt.savefig(f"{emb}_add_constants.pdf")
    plt.close()


# -------- KN --------
for emb in emb_list:
    plt.figure(figsize=(6, 4))

    for d in KN_DISCOUNT_LIST:
        plt.plot(
            k_values,
            results["kn"][emb][d],
            marker='s',
            linewidth=2,
            markersize=6,
            label=f"d = {d}"
        )

    plt.title(f"{emb.upper()} - Kneser-Ney")
    plt.xlabel("Number of Synonyms (k)")
    plt.ylabel("Perplexity")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout(pad=0.5)
    plt.savefig(f"{emb}_kn_discounts.pgf")
    plt.savefig(f"{emb}_kn_discounts.pdf")
    plt.close()


# =======================
# TABLES
# =======================

# -------- ADD-CONSTANT TABLE --------
print("\n===== ADD-CONSTANT RESULTS =====")

for emb in emb_list:
    print(f"\n--- {emb.upper()} ---")

    header = "k | " + " | ".join([f"c={c}" for c in ADD_CONST_LIST])
    print(header)
    print("-" * len(header))

    for i, k in enumerate(k_values):
        row = f"{k:2d} | "
        row += " | ".join([
            f"{results['add'][emb][c][i]:8.2f}" for c in ADD_CONST_LIST
        ])
        print(row)


# -------- KN TABLE --------
print("\n===== KNESER-NEY RESULTS =====")

for emb in emb_list:
    print(f"\n--- {emb.upper()} ---")

    header = "k | " + " | ".join([f"d={d}" for d in KN_DISCOUNT_LIST])
    print(header)
    print("-" * len(header))

    for i, k in enumerate(k_values):
        row = f"{k:2d} | "
        row += " | ".join([
            f"{results['kn'][emb][d][i]:8.2f}" for d in KN_DISCOUNT_LIST
        ])
        print(row)


# =======================
# DONE
# =======================
print("\nAll experiments complete.")
