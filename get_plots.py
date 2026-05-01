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
# STYLE (IEEE ONLY SOURCE)
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
style_path = os.path.join(BASE_DIR, "IEEEstyle.mplstyle")

plt.style.use(style_path)


# =======================
# CACHE DIR
# =======================
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# =======================
# LINE STYLE FUNCTION
# =======================
def construct_properties_dict(names, labels):
    colors = ['blue', 'brown', 'purple', 'darkorange', 'green', 'red']
    markers = ['o', '^', 's', 'P', 'X', '*']

    properties_dict = {}

    for i, name in enumerate(names):
        # Uniform style per embedding
        if "glove" in name:
            linestyle = "solid"
        elif "word2vec" in name:
            linestyle = "dashed"
        else:
            linestyle = "dashdot"

        properties_dict[name] = {
            'color': colors[i % len(colors)],
            'linestyle': linestyle,
            'marker': markers[i % len(markers)],
            'label': labels[i]
        }

    return properties_dict


# =======================
# ARGUMENTS
# =======================
parser = argparse.ArgumentParser()
parser.add_argument("--max_sentences", type=int, default=100000)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--k_cache", type=int, default=50)

parser.add_argument("--add_consts", nargs="+", type=float, default=[0.001, 0.005, 0.007])
parser.add_argument("--discounts", nargs="+", type=float, default=[0.6, 0.7, 0.8])

args = parser.parse_args()


# =======================
# SETTINGS
# =======================
m_values = [0, 5, 10, 20, 30, 40, 50]
emb_list = ["glove", "word2vec", "gpt2"]

ADD_CONST_LIST = args.add_consts
KN_DISCOUNT_LIST = args.discounts


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

    # -------- ADD-CONSTANT --------
    print("Evaluating Additive Smoothing...")

    for c in ADD_CONST_LIST:
        print(f"  Add = {c}")

        model = AddConstantBigram(
            add_constant=c,
            topk_cache=synonym_cache,
            d_estimate=d_estimate
        )

        model.fit(train_corpus)

        for m in tqdm.tqdm(m_values):
            ppl = model.perplexity(test_corpus, k_syn=m)
            results["add"][emb_name][c].append(ppl)

    # -------- KNESER-NEY --------
    print("Evaluating Kneser-Ney...")

    for d in KN_DISCOUNT_LIST:
        print(f"  Discount = {d}")

        model = SemanticBigramKneserNey(
            discount=d,
            topk_cache=synonym_cache,
            d_estimate=d_estimate
        )

        model.fit(train_corpus)

        for m in tqdm.tqdm(m_values):
            ppl = model.perplexity(test_corpus, k_syn=m, beta=args.beta)
            results["kn"][emb_name][d].append(ppl)


# =======================
# PLOTTING
# =======================

# -------- ADD (GLOVE + WORD2VEC) --------
plt.figure()

names, labels = [], []
for emb in ["glove", "word2vec"]:
    for c in ADD_CONST_LIST:
        names.append(f"{emb}_add_{c}")
        labels.append(f"{emb}, c={c}")

props_dict = construct_properties_dict(names, labels)

for emb in ["glove", "word2vec"]:
    for c in ADD_CONST_LIST:
        key = f"{emb}_add_{c}"
        props = props_dict[key]

        plt.plot(
            m_values,
            results["add"][emb][c],
            color=props["color"],
            linestyle=props["linestyle"],
            marker=props["marker"],
            label=props["label"]
        )

plt.xlabel("Number of Synonyms (m)")
plt.ylabel("Perplexity")
plt.grid(True)
plt.legend(ncol=2)

plt.tight_layout()
plt.savefig("combined_add_constants.pgf")
plt.savefig("combined_add_constants.pdf")
plt.close()


# -------- KN (GLOVE + WORD2VEC) --------
plt.figure()

names, labels = [], []
for emb in ["glove", "word2vec"]:
    for d in KN_DISCOUNT_LIST:
        names.append(f"{emb}_kn_{d}")
        labels.append(f"{emb}, d={d}")

props_dict = construct_properties_dict(names, labels)

for emb in ["glove", "word2vec"]:
    for d in KN_DISCOUNT_LIST:
        key = f"{emb}_kn_{d}"
        props = props_dict[key]

        plt.plot(
            m_values,
            results["kn"][emb][d],
            color=props["color"],
            linestyle=props["linestyle"],
            marker=props["marker"],
            label=props["label"]
        )

plt.xlabel("Number of Synonyms (m)")
plt.ylabel("Perplexity")
plt.grid(True)
plt.legend(ncol=2)

plt.tight_layout()
plt.savefig("combined_kn_discounts.pgf")
plt.savefig("combined_kn_discounts.pdf")
plt.close()


# =======================
# TABLES
# =======================
print("\n===== ADD-CONSTANT RESULTS =====")

for emb in emb_list:
    print(f"\n--- {emb.upper()} ---")

    header = "m | " + " | ".join([f"Add={c}" for c in ADD_CONST_LIST])
    print(header)
    print("-" * len(header))

    for i, m in enumerate(m_values):
        row = f"{m:2d} | "
        row += " | ".join([
            f"{results['add'][emb][c][i]:8.2f}" for c in ADD_CONST_LIST
        ])
        print(row)


print("\n===== KNESER-NEY RESULTS =====")

for emb in emb_list:
    print(f"\n--- {emb.upper()} ---")

    header = "m | " + " | ".join([f"Discount={d}" for d in KN_DISCOUNT_LIST])
    print(header)
    print("-" * len(header))

    for i, m in enumerate(m_values):
        row = f"{m:2d} | "
        row += " | ".join([
            f"{results['kn'][emb][d][i]:8.2f}" for d in KN_DISCOUNT_LIST
        ])
        print(row)


print("\nAll experiments complete.")
