"""
run_experiments.py
------------------
Runs SemanticBigramKneserNey across all combinations of:
    emb_list × discounts × m_values
and saves results to results_semantic_kn.json

Usage:
    python run_experiments.py
    python run_experiments.py --max_sentences 50000 --discounts 0.6 0.7 0.8 --k_cache 50
"""

import os
import json
import pickle
import argparse
import tqdm

from load_embed import load_embeddings
from load_dataset import load_text_corpus
from cache_synonyms import build_cache, Support
from SemanticKN import SemanticBigramKneserNey  # the class above

# =======================
# ARGUMENTS
# =======================
parser = argparse.ArgumentParser()
parser.add_argument("--max_sentences", type=int,   default=100000)
parser.add_argument("--beta",          type=float, default=1.0)
parser.add_argument("--k_cache",       type=int,   default=50)
parser.add_argument(
    "--discounts", nargs="+", type=float,
    default=[0.6, 0.7, 0.8]
)
args = parser.parse_args()

# =======================
# SETTINGS
# =======================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

m_values = [0, 5, 10, 20, 30, 40, 50]
emb_list  = ["glove", "word2vec", "gpt2"]

# =======================
# RESULTS STRUCTURE
# results[emb][discount][m_index] = perplexity
# =======================
results = {
    emb: {str(d): [] for d in args.discounts}
    for emb in emb_list
}

# =======================
# CACHE HELPERS
# =======================
def cache_path(emb_name):
    return os.path.join(CACHE_DIR, f"{emb_name}_cache.pkl")


def load_or_build_cache(emb_name, train_corpus, embeddings):
    path = cache_path(emb_name)
    if os.path.exists(path):
        print(f"  Loading synonym cache for {emb_name}...")
        with open(path, "rb") as f:
            d_estimate, synonym_cache = pickle.load(f)
    else:
        print(f"  Building synonym cache for {emb_name}...")
        support = Support(pmin=1e-5)
        d_estimate, synonym_cache = build_cache(
            train_corpus, support, embeddings, args.k_cache
        )
        with open(path, "wb") as f:
            pickle.dump((d_estimate, synonym_cache), f)
        print(f"  Saved cache → {path}")
    return d_estimate, synonym_cache


# =======================
# MAIN LOOP
# =======================
for emb_name in emb_list:
    print(f"\n{'='*50}")
    print(f"  Embedding: {emb_name}")
    print(f"{'='*50}")

    embeddings, stoi, itos = load_embeddings(emb_name)
    tokenizer_mode = "gpt2" if emb_name == "gpt2" else "word"

    train_corpus, test_corpus = load_text_corpus(
        "wikitext103",
        tokenizer_mode=tokenizer_mode,
        max_sentences=args.max_sentences,
    )

    d_estimate, synonym_cache = load_or_build_cache(
        emb_name, train_corpus, embeddings
    )

    for discount in args.discounts:
        print(f"\n  Discount = {discount}")

        # ── Build and fit model ─────────────────────────────────────────────
        model = SemanticBigramKneserNey(
            discount=discount,
            topk_cache=synonym_cache,
            d_estimate=d_estimate,
        )
        model.fit(train_corpus)

        # ── Evaluate across m_values ────────────────────────────────────────
        for m in tqdm.tqdm(m_values, desc=f"    m_values (discount={discount})"):

            if m > 0:
                # KEY SPEEDUP:
                # Pre-compute Z(h) for every unique history in test set ONCE.
                # All subsequent calls to perplexity() for this (model, m) pair
                # hit the cache — no redundant _prob_vec() calls.
                model.clear_z_cache()          # clear previous m's cache
                model.warm_up_z_cache(test_corpus, k_syn=m)

            ppl = model.perplexity(test_corpus, k_syn=m, beta=args.beta)
            results[emb_name][str(discount)].append(ppl)

            print(f"    m={m:2d} → PPL = {ppl:.4f}")


# =======================
# PRINT TABLES
# =======================
print("\n\n===== SEMANTIC KN RESULTS =====")

for emb in emb_list:
    print(f"\n--- {emb.upper()} ---")
    header = "m  | " + " | ".join([f"d={d}" for d in args.discounts])
    print(header)
    print("-" * len(header))
    for i, m in enumerate(m_values):
        row = f"{m:2d} | "
        row += " | ".join([
            f"{results[emb][str(d)][i]:8.4f}"
            for d in args.discounts
        ])
        print(row)


# =======================
# SAVE JSON
# =======================
output_path = os.path.join(BASE_DIR, "results_semantic_kn.json")
with open(output_path, "w") as f:
    json.dump(
        {
            "settings": {
                "max_sentences": args.max_sentences,
                "beta":          args.beta,
                "k_cache":       args.k_cache,
                "discounts":     args.discounts,
                "m_values":      m_values,
                "embeddings":    emb_list,
            },
            "results": results,
        },
        f,
        indent=4,
    )

print(f"\nResults saved → {output_path}")
print("Done.")
