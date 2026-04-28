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
# STYLE (IEEE)
# =======================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
style_path = os.path.join(BASE_DIR, "IEEEstyle.mplstyle")

plt.style.use(style_path)



def construct_properties_dict(names, labels):
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted', 'solid', 'solid']
    colors = ['blue', 'brown', 'purple', 'darkorange', 'green', 'red']
    markers = ['o', '^', 's', 'P', 'X', '*']

    properties_dict = {}

    for i, name in enumerate(names):
        properties_dict[name] = {
            'color': colors[i],
            'linestyle': linestyles[i],
            'marker': markers[i],
            'label': labels[i]
        }

    return properties_dict


# =======================
# ARGUMENTS
# =======================
parser = argparse.ArgumentParser()
parser.add_argument("--max_sentences", type=int, default=None)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--k_cache", type=int, default=50)

args = parser.parse_args()


# =======================
# SETTINGS
# =======================
k_values = [0, 5, 10, 20, 30, 40, 50]
emb_list = ["glove", "word2vec", "gpt2"]

ADD_CONST = 0.0005
KN_DISCOUNT = 0.85


# =======================
# STORE RESULTS
# =======================
results = {
    "add": {emb: [] for emb in emb_list},
    "kn": {emb: [] for emb in emb_list}
}


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

    support = Support(pmin=1e-5)

    print("Building synonym cache...")
    d_estimate, synonym_cache = build_cache(
        train_corpus,
        support,
        embeddings,
        args.k_cache
    )

    # =======================
    # ADD-CONSTANT MODEL
    # =======================
    add_model = AddConstantBigram(topk_cache=synonym_cache)
    add_model.fit(train_corpus)
    add_model.add_constant = ADD_CONST

    print("Evaluating Add-Constant...")
    for k in tqdm.tqdm(k_values):
        ppl = add_model.perplexity(test_corpus, k_syn=k, beta=args.beta)
        results["add"][emb_name].append(ppl)

    # =======================
    # KN MODEL
    # =======================
    kn_model = SemanticBigramKneserNey(
        discount=KN_DISCOUNT,
        topk_cache=synonym_cache,
        d_estimate=d_estimate
    )

    kn_model.fit(train_corpus)

    print("Evaluating Kneser-Ney...")
    for k in tqdm.tqdm(k_values):
        ppl = kn_model.perplexity(test_corpus, k_syn=k, beta=args.beta)
        results["kn"][emb_name].append(ppl)


# =======================
# PLOT: TWO SUBPLOTS
# =======================
names = [
    "w2v_add", "gpt2_add",
    "w2v_kn", "gpt2_kn"
]

labels = [
    "Word2Vec", "GPT-2",
    "Word2Vec", "GPT-2"
]

prop = construct_properties_dict(names, labels)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# =======================
# LEFT: ADD-CONSTANT
# =======================
ax = axes[0]

add_map = {
    "w2v_add": results["add"]["word2vec"],
    "gpt2_add": results["add"]["gpt2"]
}

for key in ["w2v_add", "gpt2_add"]:
    ax.plot(
        k_values,
        add_map[key],
        color=prop[key]['color'],
        linestyle=prop[key]['linestyle'],
        marker=prop[key]['marker'],
        label=prop[key]['label']
    )

ax.set_title("Add-Constant (c = 0.0005)")
ax.set_xlabel("Number of Synonyms (k)")
ax.set_ylabel("Perplexity")
ax.grid(True)
ax.legend()


# =======================
# RIGHT: KN
# =======================
ax = axes[1]

kn_map = {
    "w2v_kn": results["kn"]["word2vec"],
    "gpt2_kn": results["kn"]["gpt2"]
}

for key in ["w2v_kn", "gpt2_kn"]:
    ax.plot(
        k_values,
        kn_map[key],
        color=prop[key]['color'],
        linestyle=prop[key]['linestyle'],
        marker=prop[key]['marker'],
        label=prop[key]['label']
    )

ax.set_title("Kneser-Ney (d = 0.85)")
ax.set_xlabel("Number of Synonyms (k)")
ax.grid(True)
ax.legend()


# =======================
# FINAL TOUCHES
# =======================
plt.suptitle("Perplexity vs k (WikiText103)", fontsize=14)
plt.tight_layout()
plt.savefig('Plots.png')
plt.show()



# =======================
# TABLE (ALL 6 CURVES)
# =======================
print("\n===== RESULTS TABLE =====")

header = "k | GloVe Add | GloVe KN | W2V Add | W2V KN | GPT2 Add | GPT2 KN"
print(header)
print("-" * len(header))

for i, k in enumerate(k_values):
    print(
        f"{k:2d} | "
        f"{results['add']['glove'][i]:8.2f} | "
        f"{results['kn']['glove'][i]:8.2f} | "
        f"{results['add']['word2vec'][i]:8.2f} | "
        f"{results['kn']['word2vec'][i]:8.2f} | "
        f"{results['add']['gpt2'][i]:8.2f} | "
        f"{results['kn']['gpt2'][i]:8.2f}"
    )
