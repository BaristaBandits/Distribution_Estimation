# =======================
# IMPORTS
# =======================
import matplotlib
matplotlib.use("pgf")

import matplotlib.pyplot as plt
import json
import os


# =======================
# STYLE (IEEE ONLY SOURCE)
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
style_path = os.path.join(BASE_DIR, "IEEEstyle.mplstyle")

plt.style.use(style_path)


# =======================
# LOAD RESULTS
# =======================
json_path = os.path.join(BASE_DIR, "results.json")

with open(json_path, "r") as f:
    results = json.load(f)


# =======================
# SETTINGS (MUST MATCH ORIGINAL)
# =======================
m_values = [0, 5, 10, 20, 30, 40, 50]

# keys will be strings now because JSON
ADD_CONST_LIST = list(map(float, results["add"]["glove"].keys()))
KN_DISCOUNT_LIST = list(map(float, results["kn"]["glove"].keys()))


# =======================
# LINE STYLE FUNCTION
# =======================
def construct_properties_dict(names, labels):
    colors = ['blue', 'brown', 'purple', 'darkorange', 'green', 'red']
    markers = ['o', '^', 's', 'P', 'X', '*']

    properties_dict = {}

    for i, name in enumerate(names):
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
# PLOTTING: ADD-CONSTANT
# =======================
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
            results["add"][emb][str(c)],
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


# =======================
# PLOTTING: KNESER-NEY
# =======================
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
            results["kn"][emb][str(d)],
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


print("Plots generated successfully.")
