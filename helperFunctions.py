import math
from tqdm import tqdm

def compute_perplexity(model, test_corpus, k_syn=0):

    total_tokens = 0
    log_prob_sum = 0

    for sentence in tqdm(test_corpus):
        tokens = ["<s>"] + sentence.split() + ["</s>"]
        if len(tokens) < 2:
            continue
          
        for i in range(1, len(tokens)):
            prev_word = tokens[i-1]
            word = tokens[i]
            # baseline
            if k_syn == 0:
                prob = model.compute_probs(prev_word, word)
            # semantic interpolation
            else:
                prob = semantic_prob(prev_word, word, model, k_syn)
            if prob <= 0:
                prob = 1e-12
            log_prob_sum += math.log2(prob)

        total_tokens += len(tokens) - 1

    avg_log_prob = log_prob_sum / total_tokens

    return 2 ** (-avg_log_prob)
