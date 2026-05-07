from collections import Counter, defaultdict
import math
import numpy as np


class SemanticBigramKneserNey:
    def __init__(self, discount=0.75, topk_cache=None, d_estimate=None):
        self.discount = discount

        self.unigram_counts = Counter()
        self.bigram_counts = Counter()

        self.followers_of_history = defaultdict(set)
        self.predecessors_of_word = defaultdict(set)

        self.vocab = set()
        self.total_bigram_types = 0

        self.topk_cache = topk_cache or {}
        self.d_estimate = d_estimate or {}

        self.semantic_norm_cache = {}
        self.semantic_lambda_cache = {}
        self.semantic_cont_cache = {}

    # =======================
    # TRAIN
    # =======================
    def fit_counts(self, tokenized_sentences):

        for sent in tokenized_sentences:
            self.vocab.update(sent)

            for w in sent:
                self.unigram_counts[w] += 1

            for i in range(1, len(sent)):
                h = sent[i - 1]
                w = sent[i]

                self.bigram_counts[(h, w)] += 1
                self.followers_of_history[h].add(w)
                self.predecessors_of_word[w].add(h)

        self.total_bigram_types = len(self.bigram_counts)

    # =======================
    # BASE FUNCTIONS (UNCHANGED)
    # =======================
    def base_weight(self, word):
        n = self.unigram_counts.get(word, 0)
        if word in self.d_estimate and n > 0:
            d = self.d_estimate[word]
            return 2 * n / d
        return 1.0

    def continuation_prob(self, word):
        if self.total_bigram_types == 0:
            return 1.0 / max(len(self.vocab), 1)
        return len(self.predecessors_of_word[word]) / self.total_bigram_types

    def first_term(self, history, word):
        c_h = self.unigram_counts.get(history, 0)
        if c_h == 0:
            return 0.0

        c_hw = self.bigram_counts.get((history, word), 0)
        return max(c_hw - self.discount, 0.0) / c_h

    def lambda_term(self, history):
        c_h = self.unigram_counts.get(history, 0)
        if c_h == 0:
            return 1.0
        return (self.discount * len(self.followers_of_history[history])) / c_h

    # =======================
    # SYNONYMS
    # =======================
    def get_synonyms(self, word, k):
        return self.topk_cache.get(word, [])[:k]

    # =======================
    # FAST CANDIDATE SET BUILDER
    # =======================
    def build_candidate_set(self, history, k_syn):

        cand = set(self.followers_of_history.get(history, set()))

        for s, _ in self.get_synonyms(history, k_syn):
            cand.update(self.followers_of_history.get(s, set()))

        return cand

    # =======================
    # FAST SEMANTIC FIRST TERM (ONLY OVER CANDIDATES)
    # =======================
    def semantic_first_term_fast(self, history, word, k_syn):

        base = self.first_term(history, word)
        baseweight = self.base_weight(history)

        synonyms = self.get_synonyms(history, k_syn)

        total = baseweight * base
        total_w = baseweight

        for s, wgt in synonyms:
            val = self.first_term(s, word)
            total += wgt * val
            total_w += wgt

        return total / total_w if total_w > 0 else 0.0

    # =======================
    # NORMALIZER (FAST CANDIDATE VERSION)
    # =======================
    def semantic_normalizer(self, history, k_syn):

        lam = self.lambda_term(history)

        cand = self.build_candidate_set(history, k_syn)

        if len(cand) == 0:
            return 1.0

        total = 0.0

        for w in cand:
            total += self.semantic_first_term_fast(history, w, k_syn)

        # continuation term independent of history
        cont_sum = 0.0
        for w in cand:
            cont_sum += self.continuation_prob(w)

        return total + lam * cont_sum if total > 0 else 1.0

    # =======================
    # FINAL PROBABILITY
    # =======================
    def semantic_prob(self, history, word, k_syn=5):

        first = self.semantic_first_term_fast(history, word, k_syn)
        lam = self.lambda_term(history)

        p_cont = self.continuation_prob(word)

        prob = first + lam * p_cont

        if prob <= 0:
            return 1.0 / max(len(self.vocab), 1)

        Z = self.semantic_normalizer(history, k_syn)

        return prob / Z

    # =======================
    # LOG PROB
    # =======================
    def sentence_log_prob(self, sent, k_syn=5):
        log_prob = 0.0

        for i in range(1, len(sent)):
            p = self.semantic_prob(sent[i - 1], sent[i], k_syn)
            log_prob += math.log2(p)

        return log_prob

    # =======================
    # PERPLEXITY
    # =======================
    def perplexity(self, corpus, k_syn=5):
        total_log = 0.0
        total_tokens = 0

        for sent in corpus:
            if len(sent) < 2:
                continue

            total_log += self.sentence_log_prob(sent, k_syn)
            total_tokens += len(sent) - 1

        return 2 ** (-total_log / total_tokens)
