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

        # numpy state — built after fit()
        self._vocab_list   = None   # list[str], index → word
        self._word2idx     = {}     # word → index
        self._cont_probs   = None   # shape (V,)
        self._base_weights = None   # shape (V,)

        # Z(history) cache — populated lazily, survives across sentences
        self._z_cache = {}

    # =======================
    # TRAINING
    # =======================
    def fit(self, tokenized_sentences):
        for sent in tokenized_sentences:
            self.vocab.update(sent)
            for w in sent:
                self.unigram_counts[w] += 1
            for i in range(1, len(sent)):
                h, w = sent[i - 1], sent[i]
                self.bigram_counts[(h, w)] += 1
                self.followers_of_history[h].add(w)
                self.predecessors_of_word[w].add(h)

        self.total_bigram_types = len(self.bigram_counts)
        self._build_numpy_tables()

    # =======================
    # NUMPY TABLE — built ONCE after fit()
    # =======================
    def _build_numpy_tables(self):
        V = len(self.vocab)
        self._vocab_list = list(self.vocab)
        self._word2idx   = {w: i for i, w in enumerate(self._vocab_list)}
        floor            = 1.0 / max(V, 1)
        total            = self.total_bigram_types or 1

        # continuation_prob for every vocab word
        self._cont_probs = np.array(
            [len(self.predecessors_of_word[w]) / total for w in self._vocab_list],
            dtype=np.float64,
        )
        self._cont_probs = np.where(self._cont_probs == 0, floor, self._cont_probs)

        # base_weight for every vocab word
        self._base_weights = np.array(
            [self._base_weight_scalar(w) for w in self._vocab_list],
            dtype=np.float64,
        )

    # =======================
    # SCALAR HELPERS (unchanged logic, keep all edge cases)
    # =======================
    def _base_weight_scalar(self, word):
        n = self.unigram_counts.get(word, 0)
        if word in self.d_estimate:
            d = self.d_estimate[word]
            return 1 / (d / (2 * n)) if (d > 0 and n > 0) else 1.0
        return 1.0

    def base_weight(self, word):
        """Public scalar base_weight (kept for compatibility)."""
        return self._base_weight_scalar(word)

    def continuation_prob(self, word):
        if self.total_bigram_types == 0:
            return 1.0 / max(len(self.vocab), 1)
        return len(self.predecessors_of_word[word]) / self.total_bigram_types

    def first_term(self, history, word):
        c_h  = self.unigram_counts.get(history, 0)
        c_hw = self.bigram_counts.get((history, word), 0)
        if c_h == 0:
            return 0.0
        return max(c_hw - self.discount, 0.0) / c_h

    def lambda_term(self, history):
        c_h = self.unigram_counts.get(history, 0)
        if c_h == 0:
            return 1.0
        return (self.discount * len(self.followers_of_history[history])) / c_h

    def prob(self, history, word):
        """Baseline (non-semantic) KN probability."""
        first  = self.first_term(history, word)
        lam    = self.lambda_term(history)
        p_cont = self.continuation_prob(word)
        if p_cont == 0:
            p_cont = 1.0 / max(len(self.vocab), 1)
        return first + lam * p_cont

    def get_synonyms(self, word, k):
        syns = self.topk_cache.get(word, [])
        return syns[:k] if syns else []

    # =======================
    # SCALAR SEMANTIC (kept for single-query use / debugging)
    # =======================
    def semantic_first_term(self, history, word, k_syn, beta=1):
        synonyms     = self.get_synonyms(history, k_syn)
        baseweight   = self._base_weight_scalar(history)
        total        = baseweight * self.first_term(history, word)
        total_weight = baseweight
        for s, weight in synonyms:
            total        += weight * self.first_term(s, word)
            total_weight += weight
        return total / total_weight if total_weight > 0 else 0.0

    def semantic_lambda(self, history, k_syn, beta=1):
        synonyms     = self.get_synonyms(history, k_syn)
        baseweight   = self._base_weight_scalar(history)
        total        = baseweight * self.lambda_term(history)
        total_weight = baseweight
        for s, weight in synonyms:
            total        += weight * self.lambda_term(s)
            total_weight += weight
        return total / total_weight if total_weight > 0 else 0.0

    def semantic_continuation(self, word, k_syn, beta=1):
        synonyms     = self.get_synonyms(word, k_syn)
        baseweight   = self._base_weight_scalar(word)
        total        = baseweight * self.continuation_prob(word)
        total_weight = baseweight
        for u, weight in synonyms:
            total        += weight * self.continuation_prob(u)
            total_weight += weight
        return total / total_weight if total_weight > 0 else 1.0 / max(len(self.vocab), 1)

    def semantic_prob(self, history, word, k_syn=5, beta=1):
        """Scalar semantic probability (unnormalized)."""
        first  = self.semantic_first_term(history, word, k_syn, beta)
        lam    = self.semantic_lambda(history, k_syn, beta)
        p_cont = self.semantic_continuation(word, k_syn, beta)
        prob   = first + lam * p_cont
        if prob <= 0:
            return 1.0 / max(len(self.vocab), 1)
        return prob

    # =======================
    # ★ VECTORIZED CORE ★
    # Computes semantic_prob(history, w) for ALL w ∈ vocab at once.
    # All three components use numpy — zero Python loops over vocab.
    # =======================
    def _prob_vec(self, history, k_syn):
        """
        Returns shape (V,) array: unnormalized semantic_prob(history, w)
        for every w in self._vocab_list.

        Strategy
        --------
        1. first_term:   build (S×V) matrix via vectorized bigram lookup,
                         collapse with weighted dot product → (V,)
        2. lambda:       scalar weighted average over sources
        3. continuation: base cont_probs (V,) already built; synonym
                         contributions added per vocab word (shallow loop)
        """
        V     = len(self._vocab_list)
        floor = 1.0 / max(V, 1)

        # ── sources: [history] + its synonyms ───────────────────────────────
        synonyms   = self.get_synonyms(history, k_syn)          # [(s, w), ...]
        bw_h       = self._base_weight_scalar(history)
        sources    = [history] + [s for s, _ in synonyms]
        src_w      = np.array(
            [bw_h] + [w for _, w in synonyms], dtype=np.float64
        )                                                        # shape (S,)
        total_sw   = src_w.sum()

        # ── (1) first_term matrix: shape (S, V) ─────────────────────────────
        # For each source, we need max(c(src,w)-d, 0) / c(src) for all w.
        # c(src, w) is nonzero only for words w that follow src — use the
        # followers set to build a sparse update rather than looping all V.
        first_mat = np.zeros((len(sources), V), dtype=np.float64)
        for j, src in enumerate(sources):
            c_src = self.unigram_counts.get(src, 0)
            if c_src == 0:
                continue
            # only iterate over actual followers (sparse — much less than V)
            for fw in self.followers_of_history.get(src, set()):
                idx = self._word2idx.get(fw)
                if idx is None:
                    continue
                c_sw = self.bigram_counts.get((src, fw), 0)
                first_mat[j, idx] = max(c_sw - self.discount, 0.0) / c_src

        # weighted average across sources → (V,)
        first_vec = (src_w @ first_mat) / total_sw              # (S,)@(S,V) → (V,)

        # ── (2) lambda: scalar weighted average ─────────────────────────────
        lam_vals = np.array(
            [self.lambda_term(src) for src in sources], dtype=np.float64
        )
        lam_avg  = float(src_w @ lam_vals) / total_sw

        # ── (3) continuation: base + synonym contributions ──────────────────
        # Base is already in self._cont_probs (V,).
        # Each vocab word w_i has its own synonym list → shallow loop over V,
        # but inner work is just a few additions (k_syn per word).
        cont_num = self._cont_probs * self._base_weights         # (V,) element-wise
        cont_den = self._base_weights.copy()                     # (V,)

        for i, w in enumerate(self._vocab_list):
            for u, uw in self.get_synonyms(w, k_syn):
                cont_u       = self.continuation_prob(u)
                cont_num[i] += uw * cont_u
                cont_den[i] += uw

        cont_vec = cont_num / np.where(cont_den > 0, cont_den, 1.0)
        cont_vec = np.where(cont_vec <= 0, floor, cont_vec)

        # ── assemble ─────────────────────────────────────────────────────────
        prob_vec = first_vec + lam_avg * cont_vec               # (V,)
        prob_vec = np.where(prob_vec <= 0, floor, prob_vec)

        return prob_vec                                          # shape (V,)

    # =======================
    # Z(history) — cached globally across the entire test run
    # =======================
    def _get_Z(self, history, k_syn):
        """
        Normalization constant Z(h) = Σ_w semantic_prob(h, w).
        Computed once per (history, k_syn) pair and cached permanently.
        """
        key = (history, k_syn)
        if key not in self._z_cache:
            self._z_cache[key] = self._prob_vec(history, k_syn).sum()
        return self._z_cache[key]

    def clear_z_cache(self):
        """Call between experiments with different k_syn / discount."""
        self._z_cache = {}

    # =======================
    # NORMALIZED SEMANTIC PROB (single word)
    # =======================
    def normalized_semantic_prob(self, history, word, k_syn=5, beta=1):
        """
        Returns the properly normalized p(word | history).
        Uses cached Z so repeated calls for the same history are O(1).
        """
        p = self.semantic_prob(history, word, k_syn, beta)
        Z = self._get_Z(history, k_syn)
        return p / Z if Z > 0 else p

    # =======================
    # LOG PROB
    # =======================
    def sentence_log_prob(self, sent, k_syn=5, beta=1):
        log_prob = 0.0
        for i in range(1, len(sent)):
            h, w = sent[i - 1], sent[i]
            if k_syn == 0:
                p = self.prob(h, w)
            else:
                # Z is fetched from global cache (computed at most once per h)
                p = self.normalized_semantic_prob(h, w, k_syn, beta)
            log_prob += math.log2(max(p, 1e-300))
        return log_prob

    # =======================
    # PERPLEXITY
    # =======================
    def perplexity(self, tokenized_sentences, k_syn=5, beta=1):
        total_log_prob = 0.0
        total_tokens   = 0
        for sent in tokenized_sentences:
            if len(sent) < 2:
                continue
            total_log_prob += self.sentence_log_prob(sent, k_syn, beta)
            total_tokens   += (len(sent) - 1)
        return 2 ** (-total_log_prob / total_tokens)

    # =======================
    # WARM-UP: pre-compute Z for all unique histories in a corpus
    # Call this on the test set BEFORE perplexity() for max speed.
    # =======================
    def warm_up_z_cache(self, tokenized_sentences, k_syn):
        """
        Pre-computes Z(h) for every unique bigram history in the corpus.
        After this, perplexity() makes zero redundant _prob_vec() calls.
        """
        unique_histories = set()
        for sent in tokenized_sentences:
            for i in range(1, len(sent)):
                unique_histories.add(sent[i - 1])

        print(f"  Warming up Z cache: {len(unique_histories)} unique histories...")
        for h in unique_histories:
            self._get_Z(h, k_syn)   # populates cache
        print(f"  Z cache ready ({len(self._z_cache)} entries).")
