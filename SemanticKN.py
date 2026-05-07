from collections import Counter, defaultdict
import math

# OPTIONAL CYTHON IMPORT
# If compiled extension exists, it will be used automatically
try:
    from semantic_norm import compute_normalizer_cython
    CYTHON_AVAILABLE = True
except:
    CYTHON_AVAILABLE = False


class SemanticBigramKneserNey:
    def __init__(self, discount=0.75, topk_cache=None, d_estimate=None):
        self.discount = discount

        self.unigram_counts = Counter()
        self.bigram_counts = Counter()

        self.followers_of_history = defaultdict(set)
        self.predecessors_of_word = defaultdict(set)

        self.vocab = set()
        self.total_bigram_types = 0

        # semantic cache: {word: [(synonym, weight), ...]}
        self.topk_cache = topk_cache or {}
        self.d_estimate = d_estimate or {}

        # =======================
        # NEW CACHES
        # =======================
        self.semantic_norm_cache = {}
        self.semantic_lambda_cache = {}
        self.semantic_cont_cache = {}

        self.global_semantic_cont_sum = None

    # =======================
    # TRAINING
    # =======================
    def fit(self, tokenized_sentences, k_syn=5, beta=1):
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
        # PRECOMPUTE CONTINUATIONS
        # =======================
        for w in self.vocab:
            self.semantic_cont_cache[w] = self.semantic_continuation( w, k_syn, beta)

        # =======================
        # GLOBAL CONT SUM
        # =======================
        self.global_semantic_cont_sum = sum(self.semantic_cont_cache.values())

        # =======================
        # PRECOMPUTE LAMBDAS
        # =======================
        for h in self.vocab:
            self.semantic_lambda_cache[h] = self.semantic_lambda(h, k_syn, beta)

        # =======================
        # PRECOMPUTE NORMALIZERS
        # =======================
        for h in self.vocab:

            # Use Cython if available
            if CYTHON_AVAILABLE:
                z = compute_normalizer_cython(self, h,k_syn,beta)
            else:
                z = self._semantic_normalizer_python(h, k_syn, beta)

            self.semantic_norm_cache[h] = z

    def base_weight(self, word):
        n = self.unigram_counts.get(word, 0)

        if word in self.d_estimate:
            d = self.d_estimate[word]
            return 1 / (d / (2 * n))
        else:
            return 1

    # =======================
    # BASE COMPONENTS
    # =======================
    def continuation_prob(self, word):
        if self.total_bigram_types == 0:
            return 1.0 / max(len(self.vocab), 1)

        return (
            len(self.predecessors_of_word[word])
            / self.total_bigram_types
        )

    def first_term(self, history, word):
        c_h = self.unigram_counts.get(history, 0)
        c_hw = self.bigram_counts.get((history, word), 0)

        if c_h == 0:
            return 0.0

        return max(c_hw - self.discount, 0.0) / c_h

    def lambda_term(self, history):
        c_h = self.unigram_counts.get(history, 0)

        if c_h == 0:
            return 1.0

        return (
            self.discount
            * len(self.followers_of_history[history])
        ) / c_h

    def prob(self, history, word):
        first = self.first_term(history, word)

        lam = self.lambda_term(history)

        p_cont = self.continuation_prob(word)

        if p_cont == 0:
            p_cont = 1.0 / max(len(self.vocab), 1)

        return first + lam * p_cont

    # =======================
    # SEMANTIC COMPONENTS
    # =======================
    def get_synonyms(self, word, k):
        syns = self.topk_cache.get(word, [])
        return syns[:k] if syns else []

    def semantic_first_term(self, history, word, k_syn, beta):
        synonyms = self.get_synonyms(history, k_syn)

        base = self.first_term(history, word)

        baseweight = self.base_weight(history)

        total = baseweight * base
        total_weight = baseweight

        for s, weight in synonyms:
            val = self.first_term(s, word)

            total += weight * val
            total_weight += weight

        return (
            total / total_weight
            if total_weight > 0 else 0.0
        )

    def semantic_lambda(self, history, k_syn, beta):
        synonyms = self.get_synonyms(history, k_syn)

        base = self.lambda_term(history)

        baseweight = self.base_weight(history)

        total = base * baseweight
        total_weight = baseweight

        for s, weight in synonyms:
            lam = self.lambda_term(s)

            total += weight * lam
            total_weight += weight

        return (
            total / total_weight
            if total_weight > 0 else 0.0
        )

    def semantic_continuation(self, word, k_syn, beta):
        synonyms = self.get_synonyms(word, k_syn)

        base = self.continuation_prob(word)

        baseweight = self.base_weight(word)

        total = base * baseweight
        total_weight = baseweight

        for u, weight in synonyms:
            p = self.continuation_prob(u)

            total += weight * p
            total_weight += weight

        return (
            total / total_weight
            if total_weight > 0
            else 1.0 / max(len(self.vocab), 1)
        )

    # =======================
    # PYTHON NORMALIZER
    # =======================
    def _semantic_normalizer_python(self, history, k_syn=5, beta=1):
        total = 0.0

        # Cached lambda
        lam = self.semantic_lambda_cache[history]

        # IMPORTANT:
        # use cached continuation
        for w in self.vocab:

            first = self.semantic_first_term(
                history,
                w,
                k_syn,
                beta
            )

            p_cont = self.semantic_cont_cache[w]

            p = first + lam * p_cont

            if p > 0:
                total += p

        return total if total > 0 else 1.0

    # =======================
    # FINAL PROBABILITY
    # =======================
    def semantic_prob(self, history, word, k_syn=5, beta=1):
        first = self.semantic_first_term(
            history,
            word,
            k_syn,
            beta
        )

        # Cached lambda
        lam = self.semantic_lambda_cache.get(
            history,
            self.semantic_lambda(history, k_syn, beta)
        )

        # Cached continuation
        p_cont = self.semantic_cont_cache.get(
            word,
            self.semantic_continuation(word, k_syn, beta)
        )

        prob = first + lam * p_cont

        if prob <= 0:
            return 1.0 / max(len(self.vocab), 1)

        # =======================
        # TRUE NORMALIZATION
        # =======================
        Z = self.semantic_norm_cache.get(history, 1.0)

        return prob / Z

    # =======================
    # LOG PROB
    # =======================
    def sentence_log_prob(self, sent, k_syn=5, beta=1):
        log_prob = 0.0

        for i in range(1, len(sent)):

            if k_syn == 0:
                p = self.prob(
                    sent[i - 1],
                    sent[i]
                )

            else:
                p = self.semantic_prob(
                    sent[i - 1],
                    sent[i],
                    k_syn,
                    beta
                )

            log_prob += math.log2(p)

        return log_prob

    # =======================
    # PERPLEXITY
    # =======================
    def perplexity(self, tokenized_sentences, k_syn=5, beta=1):
        total_log_prob = 0.0
        total_tokens = 0

        for sent in tokenized_sentences:

            if len(sent) < 2:
                continue

            total_log_prob += self.sentence_log_prob(
                sent,
                k_syn,
                beta
            )

            total_tokens += (len(sent) - 1)

        return 2 ** ( -total_log_prob / total_tokens)
