from collections import Counter, defaultdict
import math

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
    # TRAINING
    # =======================
    def fit(self, tokenized_sentences):
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
    
    def base_weight(self, word):
      n = self.unigram_counts.get(word, 0)
      if word in self.d_estimate:
        d = self.d_estimate[word]
        return (2.0 * n) / d
      else:
        return 1

    # =======================
    # BASE COMPONENTS
    # =======================
    def continuation_prob(self, word):
        if self.total_bigram_types == 0:
            return 1.0 / max(len(self.vocab), 1)
        return len(self.predecessors_of_word[word]) / self.total_bigram_types

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

        return (self.discount * len(self.followers_of_history[history])) / c_h

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

    def semantic_first_term(self, history, word, k_syn, beta ):
        synonyms = self.get_synonyms(history, k_syn)
        # include base history itself
        base = self.first_term(history, word)
        baseweight = self.base_weight(history)
        total = math.exp(- beta * baseweight) * base
        total_weight =  math.exp(- beta * baseweight)

        for s, weight in synonyms:
            val = self.first_term(s, word)
            total += math.exp ( -beta * weight) * val
            total_weight += math.exp( - beta * weight)

        return total / total_weight if total_weight > 0 else 0.0

    def semantic_lambda(self, history, k_syn, beta ):
        synonyms = self.get_synonyms(history, k_syn)

        # include base history itself
        base = self.lambda_term(history)
        baseweight = self.base_weight(history)
        total = base * math.exp ( - beta * baseweight)
        total_weight = math.exp ( - beta * baseweight)

        for s, weight in synonyms:
            lam = self.lambda_term(s)
            total += math.exp ( - beta * weight) * lam
            total_weight += math.exp ( - beta * weight)

        return total / total_weight if total_weight > 0 else 0.0

    def semantic_continuation(self, word, k_syn, beta ):
        synonyms = self.get_synonyms(word, k_syn)
        
        # include base history itself
        base = self.continuation_prob(word)
        baseweight = self.base_weight(word)
        total = base * math.exp ( - beta * baseweight)
        total_weight = math.exp ( - beta * baseweight)

        for u, weight in synonyms:
            p = self.continuation_prob(u)
            total += math.exp ( - beta * weight) * p
            total_weight += math.exp ( - beta * weight)

        return total / total_weight if total_weight > 0 else 1.0 / max(len(self.vocab), 1)

    # =======================
    # FINAL PROBABILITY
    # =======================
    def semantic_prob(self, history, word, k_syn=5, beta = 1):
        first = self.semantic_first_term(history, word, k_syn, beta)
        lam = self.semantic_lambda(history, k_syn, beta)
        p_cont = self.semantic_continuation(word, k_syn, beta)

        prob = first + lam * p_cont
        if prob <= 0:
          return 1.0 / max(len(self.vocab), 1)
        return prob

    # =======================
    # LOG PROB
    # =======================
    def sentence_log_prob(self, sent, k_syn=5):
        log_prob = 0.0

        #Empirical Normalization
        GLOBAL_Z = 1.0369
        for i in range(1, len(sent)):
            if k_syn == 0:
                p = self.prob(sent[i - 1], sent[i])   #baseline KN
            else:
                p = self.semantic_prob(sent[i - 1], sent[i], k_syn)
                p /= GLOBAL_Z 
            log_prob += math.log2(p)

        return log_prob

    # =======================
    # PERPLEXITY
    # =======================
    def perplexity(self, tokenized_sentences, k_syn=5):
        total_log_prob = 0.0
        total_tokens = 0

        for sent in tokenized_sentences:
            if len(sent) < 2:
                continue

            total_log_prob += self.sentence_log_prob(sent, k_syn)
            total_tokens += (len(sent) - 1)

        return 2 ** (-total_log_prob / total_tokens)

