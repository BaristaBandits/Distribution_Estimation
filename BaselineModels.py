from collections import defaultdict
from collections import Counter, defaultdict
import math

class JelinekMercerSmoothing:
  def __init__(self, data):
    self.unigrams= defaultdict(int)
    self.bigrams=defaultdict(int)
    self.total_unigrams=0
    self.compute_grams(data)
    self.lambda_=0.5
    
  # Set_lambda invoked customly so that class computes unigrams and bigrams only once on the test corpus
  def set_lambda(self, lambda_):
    self.lambda_=lambda_
    
  def compute_grams(self, data):
    for sentence in data:
      for i in range(len(sentence)):
        self.unigrams[sentence[i]]+=1
        self.total_unigrams+=1
        if i>0:
          self.bigrams[(sentence[i-1], sentence[i])]+=1

  # Jelenik mercer smoothing
  def compute_probs(self, word1, word2):
    count_word1 = self.unigrams.get(word1, 0)

    unigram_prob = self.unigrams.get(word2, 0)/ self.total_unigrams
    bigram_prob = self.bigrams.get((word1, word2), 0)/ count_word1 if count_word1>0 else 0
    return max(self.lambda_ * bigram_prob + (1 - self.lambda_) * unigram_prob, 1e-12)

  def perplexity(self, tokenized_sentences, k_syn=0):
    total_tokens = 0
    log_prob_sum = 0.0

    for sent in tokenized_sentences:
        if len(sent) < 2:
            continue

        for i in range(1, len(sent)):

            prev_word = sent[i - 1]
            word = sent[i]

            try:
                if k_syn == 0:
                    prob = self.compute_probs(prev_word, word)
                else:
                    prob = self.semantic_prob(prev_word, word, k_syn)

                if prob <= 0:
                    prob = 1e-12

                log_prob_sum += math.log2(prob)
                total_tokens += 1

            except KeyError:
                continue

    if total_tokens == 0:
        return float("inf")

    avg_log_prob = log_prob_sum / total_tokens
    return 2 ** (-avg_log_prob)

  # Next word predictor
  def predict_next(self, current):
    best_word=None
    best_probab=0

    for word2 in self.unigrams:
      prob=self.compute_probs(current, word2)
      if prob>best_probab:
        best_probab=prob
        best_word = word2
    return best_word, best_probab


from collections import Counter, defaultdict
import math


class BigramKneserNeyNaive:
    def __init__(self, discount=0.75, topk_cache=None, d_estimate=None):
        self.discount = discount

        self.unigram_counts = Counter()
        self.bigram_counts = Counter()

        self.followers_of_history = defaultdict(set)
        self.predecessors_of_word = defaultdict(set)

        self.vocab = set()
        self.total_bigram_types = 0

        # semantic components
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

    # =======================
    # CONTINUATION PROB
    # =======================
    def continuation_prob(self, word):
        if self.total_bigram_types == 0:
            return 1.0 / max(len(self.vocab), 1)

        val = len(self.predecessors_of_word[word]) / self.total_bigram_types

        if val == 0:
            return 1.0 / max(len(self.vocab), 1)

        return val

    # =======================
    # BASE KN COMPONENTS
    # =======================
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

    # =======================
    # BASE PROB (matches semantic k=0)
    # =======================
    def prob(self, history, word):
        first = self.first_term(history, word)
        lam = self.lambda_term(history)
        p_cont = self.continuation_prob(word)

        return first + lam * p_cont

    # =======================
    # SEMANTIC PROBABILITY
    # =======================
    def semantic_prob(self, prev_word, word, k_syn=5, beta = 1):

        synonyms = self.topk_cache.get(prev_word, [])[:k_syn]
        if not synonyms:
            first = self.first_term(prev_word, word)
            lam = self.lambda_term(prev_word)
            p_cont = self.continuation_prob(word)
            return first + lam * p_cont

        base_prob = self.prob(prev_word, word)

        n = self.unigram_counts.get(prev_word, 0)

        if base_prob > 0 and prev_word in self.d_estimate and n > 0:
            d_est = self.d_estimate[prev_word]
            base_weight = 1 / (d_est / (2 * n))
        else:
            base_weight = 0.0

        weighted_prob = math.exp(- beta * base_weight) * base_prob
        total_weight = math.exp(- beta * base_weight)

        for s, weight in synonyms:
            if s != prev_word:
                prob = self.prob(s, word)
                weighted_prob += math.exp(- beta * weight) * prob
                total_weight += math.exp(- beta * weight)

        if total_weight == 0:
            return base_prob

        return weighted_prob / total_weight

    # =======================
    # NORMAL LOG PROB
    # =======================
    def sentence_log_prob(self, sent):
        log_prob = 0.0

        for i in range(1, len(sent)):
            p = self.prob(sent[i - 1], sent[i])
            log_prob += math.log2(p)

        return log_prob

    # =======================
    # SEMANTIC LOG PROB
    # =======================
    def sentence_log_prob_semantic(self, sent, k_syn=5):
        log_prob = 0.0

        for i in range(1, len(sent)):
            p = self.semantic_prob(sent[i - 1], sent[i], k_syn)
            log_prob += math.log2(p)

        return log_prob

    # =======================
    # PERPLEXITY
    # =======================
    def perplexity(self, tokenized_sentences):
        total_log_prob = 0.0
        total_tokens = 0

        for sent in tokenized_sentences:
            if len(sent) < 2:
                continue

            total_log_prob += self.sentence_log_prob(sent)
            total_tokens += (len(sent) - 1)

        return 2 ** (-total_log_prob / total_tokens)

    # =======================
    # SEMANTIC PERPLEXITY
    # =======================
    def semantic_perplexity(self, tokenized_sentences, k_syn=5):
        total_log_prob = 0.0
        total_tokens = 0

        for sent in tokenized_sentences:
            if len(sent) < 2:
                continue

            total_log_prob += self.sentence_log_prob_semantic(sent, k_syn)
            total_tokens += (len(sent) - 1)

        return 2 ** (-total_log_prob / total_tokens)



