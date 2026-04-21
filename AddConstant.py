from collections import defaultdict, Counter
import math

class AddConstantBigram:

    def __init__(self, add_constant=0.75, topk_cache=None, d_estimate=None):
        self.add_constant = add_constant

        self.unigrams = Counter()
        self.bigrams = Counter()
        self.total_unigrams = 0

        self.vocab = set()
        self.vocab_size = 0

        self.continuations = defaultdict(set)

        # semantic components
        self.topk_cache = topk_cache or {}
        self.d_estimate = d_estimate or {}


    def fit(self, tokenized_sentences):
        for sent in tokenized_sentences:
            self.vocab.update(sent)

            for i in range(len(sent)):
                w = sent[i]
                self.unigrams[w] += 1
                self.total_unigrams += 1

                if i > 0:
                    h = sent[i - 1]
                    self.bigrams[(h, w)] += 1
                    self.continuations[h].add(w)

        self.vocab_size = len(self.vocab)

    def prob(self, prev_word, word):

        unigram_count = self.unigrams.get(prev_word, 0)
        bigram_count = self.bigrams.get((prev_word, word), 0)

        numerator = bigram_count + self.add_constant
        denominator = unigram_count + self.add_constant * self.vocab_size

        return max(numerator / denominator, 1e-12)

    def semantic_prob(self, prev_word, word, k_syn=5):

      base_prob = self.prob(prev_word, word)

      # EXACT match to old behavior
      synonyms = self.topk_cache.get(prev_word, [])[:k_syn]

      if not synonyms:
          return base_prob

      n = self.unigrams.get(prev_word, 0)

      if base_prob > 0:
          d_est = self.d_estimate[prev_word]   # no safety check (same as old)
          base_weight = 1 / (d_est / (2 * n))
      else:
          base_weight = 0

      weighted_prob = base_weight * base_prob
      total_weight = base_weight

      for s, weight in synonyms:
          if s != prev_word:
              prob = self.prob(s, word)
              weighted_prob += weight * prob
              total_weight += weight

      return weighted_prob / total_weight



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
                      prob = self.prob(prev_word, word)
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

    def compute_d_over_n(self):

        d_over_n = {}

        for word in self.unigrams:
            n = self.unigrams[word]
            d = len(self.continuations[word])

            if n > 0:
                d_over_n[word] = d / n
            else:
                d_over_n[word] = 0.0

        return d_over_n

