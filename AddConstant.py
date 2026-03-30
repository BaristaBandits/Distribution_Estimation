import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from tqdm import tqdm

class AddConstant:

    def __init__(self, data):
        self.unigrams = defaultdict(int)
        self.bigrams = defaultdict(int)
        self.total_unigrams = 0
        self.compute_grams(data)
        self.vocab_size = len(self.unigrams)
        self.add_constant = 0.75
        #self.continuations = defaultdict(set)


    # Set smoothing parameter
    def set_lambda(self, add_constant):
        self.add_constant = add_constant


    # Compute unigram and bigram counts
    def compute_grams(self, data):
        for sentence in data:

            tokens = ["<s>"] + sentence.split() + ["</s>"]

            for i in range(len(tokens)):

                self.unigrams[tokens[i]] += 1
                self.total_unigrams += 1

                if i > 0:
                    self.bigrams[(tokens[i-1], tokens[i])] += 1
                    #self.continuations[tokens[i-1]].add(tokens[i])


    # Add-constant bigram probability
    def compute_probs(self, word1, word2):

        unigram_count = self.unigrams.get(word1, 0)
        bigram_count = self.bigrams.get((word1, word2), 0)

        numerator = bigram_count + self.add_constant
        denominator = unigram_count + self.add_constant * self.vocab_size

        return max(numerator / denominator, 1e-12)

  
    #Bigram_count for accessing synonym cache
    def build_bigram_dict(self):
      bigram_counts = defaultdict(Counter)

      for (w1, w2), count in self.bigrams.items():
          bigram_counts[w1][w2] = count

      return bigram_counts


    # Next word prediction
    def predict_next(self, current):

        best_word = None
        best_probab = 0

        for word2 in self.unigrams:

            if word2 == "<s>":
                continue

            prob = self.compute_probs(current, word2)

            if prob > best_probab:
                best_probab = prob
                best_word = word2

        return best_word, best_probab
      
    #This is used for naive empirical d/n interpolation
    #comment out to experiment

    #def compute_d_over_n(self):

      #d_over_n = {}

     # for word in self.unigrams:

          #n = self.unigrams.get(word, 0)                 # occurrences
          #d = len(self.continuations[word])       # distinct next words

          #if n > 0:
              #d_over_n[word] = d / n
          #else:
              #d_over_n[word] = 0

      #return d_over_n

