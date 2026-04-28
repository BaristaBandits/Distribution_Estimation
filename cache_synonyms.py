#!/usr/bin/env python3
from math import log, floor
import numpy as np
import scipy.special

class Support():
    """
    Support estimator
    """
    def __init__(self, pmin, L=None, M=None):
        """
        Args:
        pmin: float, required
        =1/k. Preset minimum non-zero mass

        L: int
        Polynoimal degree. Default c0*log(k)

        M: float
        M/n is the right-end of approximation interval. Default c1*log(k)

        """
        self.pmin = pmin
        self.degree = L if L != None else floor(0.45*log(1./pmin))
        self.ratio = M if M != None else 0.5*log(1./pmin)

    def estimate(self, fin):
        """
        Our rate-optimal estimator from a given fingerprint

        Args:
        fin: list of tuples (frequency, count)
        fingerprint of samples, fin[i] is the number of symbols that appeared exactly i times

        Return:
        the estimated entropy (bits)
        """
        # get total sample size
        num = get_sample_size(fin)

        if self.pmin < self.ratio / num:
            # get linear estimator coefficients
            cheb = np.polynomial.chebyshev.Chebyshev.basis(self.degree)
            cheb_coeffs = np.polynomial.chebyshev.cheb2poly(cheb.coef)
            shift = (self.ratio + num*self.pmin) / (self.ratio - num*self.pmin)
            a_coeffs = shift_polynomial(cheb_coeffs, -shift)
            g_coeffs = -a_coeffs / a_coeffs[0]
            g_coeffs[0] = 0
            scaling = 2. / (self.ratio-num*self.pmin)
            for j in range(1, self.degree+1):
                for i in range(1, j+1):
                    g_coeffs[j] *= (i*scaling)
                g_coeffs[j] += 1

            # get estimate
            s_estimate = 0
            for freq, cnt in fin:
                if freq > self.degree: # plug-in
                    s_estimate += 1*cnt
                else: # polynomial
                    s_estimate += g_coeffs[freq]*cnt
        else:
            s_estimate = self.estimate_plug(fin)

        return s_estimate

    def estimate_plug(self, fin):
        """
        Plug-in estimate of support size: number of seen symbols
        """
        s_estimate = 0
        for freq, cnt in fin:
            s_estimate += cnt
        return s_estimate

    def coverage_turing_good(self, fin):
        """
        Turing-Good coverage
        """
        num = get_sample_size(fin)
        fin1 = 0
        for freq, cnt in fin:
            if freq == 1:
                fin1 = cnt
                break
        return 1.-1.*fin1/num

    def estimate_turing_good(self, fin):
        """
        Turing-Good estimator
        Ref: "The Population Frequencies of Species and the Estimation of Population Parameters",
        I. J. Good, 1953
        """
        return 1.*self.estimate_plug(fin)/self.coverage_turing_good(fin)

    def estimate_jackknife1(self, fin):
        """
        First order Jackknife
        Ref: "Robust Estimation of Population Size When Capture Probabilities Vary Among Animals",
        K. P. Burnham and W. S. Overton, 1979
        """
        num = get_sample_size(fin)
        fin1 = 0
        for freq, cnt in fin:
            if freq == 1:
                fin1 = cnt
                break
        return self.estimate_plug(fin) + (num-1.0)/num*fin1

    def estimate_chao1(self, fin):
        """
        Chao 1 estimator
        Ref original Chao 1: "Nonparametric estimation of the number of classes in a population."
        A. Chao, 1984
        Ref this bias-corrected Chao 1: "Estimating species richness",
        N. J. Gotelli and R. K. Colwell, 2011
        """
        fin1 = 0
        fin2 = 0
        for freq, cnt in fin:
            if freq == 1:
                fin1 = cnt
                break
        for freq, cnt in fin:
            if freq == 2:
                fin2 = cnt
                break
        return self.estimate_plug(fin)+fin1*(fin1-1)/(2*(fin2+1))

    def estimate_chao_lee1(self, fin):
        """
        Chao-Lee 1, for small coefficient of variation
        Ref: "Estimating the Number of Classes via Sample Coverage", A. Chao and S. Lee, 1992
        """
        num = get_sample_size(fin)
        estimate1 = self.estimate_turing_good(fin)
        coverage = self.coverage_turing_good(fin)

        sums = 0.
        for freq, cnt in fin:
            sums += freq*(freq-1)*cnt
        gamma_sq = max(1.*estimate1*sums/num/(num-1)-1, 0)
        return estimate1 + num*(1-coverage)/coverage*gamma_sq

    def estimate_chao_lee2(self, fin):
        """
        Chao-Lee 2, for large coefficient of variation
        Ref: "Estimating the Number of Classes via Sample Coverage", A. Chao and S. Lee, 1992
        """
        num = get_sample_size(fin)
        estimate1 = self.estimate_turing_good(fin)
        coverage = self.coverage_turing_good(fin)

        sums = 0.
        for freq, cnt in fin:
            sums += freq*(freq-1)*cnt
        gamma_sq = max(1.*estimate1*sums/num/(num-1)-1, 0)
        gamma_sq2 = max(gamma_sq*(1+num*(1-coverage)*sums/num/(num-1)/coverage), 0)
        return estimate1 + num*(1-coverage)/coverage*gamma_sq2

def get_sample_size(fin):
    """
    get total sample size from a given fingerprint
    """
    num = 0
    for freq, cnt in fin:
        num += freq * cnt
    return num


def sample_to_fin(samples):
    """
    Return a fingerprint from samples
    """
    return hist_to_fin(sample_to_hist(samples))


def hist_to_fin(hist):
    """
    Return a fingerprint from histogram
    """
    fin = Counter()
    for freq in hist:
        fin[freq] += 1
    return fin.items()

def sample_to_hist(samples):
    """
    Return a histogram of samples
    """
    freq = Counter()
    for symbol in samples:
        freq[symbol] += 1
    return np.asarray(list(freq.values()))


def shift_polynomial(coeffs, shift):
    """
    Return coefficients of a shifted polynomial
    P(x) = p(x+shift)

    Args:
    coeffs: list of float
    coefficients of p

    shift: float
    right shift of polynomial

    Returns:
    coefficients of P
    """
    length = len(coeffs)
    coeffs_shift = np.zeros(length)
    for i in range(length):
        for j in range(i+1):
            coeffs_shift[j] += coeffs[i] * (shift**(i-j)) * scipy.special.binom(i, j)
    return coeffs_shift


class Counter(dict):
    """
    Class for counting items
    """
    def __missing__(self, key):
        return 0

support = Support(pmin = 1e-5)


import numpy as np
from collections import defaultdict, Counter

def build_cache(train_corpus, support, embeddings, k=50):

  #================COUNTING==================
  bigram_counts = defaultdict(Counter)
  unigram_counts = Counter()
  vocab = set()
  for tokens in train_corpus:
      for i in range(len(tokens)):
          word = tokens[i]
          vocab.add(word)
          unigram_counts[word] += 1
          if i > 0:
              prev = tokens[i-1]
              curr = tokens[i]
              bigram_counts[prev][curr] += 1

  V = len(vocab)
  print("Vocabulary size:", V)

  #==================D_Estimate================
  d_estimate = {}

  for word in vocab:
      next_word_counts = bigram_counts.get(word, Counter())

      if len(next_word_counts) == 0:
          d_estimate[word] = 0
          continue

      freq_of_freq = Counter(next_word_counts.values())
      fin = [[freq, count] for freq, count in sorted(freq_of_freq.items())]

      d_estimate[word] = support.estimate(fin)

  d_estimate = {key: np.maximum(1e-4, value) for (key, value) in d_estimate.items()}

  print("===============Sanity Check=============")
  print('the', d_estimate["the"])
  print('a', d_estimate["a"])
  print('from', d_estimate["from"])

  #==================Building_Vocab================
  d = {}
  n = {}

  for word in vocab:
      d[word] = sum(1 for c in bigram_counts[word].values() if c >= 1)
      n[word] = unigram_counts[word]

  stoi = {w: i for i, w in enumerate(vocab)}
  itos = {i: w for w, i in stoi.items()}

  print(d['the'])
  print(n['the'])

  print("Vocab size:", len(stoi))
  print("Top words:", list(stoi.keys())[:20])



  #==================Max Norm================

  max_norm = 0
  max_word = None

  for word in embeddings.index_to_key:
      vec = embeddings[word]
      norm = np.linalg.norm(vec, ord = np.inf)

      if norm > max_norm:
          max_norm = norm
          max_word = word

  print("Max L1 norm:", max_norm)
  print("Word with max norm:", max_word)

  #fixing max_norm to be a constant 
  max_norm = 5.0

  #==================Caching================
  words_ge = [w for w in vocab if n.get(w, 0) >= 10 and d.get(w, 0) >= 5 and w in embeddings]
  print(len(words_ge))

  candidate_matrix = np.stack([embeddings[w] for w in words_ge])   # shape: (M, D)
  n_vec = np.array([n[w] for w in words_ge], dtype=np.float32)
  d_est_vec = np.array([d_estimate[w] for w in words_ge], dtype=np.float32)

  base_term = d_est_vec / (2.0 * n_vec)   # shape: (M,)

  top20_cache = {}

  for i, prev_word in enumerate(vocab):

      if prev_word not in embeddings:
          top20_cache[prev_word] = []
          continue

      prev_vec = embeddings[prev_word]   # shape: (D,)

      # L2 distance to all candidate words
      diff = candidate_matrix - prev_vec
      Delta = np.linalg.norm(diff, axis=1, ord = 1) * max_norm
      error = base_term + Delta + np.sqrt(Delta) * np.log(d_est_vec + n_vec)

      weights = 1.0 / (error)
      top_idx = np.argpartition(weights, -k)[-k:]
      top_idx = top_idx[np.argsort(weights[top_idx])[::-1]]

      top20_cache[prev_word] = [(words_ge[j], float(weights[j])) for j in top_idx]

      if i % 10000 == 0:
          print(i, prev_word, top20_cache[prev_word])

  #Return Statement
  return d_estimate, top20_cache
