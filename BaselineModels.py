from collections import defaultdict

class JelinekMercerSmoothing:

  def __init__(self, data):
    self.unigrams= defaultdict(int)
    self.bigrams=defaultdict(int)
    self.total_unigrams=0
    self.compute_grams(data)
    self.lambda_=0.5
    
  #Set_lambda invoked customly so that class computes unigrams and bigrams only once on the test corpus
  def set_lambda(self, lambda_):
    self.lambda_=lambda_
    
  # Assume that the data is a set of all sentences
  def compute_grams(self, data):
    for sentence in data:
      tokens= ["<s>"] + sentence.split() + ["</s>"]
      for i in range(len(tokens)):
        self.unigrams[tokens[i]]+=1
        self.total_unigrams+=1
        if i>0:
          self.bigrams[(tokens[i-1], tokens[i])]+=1

  # Jelenik mercer smoothing
  def compute_probs(self, word1, word2):
    count_word1 = self.unigrams.get(word1, 0)

    unigram_prob = self.unigrams.get(word2, 0)/ self.total_unigrams
    bigram_prob = self.bigrams.get((word1, word2), 0)/ count_word1 if count_word1>0 else 0
    return max(self.lambda_ * bigram_prob + (1 - self.lambda_) * unigram_prob, 1e-12)

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


