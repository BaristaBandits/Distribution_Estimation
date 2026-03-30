import nltk
nltk.download('punkt_tab')
from datasets import load_dataset

class DataPreprocessor():

  def __init__(self, dataset, train_size=100000, test_size=10000, valid_size=10000):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_data = dataset["train"]

