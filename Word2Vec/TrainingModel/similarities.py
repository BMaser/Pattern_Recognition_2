import argparse
import numpy as np
import re

def load_vocabulary(file_name):
  with open(file_name, "r") as f:
    lines = [l.strip().lower() for l in f.readlines() if not l.isspace()]
  return lines

def load_embeddings(file_name):
  with open(file_name, "r") as f:
    lines = [l.strip().lower() for l in f.readlines() if not l.isspace()]
  embeddings = [np.array([float(s) for s in l.split(" ")], dtype=np.float32) for l in lines]
  return embeddings

def embedding_sum(vocabulary, embeddings, words):
  vector = np.zeros(len(embeddings[0]), dtype=np.float32)
  pattern = re.compile("(\\+|-)?(\\w+)")
  for match in re.finditer(pattern, words):
    symbol, word = match.group(1), match.group(2)
    current_vector = embeddings[vocabulary.index(word)]
    if symbol is None or symbol == "+":
      pass
    elif symbol == "-":
      current_vector *= -1
    else:
      raise ArgumentError("invalid symbol")
    vector += current_vector
  return vector

def cosine_similarities(embeddings, vector):
  similarities = []
  for embedding in embeddings:
    similarity = embedding.dot(vector) / (np.linalg.norm(embedding) * np.linalg.norm(vector))
    similarities.append(similarity)
  return similarities

def print_sorted_similarities(vocabulary, similarities):
  similarities = [(similarity, vocabulary[i]) for i, similarity in enumerate(similarities)]
  similarities = sorted(similarities, key = lambda s : (-s[0], s[1]))
  for similarity, word in similarities:
    print(similarity, word)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-v", "--vocabularyFile",
                      default="../ExtractSentences/ef_top1000.txt")
  parser.add_argument("-e", "--embeddingFile",
                      default="embeddings.txt")
  parser.add_argument("words")
  args = parser.parse_args()
  vocabulary = load_vocabulary(args.vocabularyFile)
  embeddings = load_embeddings(args.embeddingFile)
  vector = embedding_sum(vocabulary, embeddings, args.words)
  similarities = cosine_similarities(embeddings, vector)
  print_sorted_similarities(vocabulary, similarities)

if __name__ == "__main__":
  main()
