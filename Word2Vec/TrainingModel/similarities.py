import argparse
import numpy as np
import re

def normalized(v):
  return v / np.sqrt(sum(v**2))

def load_embeddings(file_name):
  with open(file_name, "r") as f:
    lines = f.read().splitlines()
  embeddings = {}
  for l in lines:
    lSplit = l.split(" ")
    word = lSplit[0]
    embedding = np.array([float(f) for f in lSplit[1:]], dtype=np.float32)
    embeddings[word] = embedding
  return embeddings

def parse_term(s):
  term = []
  for m in re.finditer("(\\+|-)?(\\w+)", s):
    word, symbol = m.group(2), m.group(1)
    if symbol is None or symbol == "+":
      factor = 1
    elif symbol == "-":
      factor = -1
    else:
      raise ValueError("invalid symbol")
    term.append((word, factor))
  return term

def embedding_sum(embeddings, term):
  vector = np.zeros(len(next(iter(embeddings.values()))), dtype=np.float32)
  for word, factor in term:
    vector += embeddings[word] * factor
  return vector

def cosine_similarities(embeddings, vector):
  similarities = {}
  for word, embedding in embeddings.items():
    similarity = embedding.dot(vector) / (np.linalg.norm(embedding) * np.linalg.norm(vector))
    similarities[word] = similarity
  return similarities

def print_sorted_similarities(similarities):
  similarities = [(word, similarity) for word, similarity in similarities.items()]
  similarities = sorted(similarities, key = lambda s : (-s[1], s[0]))
  for word, similarity in similarities:
    print(similarity, word)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-e", "--embeddingFile",
                      default="embeddings.txt")
  parser.add_argument("-n", "--normalize", action="store_true")
  parser.add_argument("wordsTerm")
  args = parser.parse_args()
  embeddings = load_embeddings(args.embeddingFile)
  if args.normalize:
    embeddings = {word: normalized(embedding) for word, embedding in embeddings.items()}
  term = parse_term(args.wordsTerm)
  term_words = set(word for word, symbol in term)
  sum_vector = embedding_sum(embeddings, term)
  if args.normalize:
    sum_vector = normalized(sum_vector)
  compared_embeddings = {word: embedding for word, embedding in embeddings.items() if word not in term_words}
  similarities = cosine_similarities(compared_embeddings, sum_vector)
  print_sorted_similarities(similarities)

if __name__ == "__main__":
  main()
