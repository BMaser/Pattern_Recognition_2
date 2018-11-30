from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse

def load_vocabulary(file_name):
  with open(file_name, "r") as f:
    lines = [l.strip().lower() for l in f.readlines() if not l.isspace()]
  return lines

def load_embeddings(file_name):
  with open(file_name, "r") as f:
    lines = [l.strip().lower() for l in f.readlines() if not l.isspace()]
  embeddings = [[float(s) for s in l.split(" ")] for l in lines]
  return embeddings

def plot_vocabulary_2d(vocabulary, coordinates, dot_size):
  plt.scatter(coordinates[:,0], coordinates[:,1], s=dot_size)
  for i, word in enumerate(vocabulary):
    plt.text(coordinates[i][0], coordinates[i][1], word)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-v", "--vocabularyFile",
                      default="../ExtractSentences/ef_top1000.txt")
  parser.add_argument("-e", "--embeddingFile",
                      default="embeddings.txt")
  parser.add_argument("-s", "--dotSize", type=float, default=1.0)
  args = parser.parse_args()
  vocabulary = load_vocabulary(args.vocabularyFile)
  embeddings = load_embeddings(args.embeddingFile)
  embeddings_embedded = TSNE(n_components=2).fit_transform(embeddings)
  plot_vocabulary_2d(vocabulary, embeddings_embedded, args.dotSize)
  plt.show()

if __name__ == "__main__":
  main()
