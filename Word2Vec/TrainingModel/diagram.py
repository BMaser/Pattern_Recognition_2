from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse

def load_embeddings(file_name):
  with open(file_name, "r") as f:
    lines = f.read().splitlines()
  lines = [l.split(" ") for l in lines]
  vocabulary = [l[0] for l in lines]
  embeddings = [[float(f) for f in l[1:]] for l in lines]
  return vocabulary, embeddings

def plot_vocabulary_2d(vocabulary, coordinates, dot_size):
  plt.scatter(coordinates[:,0], coordinates[:,1], s=dot_size)
  for i, word in enumerate(vocabulary):
    plt.text(coordinates[i][0], coordinates[i][1], word)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-e", "--embeddingFile",
                      default="embeddings.txt")
  parser.add_argument("-s", "--dotSize", type=float, default=1.0)
  args = parser.parse_args()
  vocabulary, embeddings = load_embeddings(args.embeddingFile)
  embeddings_embedded = TSNE(n_components=2, random_state=1).fit_transform(embeddings)
  plot_vocabulary_2d(vocabulary, embeddings_embedded, args.dotSize)
  plt.show()

if __name__ == "__main__":
  main()
