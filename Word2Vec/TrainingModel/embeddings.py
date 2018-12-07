from keras.models import load_model
import argparse

def print_embeddings(model, vocabulary, file_handle):
  weights = model.layers[0].get_weights()[0]
  for i, embedding in enumerate(weights):
    line = vocabulary[i] + " "
    line += " ".join(str(v) for v in embedding)
    print(line, file=file_handle)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--modelFile",
                      default="keras_model")
  parser.add_argument("-v", "--vocabularyFile",
                      default="../ExtractSentences/wikipedia_top1000.txt")
  parser.add_argument("-e", "--embeddingFile",
                      default="embeddings.txt")
  args = parser.parse_args()
  model = load_model(args.modelFile)
  with open(args.vocabularyFile) as f:
    vocabulary = f.read().splitlines()
  with open(args.embeddingFile, "w") as f:
    print_embeddings(model, vocabulary, f)

if __name__ == "__main__":
  main()
