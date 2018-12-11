from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.initializers import Zeros, RandomNormal, RandomUniform
from keras.backend import set_session
from math import ceil
import tensorflow as tf
import numpy as np
import argparse
import sys
import os

def load_vocabulary(file_name):
  with open(file_name, "r") as f:
    lines = [l.strip().lower() for l in f.readlines() if not l.isspace()]
  return {word : i for i, word in enumerate(lines)}

def number_of_training_pairs(training_file):
  with open(training_file, "r") as f:
    return sum(1 for l in f if not l.isspace())

def generate_training_matrices(training_file, vocabulary, batch_size, cache_size=100000):
  while True:
    pairs_cache = []
    with open(training_file, "r") as f:
      while True:
        if len(pairs_cache) == 0:
          lines = [f.readline().strip() for i in range(cache_size)]
          word_pairs = [l.split(" ") for l in lines if len(l) > 0]
          pairs_cache = [(vocabulary[w1], vocabulary[w2]) for w1, w2 in word_pairs]
          if len(pairs_cache) == 0:
            break
        index_pairs, pairs_cache = pairs_cache[:batch_size], pairs_cache[batch_size:]
        input_matrix, output_matrix = prepare_input_output_matrix(index_pairs, len(vocabulary))
        yield (input_matrix, output_matrix)

def prepare_input_output_matrix(index_pairs, vocabulary_size):
  input_matrix = np.zeros((len(index_pairs), vocabulary_size), np.byte)
  output_matrix = np.zeros((len(index_pairs), vocabulary_size), np.byte)
  for i, pair in enumerate(index_pairs):
    input_matrix[i][pair[0]] = 1
    output_matrix[i][pair[1]] = 1
  return input_matrix, output_matrix

def train_network(generator, input_dim, hidden_dim, steps_per_epoch, epochs, learn_rate, model_file):
  initializer = "glorot_uniform"
  optimizer = SGD(lr=learn_rate)
  loss = "categorical_crossentropy"
  if os.path.isfile(model_file):
    model = load_model(model_file)
  else:
    model = Sequential()
    model.add(Dense(hidden_dim, input_dim=input_dim, activation="linear", use_bias=False, kernel_initializer=initializer))
    model.add(Dense(input_dim, activation="softmax", use_bias=False, kernel_initializer=initializer))
  model.compile(optimizer=optimizer, loss=loss)
  checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=0,
                               save_best_only=False, save_weights_only=False,
                               mode='auto', period=1)
  model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[checkpoint])

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-v", "--vocabularyFile",
                      default="../ExtractSentences/wikipedia_vocabulary.txt")
  parser.add_argument("-t", "--trainingFile",
                      default="../ExtractSentences/skipgrams/text8_ss.txt")
  parser.add_argument("-m", "--modelFile",
                      default="keras_model")
  parser.add_argument("-d", "--embeddingDimensions", type=int, default=128)
  parser.add_argument("-b", "--batchSize", type=int, default=32)
  parser.add_argument("-e", "--epochs", type=int, default=1000)
  parser.add_argument("-l", "--learnRate", type=float, default=1.0)
  args = parser.parse_args()
  
  vocabulary = load_vocabulary(args.vocabularyFile)
  steps_per_epoch = int(ceil(number_of_training_pairs(args.trainingFile) / args.batchSize))
  generator = generate_training_matrices(args.trainingFile, vocabulary, args.batchSize)
  train_network(generator, len(vocabulary), args.embeddingDimensions, steps_per_epoch,
                args.epochs, args.learnRate, args.modelFile)

if __name__ == "__main__":
  main()
