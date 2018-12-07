from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.initializers import Zeros, RandomNormal, RandomUniform
from keras.backend import set_session
import tensorflow as tf
import numpy as np
import argparse
import sys
import os

def load_vocabulary(file_name):
  with open(file_name, "r") as f:
    lines = [l.strip().lower() for l in f.readlines() if not l.isspace()]
  return lines

def load_pairs(training_file, vocabulary):
  with open(training_file, "r") as f:
    lines = [l.strip().lower() for l in f.readlines() if not l.isspace()]
  word_pairs = [l.split(" ") for l in lines]
  index_pairs = [(vocabulary.index(w1), vocabulary.index(w2)) for w1, w2 in word_pairs]
  return index_pairs

def prepare_input_output_matrix(index_pairs, vocabulary_size):
  input_matrix = np.zeros((len(index_pairs), vocabulary_size), np.byte)
  output_matrix = np.zeros((len(index_pairs), vocabulary_size), np.byte)
  for i, pair in enumerate(index_pairs):
    input_matrix[i][pair[0]] = 1
    output_matrix[i][pair[1]] = 1
  return input_matrix, output_matrix

def train_network(input_matrix, output_matrix, hidden_dimensions, epochs, batch_size, file_path):
  #~ config = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8,
                          #~ allow_soft_placement=True, device_count = {'CPU': 1})
  #~ session = tf.Session(config=config)
  #~ set_session(session)
  vector_size = input_matrix.shape[1]
  if os.path.isfile(file_path):
    model = load_model(file_path)
    model.compile("adam", loss="categorical_crossentropy")
  else:
    model = Sequential()
    #initializer = RandomUniform(minval=-0.0001, maxval=0.0001)
    initializer = "glorot_uniform"
    #~ initializer = "random_u"
    #~ initializer = RandomNormal(stddev=0.1)
    #~ optimizer = ["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"][0]
    optimizer = SGD(lr=1.0)
    #~ optimizer = 
    model.add(Dense(hidden_dimensions, input_dim=vector_size, activation="linear", use_bias=False, kernel_initializer=initializer))
    model.add(Dense(vector_size, activation="softmax", use_bias=False, kernel_initializer=initializer))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy")
  checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=0,
                               save_best_only=False, save_weights_only=False,
                               mode='auto', period=1)
  model.fit(input_matrix, output_matrix, epochs=epochs,
            batch_size=batch_size, callbacks=[checkpoint])

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-v", "--vocabularyFile",
                      default="../ExtractSentences/ef_top1000.txt")
  parser.add_argument("-t", "--trainingFile",
                      default="../ExtractSentences/skipgrams/news_10k.txt")
  parser.add_argument("-m", "--modelFile",
                      default="keras_model")
  parser.add_argument("-d", "--embeddingDimensions", type=int, default=128)
  parser.add_argument("-b", "--batchSize", type=int, default=32)
  parser.add_argument("-e", "--epochs", type=int, default=1000)
  args = parser.parse_args()
  
  vocabulary = load_vocabulary(args.vocabularyFile)
  index_pairs = load_pairs(args.trainingFile, vocabulary)
  #~ shuffle(index_pairs)
  #~ index_pairs = index_pairs[:100000]
  #~ index_pairs = index_pairs[:1]
  input_matrix, output_matrix = prepare_input_output_matrix(index_pairs, len(vocabulary))
  train_network(input_matrix, output_matrix, args.embeddingDimensions,
                args.epochs, args.batchSize, args.modelFile)

if __name__ == "__main__":
  main()
