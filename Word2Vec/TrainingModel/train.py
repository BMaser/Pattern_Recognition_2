import tensorflow as tf
import numpy as np
import argparse
import sys

def load_vocabulary(file_name):
  with open(file_name, "r") as f:
    lines = [l.strip().lower() for l in f.readlines() if not l.isspace()]
  return lines

def one_hot_vectors_list(vocab_size):
  vectors = []
  for i in range(vocab_size):
    vector = np.zeros(vocab_size, np.float32)
    vector[i] = 1.0
    vectors.append(vector)
  return vectors

def load_pairs(training_file, vocabulary):
  with open(training_file, "r") as f:
    lines = [l.strip().lower() for l in f.readlines() if not l.isspace()]
  word_pairs = [l.split(" ") for l in lines]
  index_pairs = [(vocabulary.index(w1), vocabulary.index(w2)) for w1, w2 in word_pairs]
  return index_pairs

def train_network(input_vectors, output_vectors, hidden_dimensions, learning_rate,
                  epochs, batch_size):
  vector_length = len(input_vectors[0])
  in_vec = tf.placeholder(tf.float32, [None, vector_length])
  out_vec = tf.placeholder(tf.float32, [None, vector_length])
  W = tf.Variable(tf.random_normal([vector_length, hidden_dimensions], mean=0.0, stddev=0.02, dtype=tf.float32))
  b = tf.Variable(tf.random_normal([hidden_dimensions], mean=0.0, stddev=0.02, dtype=tf.float32))
  W_outer = tf.Variable(tf.random_normal([hidden_dimensions, vector_length], mean=0.0, stddev=0.02, dtype=tf.float32))
  b_outer = tf.Variable(tf.random_normal([vector_length], mean=0.0, stddev=0.02, dtype=tf.float32))
  hidden = tf.add(tf.matmul(in_vec, W), b)
  logits = tf.add(tf.matmul(hidden, W_outer), b_outer)
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=out_vec))
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
  batches = len(input_vectors) // batch_size
  
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
      batch_index = 0
      for batch_num in range(batches + 1):
        in_batch = input_vectors[batch_index:batch_index + batch_size]
        out_batch = output_vectors[batch_index:batch_index + batch_size]
        sess.run(optimizer, feed_dict={in_vec: in_batch, out_vec: out_batch})
        print("epoch:", epoch, "loss:",
              sess.run(cost, feed_dict={in_vec: in_batch, out_vec: out_batch}),
              file=sys.stderr)
    W_embed_trained = sess.run(W)
  return W_embed_trained

def print_embeddings(weights):
  for embedding in weights:
    print(" ".join(str(x) for x in embedding))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-v", "--vocabularyFile",
                      default="../ExtractSentences/ef_top1000.txt")
  parser.add_argument("-t", "--trainingFile",
                      default="../ExtractSentences/skipgrams/news_10k.txt")
  parser.add_argument("-e", "--epochs", type=int, default=1)
  parser.add_argument("-d", "--embeddingDimensions", type=int, default=128)
  parser.add_argument("-r", "--learningRate", type=float, default=0.1)
  parser.add_argument("-b", "--batchSize", type=int, default=10)
  args = parser.parse_args()
  
  vocabulary = load_vocabulary(args.vocabularyFile)
  one_hot_vectors = one_hot_vectors_list(len(vocabulary))
  index_pairs = load_pairs(args.trainingFile, vocabulary)
  train_input = [one_hot_vectors[ip[0]] for ip in index_pairs]
  train_output = [one_hot_vectors[ip[1]] for ip in index_pairs]
  weights = train_network(train_input, train_output, args.embeddingDimensions,
                          args.learningRate, args.epochs, args.batchSize)
  print_embeddings(weights)

if __name__ == "__main__":
  main()
