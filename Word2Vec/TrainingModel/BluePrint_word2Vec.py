import tensorflow as tf
import numpy as np
import nltk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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

def load_skipgrams(file_name, vocabulary):
  with open(file_name, "r") as f:
    lines = [l.strip().lower() for l in f.readlines() if not l.isspace()]
  word_pairs = [l.split(" ") for l in lines]
  skipgrams = [(vocabulary.index(w1), vocabulary.index(w2)) for w1, w2 in word_pairs]
  return skipgrams

vocab_file_name = "../ExtractSentences/ef_top1000.txt"
skipgram_file_name = "../ExtractSentences/skipgrams/news_10k.txt"
vocabulary = load_vocabulary(vocab_file_name)
one_hot_vectors = one_hot_vectors_list(len(vocabulary))
skipgrams = load_skipgrams(skipgram_file_name, vocabulary)
skipgrams = skipgrams[:100000]

x_train = [one_hot_vectors[s[0]] for s in skipgrams]
y_train = [one_hot_vectors[s[1]] for s in skipgrams]

#---------------------------------------------
# Build the Neural Net and Invoke training
#---------------------------------------------
# Placeholders for Input output
#----------------------------------------------
x = tf.placeholder(tf.float32,[None, len(vocabulary)])
y = tf.placeholder(tf.float32,[None, len(vocabulary)])
#---------------------------------------------
# Define the Embedding matrix weights and a bias
#----------------------------------------------

emb_dims = 128
learning_rate = 1.0

W = tf.Variable(tf.random_normal([len(vocabulary),emb_dims],mean=0.0,stddev=0.02,dtype=tf.float32))
b = tf.Variable(tf.random_normal([emb_dims],mean=0.0,stddev=0.02,dtype=tf.float32))
W_outer = tf.Variable(tf.random_normal([emb_dims,len(vocabulary)],mean=0.0,stddev=0.02,dtype=tf.float32))
b_outer = tf.Variable(tf.random_normal([len(vocabulary)],mean=0.0,stddev=0.02,dtype=tf.float32))

hidden = tf.add(tf.matmul(x,W),b)
logits = tf.add(tf.matmul(hidden,W_outer),b_outer)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

epochs,batch_size = 1,10
batches = len(x_train) // batch_size

# train for n_iter iterations
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        batch_index = 0
        for batch_num in range(batches + 1):
            x_batch = x_train[batch_index: batch_index +batch_size]
            y_batch = y_train[batch_index: batch_index +batch_size]
            sess.run(optimizer,feed_dict={x: x_batch,y: y_batch})
            print('epoch:',epoch,'loss:', sess.run(cost,feed_dict={x: x_batch,y: y_batch}))
    W_embed_trained = sess.run(W)

W_embedded = TSNE(n_components=2).fit_transform(W_embed_trained)

plt.scatter(W_embedded[:,0], W_embedded[:,1], s=0)
for i in range(len(vocabulary)):
  plt.text(W_embedded[i,0], W_embedded[i,1], vocabulary[i])
plt.show()
