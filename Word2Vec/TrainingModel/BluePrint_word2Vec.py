
# coding: utf-8

# In[5]:


import tensorflow as tf
import numpy as np
import nltk


# In[6]:


def one_hot(ind,vocab_size):
    rec = np.zeros(vocab_size)
    rec[ind] = 1
    return rec
def xrange(x):

    return iter(range(x))


# In[20]:


def data_processing(corpus,windSize):
    world_list=[]
    text=corpus
    for sentence in text.split('.'):
#         print(sentence)
        for word in sentence.split():
#             print(word)
            world_list.append(word)
#     print(world_list)
#     print('len :', len(world_list))
    world_list=set(world_list)
#     print(world_list)
        
#     word to index 
#     index to word
    dic_wordind={}
    dic_indword={}
    for i,w  in enumerate(world_list):
#         print(i, w)
        dic_wordind[w]=i
        dic_indword[i]=w
    
#     print(dic_wordind)
#     print(dic_indword)
#     print(world_list)
#     each sentence must form a list 
    arrWords=[]
    for sentence in text.split('.'):
#         print(sentence)
        for listWord in [sentence.split()]:
#             print(listWord) 
            arrWords=arrWords+[listWord]

    print(arrWords)    
    print()
    data_recs=[]
    for words in arrWords:
#         print(words)
        for ind,w in enumerate(words):
            rec = []
            lowBoundry = max(ind - windSize, 0)
            upperBoundry = min(ind + windSize, len(words)) + 1
#             print(lowBoundry,':', upperBoundry)
            for nb_w in words[lowBoundry :upperBoundry ] : 
#                 print(lowBoundry,':', upperBoundry)
                if nb_w != w:
                    rec.append(nb_w)
                    data_recs.append([rec,w,])
    print(data_recs)
    x_train=[]
    y_train=[]
#     dic_wordind
#     dic_indword
    vocab_size=len(world_list)
    for rec in data_recs:
        input_ = np.zeros(vocab_size)
        for i in xrange(windSize-1):
            input_ += one_hot(dic_wordind[ rec[0][i] ], vocab_size)
        input_ = input_/len(rec[0])
        x_train.append(input_)
        y_train.append(one_hot(dic_wordind[ rec[1] ], vocab_size))
        
        
    return x_train,y_train,dic_wordind,dic_indword,vocab_size
    




# In[25]:


# corpus_raw = "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 19. 20 21 22 23 24."
corpus_raw ="the quick brown fox jumped over the lazy dog"
corpus_raw = (corpus_raw).lower()
x_train,y_train,dic_wordind,dic_indword,vocab_size = data_processing(corpus_raw,windSize=2)


# In[26]:


# x_train


# In[27]:


# y_train


# In[28]:


#---------------------------------------------
# Build the Neural Net and Invoke training
#---------------------------------------------
# Placeholders for Input output
#----------------------------------------------
x = tf.placeholder(tf.float32,[None,vocab_size])
y = tf.placeholder(tf.float32,[None,vocab_size])
#---------------------------------------------
# Define the Embedding matrix weights and a bias
#----------------------------------------------

emb_dims = 128
learning_rate = 0.001

W = tf.Variable(tf.random_normal([vocab_size,emb_dims],mean=0.0,stddev=0.02,dtype=tf.float32))
b = tf.Variable(tf.random_normal([emb_dims],mean=0.0,stddev=0.02,dtype=tf.float32))
W_outer = tf.Variable(tf.random_normal([emb_dims,vocab_size],mean=0.0,stddev=0.02,dtype=tf.float32))
b_outer = tf.Variable(tf.random_normal([vocab_size],mean=0.0,stddev=0.02,dtype=tf.float32))

hidden = tf.add(tf.matmul(x,W),b)
logits = tf.add(tf.matmul(hidden,W_outer),b_outer)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

epochs,batch_size = 100,10
batch = len(x_train)//batch_size

# train for n_iter iterations
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print ('was here')
    for epoch in xrange(epochs):
        batch_index = 0 
        for batch_num in xrange(batch):
            x_batch = x_train[batch_index: batch_index +batch_size]
            y_batch = y_train[batch_index: batch_index +batch_size]
            sess.run(optimizer,feed_dict={x: x_batch,y: y_batch})
            print('epoch:',epoch,'loss :', sess.run(cost,feed_dict={x: x_batch,y: y_batch}))
    W_embed_trained = sess.run(W)


# In[29]:


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
W_embedded = TSNE(n_components=2).fit_transform(W_embed_trained)
plt.figure(figsize=(10,10))
for i in xrange(len(W_embedded)):
    plt.text(W_embedded[i,0],W_embedded[i,1],dic_indword[i])

plt.xlim(-400,400)
plt.ylim(-400,400)
print ("TSNE plot of the Word Vector Embeddings")

