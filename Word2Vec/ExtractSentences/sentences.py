#!/usr/bin/python3

import re
import random

def loadVocabulary(fileName):
  with open(fileName, "r") as f:
    lines = [l.strip().lower() for l in f.readlines() if not l.isspace()]
  return lines

def extractChunks(content, vocabulary):
  pattern = re.compile("[A-za-z]+")
  words = re.findall(pattern, content)
  chunks, chunk = [], []
  for w in words:
    lw = w.lower()
    if lw in vocabulary:
      chunk.append(lw)
    else:
      chunks.append(chunk)
      chunk = []
  if len(chunk) > 0:
    chunks.append(chunk)
  return chunks

def keepProbability(relOccurrence):
  if relOccurrence == 0.0:
    return 0.0
  v = (((relOccurrence / 0.001)**0.5) + 1)
  v *= 0.001 / relOccurrence
  return min(v, 1.0)

def subsampleChunks(chunks, vocabulary):
  occurrences = [0] * len(vocabulary)
  for c in chunks:
    for w in c:
      occurrences[vocabulary.index(w)] += 1
  numberOfWords = sum(occurrences)
  relOccurrences = [n / float(numberOfWords) for n in occurrences]
  keepProbabilities = [keepProbability(p) for p in relOccurrences]
  sampledChunks = []
  for c in chunks:
    sc = []
    for w in c:
      if random.random() < keepProbabilities[vocabulary.index(w)]:
        sc.append(w)
    if len(sc) > 0:
      sampledChunks.append(sc)
  return sampledChunks
