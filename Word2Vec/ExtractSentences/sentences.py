#!/usr/bin/python3

import re

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
