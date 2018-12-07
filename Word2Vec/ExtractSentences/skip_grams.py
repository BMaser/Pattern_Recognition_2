#!/usr/bin/python3 -u

import sys, argparse, re, random

def keepProbabilities(chunks, vocabulary):
  occurrences = [0] * len(vocabulary)
  for chunk in chunks:
    for word in chunk:
      if word in vocabulary:
        occurrences[vocabulary[word]] += 1
  total = sum(occurrences)
  relative = [o / float(total) for o in occurrences]
  keepProbs = []
  for i in range(len(vocabulary)):
    if relative[i] > 0.0:
      v = (((relative[i] / 0.001)**0.5) + 1.0) * 0.001 / relative[i]
      v = min(v, 1.0)
      keepProbs.append(v)
    else:
      keepProbs.append(0.0)
  return keepProbs

def filterWords(words, vocabulary, keepProbs=None):
  filtered = []
  for w in words:
    if w in vocabulary:
      if keepProbs is not None:
        if random.random() < keepProbs[vocabulary[w]]:
          filtered.append(w)
        else:
          filtered.append(None)
      else:
        filtered.append(w)
    else:
      filtered.append(None)
  return filtered

def generateSkipGrams(words, vocabulary, windowSize):
  skipGrams = []
  for i in range(len(words)):
    currentWord = words[i]
    if currentWord is not None:
      lowerLimit = max(0, i - windowSize)
      upperLimit = min(len(words) - 1, i + windowSize)
      for iOther in range(lowerLimit, upperLimit + 1):
        otherWord = words[iOther]
        if iOther != i and otherWord is not None:
          skipGrams.append((currentWord, otherWord))
  return skipGrams

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-v", "--vocabularyFile", required=True)
  parser.add_argument("-w", "--windowSize", type=int, required=True)
  parser.add_argument("-b", "--separators",
                      choices=["none", "line", "dot", "comma"],
                      default="none")
  parser.add_argument("-s", "--subsampling", action="store_true")
  args = parser.parse_args()
  separatorPatterns = {"line" : "\n",
                       "dot" : "\n|!|\\?|\\.",
                       "comma" : "\n|!|\\?|\\.|,|:|;"}
  
  with open(args.vocabularyFile) as f:
    vocabulary = f.read().splitlines()
  vocabulary = {v:i for i, v in enumerate(vocabulary)}
  text = sys.stdin.read().lower()
  if args.separators != "none":
    chunks = re.split(separatorPatterns[args.separators], text)
  else:
    chunks = [text]
  chunks = [re.findall("[A-Za-z]+", c) for c in chunks]
  if args.subsampling:
    keepProbs = keepProbabilities(chunks, vocabulary)
    chunks = [filterWords(c, vocabulary, keepProbs) for c in chunks]
  else:
    chunks = [filterWords(c, vocabulary) for c in chunks]
  skipGrams = [sg for c in chunks
                  for sg in generateSkipGrams(c, vocabulary, args.windowSize)]
  for sg in skipGrams:
    print(sg[0] + " " + sg[1])

if __name__ == "__main__":
  main()
