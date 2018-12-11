#!/usr/bin/python3 -u

import sys, argparse, re, random, itertools

def keepProbabilities(words, vocabulary):
  occurrences = {word : 0 for word in vocabulary}
  for word in words:
    if word in vocabulary:
      occurrences[word] += 1
  total = sum(occurrences[word] for word in vocabulary)
  relative = {word : occurrences[word] / float(total) for word in vocabulary}
  keepProbs = {}
  for word in vocabulary:
    if relative[word] > 0.0:
      v = (((relative[word] / 0.001)**0.5) + 1.0) * 0.001 / relative[word]
      v = min(v, 1.0)
      keepProbs[word] = v
    else:
      keepProbs[word] = 1.0
  return keepProbs

def generateSkipGrams(words, windowSize):
  for i, word in enumerate(words):
    if word is not None:
      lowerLimit = max(0, i - windowSize)
      upperLimit = min(len(words) - 1, i + windowSize)
      for iOther in range(lowerLimit, upperLimit + 1):
        otherWord = words[iOther]
        if iOther != i and otherWord is not None:
          yield (word, otherWord)

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
    vocabulary = set(f.read().splitlines())
  text = sys.stdin.read().lower()
  if args.separators != "none":
    chunks = re.split(separatorPatterns[args.separators], text)
  else:
    chunks = [text]
  if args.subsampling:
    words = (m.group(0) for c in chunks for m in re.finditer("[A-Za-z]+", c))
    keepProbs = keepProbabilities(words, vocabulary)
  else:
    keepProbs = {w : 1.0 for w in vocabulary}
  wordChunks = ((m.group(0) for m in re.finditer("[A-Za-z]+", c)) for c in chunks)
  wordChunks = [[(w if w in vocabulary and random.random() < keepProbs[w] else None) for w in wc] for wc in wordChunks]
  skipGrams = (sg for wc in wordChunks for sg in generateSkipGrams(wc, args.windowSize))
  for sg in skipGrams:
    print(sg[0] + " " + sg[1])

if __name__ == "__main__":
  main()
