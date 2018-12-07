#!/usr/bin/python3

import sys, argparse

def readWords():
  lines = [l.strip().lower() for l in sys.stdin.readlines() if not l.isspace()]
  words = [w for l in lines for w in l.split(" ")]
  return words

def countVocabulary(words, vocabulary):
  counts = [0] * len(vocabulary)
  vocabularyWords = sorted(vocabulary.keys())
  for w in words:
    counts[vocabulary[w]] += 1
  countsValues = sorted(list(set(counts)))
  countsCounts = [(k, [vocabularyWords[i] for i in range(len(vocabulary)) if counts[i] == k])
                  for k in countsValues]
  return countsCounts

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-v", "--vocabularyFile", required=True)
  args = parser.parse_args()
  with open(args.vocabularyFile) as f:
    vocabulary = f.read().splitlines()
  vocabulary = {v:i for i, v in enumerate(vocabulary)}
  words = readWords()
  counts = countVocabulary(words, vocabulary)
  for count, words in counts:
    if len(words) > 5:
      wordsString = ", ".join(words[:5]) + ", ..."
    else:
      wordsString = ", ".join(words)
    print("{} {} {}".format(count, len(words), wordsString))

if __name__ == "__main__":
  main()
