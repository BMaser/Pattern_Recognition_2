#!/usr/bin/python3

import sys, argparse
import sentences

def readWords():
  lines = [l.strip().lower() for l in sys.stdin.readlines() if not l.isspace()]
  words = [w for l in lines for w in l.split(" ")]
  return words

def countVocabulary(words, vocabulary):
  counts = [sum(1 for w in words if w == v) for v in vocabulary]
  countsValues = sorted(list(set(counts)))
  countsCounts = [(k, [vocabulary[i] for i in range(len(vocabulary)) if counts[i] == k])
                  for k in countsValues]
  return countsCounts

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-v", "--vocabularyFile", required=True)
  args = parser.parse_args()
  vocabulary = sentences.loadVocabulary(args.vocabularyFile)
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
