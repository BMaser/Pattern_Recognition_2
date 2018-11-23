#!/usr/bin/python3

import sys, argparse

def readSentences():
  return [l.strip().split(" ") for l in sys.stdin.readlines() if not l.isspace()]

def generateSkipGrams(sentence, windowSize):
  return [ (sentence[i], sentence[j])
           for i in range(len(sentence))
           for j in range(max(0, i - windowSize),
                          min(i + windowSize + 1, len(sentence)))
           if i != j]

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-w", "--windowSize", type=int, required=True)
  args = parser.parse_args()
  sentences = readSentences()
  skipGrams = [sg for s in sentences for sg in generateSkipGrams(s, args.windowSize)]
  for sg in skipGrams:
    print(" ".join(sg))

if __name__ == "__main__":
  main()
