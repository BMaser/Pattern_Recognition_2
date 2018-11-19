#!/usr/bin/python3 -u

import sys, argparse
import sentences

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-v", "--vocabularyFile", required=True)
  parser.add_argument("-n", "--minChunkSize", type=int, required=True)
  args = parser.parse_args()
  vocabulary = sentences.loadVocabulary(args.vocabularyFile)
  content = sys.stdin.read()
  chunks = sentences.extractChunks(content, vocabulary)
  chunks = [c for c in chunks if len(c) >= args.minChunkSize]
  for c in chunks:
    print(" ".join(c))

if __name__ == "__main__":
  main()
