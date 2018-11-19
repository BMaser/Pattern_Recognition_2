#!/usr/bin/python3 -u

import wikipedia, re, argparse
import sentences

def extractFromRandomPage(vocabulary, language, minSize):
  wikipedia.set_lang(language)
  title = wikipedia.random()
  try:
    content = wikipedia.page(title).content
  except wikipedia.exceptions.WikipediaException:
    return []
  chunks = sentences.extractChunks(content, vocabulary)
  chunks = [c for c in chunks if len(c) >= minSize]
  return chunks

def extractLoop(vocabulary, language, minSize):
  while True:
    for c in extractFromRandomPage(vocabulary, language, minSize):
      print(" ".join(c))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-v", "--vocabularyFile", required=True)
  parser.add_argument("-l", "--languageCode", required=True)
  parser.add_argument("-n", "--minChunkSize", type=int, required=True)
  args = parser.parse_args()
  vocabulary = sentences.loadVocabulary(args.vocabularyFile)
  extractLoop(vocabulary, args.languageCode, args.minChunkSize)

if __name__ == "__main__":
  main()
