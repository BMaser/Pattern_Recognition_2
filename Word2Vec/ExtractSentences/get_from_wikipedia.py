#!/usr/bin/python3 -u

import wikipedia, sys

wikipedia.set_lang("en")
while True:
  try:
    content = wikipedia.page(wikipedia.random()).content
    print(content)
  except wikipedia.exceptions.WikipediaException as e:
    print(e, file=sys.stderr)
