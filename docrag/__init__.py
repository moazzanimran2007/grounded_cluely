"""
docrag: document context via retrieval (RAG).

This package is deliberately independent of the audio and transcript code. It
takes a single document, splits it into chunks, embeds them, and answers typed
questions grounded in the retrieved passages. Later, the live transcript can be
fed into the same retrieve-and-answer step as just another source.

The steps map one to one onto modules:

  extract.py   step 1, pull raw text out of a PDF or text file
  chunk.py     step 2, split the text into overlapping word windows
  store.py     step 3, embed the chunks and save them to a local JSON index
  retrieve.py  step 4, find the chunks most similar to a question
  answer.py    step 5, ask the model for an answer grounded in those chunks
  cli.py       a small command line so you can run each step on its own
"""
