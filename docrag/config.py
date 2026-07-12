"""
One place for the knobs and the API key, so the other modules stay focused on
their one job. Nothing here talks to the network at import time. The Gemini
client is only built when something actually needs to embed or answer.
"""

import os

from dotenv import load_dotenv

# Load GEMINI_API_KEY from the .env file into the environment.
load_dotenv()

# Models. Both are on Google's free tier.
#   gemini-embedding-001 turns text into an embedding vector. This is the
#   embedding model the current API exposes (the older text-embedding-004 name
#   returns a 404 on this API version).
#   gemini-2.5-flash is a fast, cheap chat model for the final answer. It is the
#   flash model the free tier actually allows on this key (the 2.0 flash models
#   came back with a zero free quota).
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"

# Chunking. A few hundred words per chunk, with a small overlap so an idea that
# straddles a boundary still lands whole in at least one chunk.
CHUNK_WORDS = 250
CHUNK_OVERLAP_WORDS = 40

# How many chunks to retrieve per question.
TOP_K = 4

# Where the local vector index files live. One JSON file per source document.
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")


def require_gemini_key(env=None):
    """
    Return the Gemini API key, or stop with a clear message if it is missing.
    Same pattern as the audio code: fail loudly and helpfully, not deep in a
    network call.
    """
    if env is None:
        env = os.environ
    key = env.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit(
            "No GEMINI_API_KEY found. Get a free key at "
            "https://aistudio.google.com/apikey and add it to .env as "
            "GEMINI_API_KEY=your_key."
        )
    return key


def get_client():
    """
    Build a Gemini client. Imported here, not at the top, so that the pure steps
    (extract, chunk) and their tests do not need the SDK or a key to run.
    """
    from google import genai

    return genai.Client(api_key=require_gemini_key())
