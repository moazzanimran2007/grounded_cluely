"""
Step 5: the RAG loop. Answer the question using only the retrieved chunks.

The whole point of grounding is that the model must answer from the passages we
give it, not from whatever it happens to remember. So the prompt does two
things: it hands over the retrieved chunks as the only allowed source, and it
tells the model to say plainly when the answer is not there. That is what stops
it from inventing a confident wrong answer.
"""

from .config import CHAT_MODEL, get_client

# The exact sentence the model should use when the context does not answer the
# question. Keeping it fixed makes the "I do not know" case easy to recognise.
NOT_FOUND = "I could not find that in the document."

GROUNDING_INSTRUCTION = (
    "You answer questions about a single document. Use only the context "
    "passages below. Do not use any outside knowledge. If the answer is not "
    f"contained in the context, reply exactly: {NOT_FOUND}"
)


def build_prompt(question, retrieved):
    """Assemble the grounding instruction, the chunks, and the question."""
    context = "\n\n".join(
        f"[chunk {r['id']}]\n{r['text']}" for r in retrieved
    )
    return (
        f"{GROUNDING_INSTRUCTION}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def answer(question, retrieved):
    """
    Send the question and its retrieved chunks to the model and return the
    grounded answer text. If retrieval came back empty, we do not even call the
    model: there is nothing to ground on, so we return the not found sentence.
    """
    if not retrieved:
        return NOT_FOUND

    prompt = build_prompt(question, retrieved)
    client = get_client()
    resp = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return (resp.text or "").strip()
