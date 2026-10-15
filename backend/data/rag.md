# Retrieval-augmented generation

Retrieval-augmented generation (RAG) answers a question in two steps. First a retriever finds the passages in a document collection that are most relevant to the question. Then a language model writes the answer using only those passages, and cites them.

RAG lets a model answer from private or recent documents it was never trained on, and it makes answers checkable, because every claim points to a source passage.

## Chunking

Documents are split into chunks before indexing. Chunks that are too large dilute the relevant sentence with unrelated text; chunks that are too small lose the context needed to understand them. A common starting point is 200 to 500 tokens per chunk with a small overlap, so a sentence cut at a boundary still appears whole in one chunk.

## Keyword and vector retrieval

BM25 is a keyword ranking function. It scores a chunk by how often the question's terms appear in it, weighted by how rare each term is across the collection. It needs no model and is strong on exact names, codes and rare terms.

Vector retrieval embeds the question and every chunk with an embedding model and ranks chunks by similarity. It finds passages that mean the same thing in different words.

Hybrid retrieval runs both and merges the rankings, often with reciprocal rank fusion. A reranker, usually a cross-encoder model, can then reorder the merged list by reading the question and each passage together.

## Evaluating a RAG system

Retrieval and generation fail in different ways, so they are measured separately. Retrieval metrics ask whether the right passages were found, for example recall at k. Generation metrics ask whether the answer is faithful to the retrieved passages and whether it actually answers the question.
