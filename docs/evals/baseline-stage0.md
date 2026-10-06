# Retrieval evaluation: baseline-stage0

- Generated: 2026-10-06T13:00:13+00:00 (git `0ee0a91`)
- Document: `data/evaluation_document.txt`; queries: `data/evaluation_queries.json` (14 queries)
- Chunking: chunk_size=50, overlap=0 -> 3 chunks
- Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (candidate_k=5)

| config | hit@1 | hit@3 | hit@5 | recall@1 | recall@3 | recall@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| dense | 0.571 | 1.000 | 1.000 | 0.571 | 0.964 | 0.964 | 0.774 |
| dense+rerank | 0.929 | 1.000 | 1.000 | 0.929 | 0.964 | 0.964 | 0.964 |

Unreachable queries (a span is not wholly inside any chunk): q12
Note: only 3 chunks exist, so cut-offs >= 3 retrieve the whole document.

## Chunking sweep (context only)

| chunk_size | overlap | chunks | config | hit@1 | hit@3 | recall@3 | MRR | unreachable |
|---|---|---|---|---|---|---|---|---|
| 8 | 0 | 14 | dense | 0.714 | 0.786 | 0.714 | 0.750 | 5 |
| 8 | 0 | 14 | dense+rerank | 0.786 | 0.786 | 0.714 | 0.786 | 5 |
| 8 | 4 | 26 | dense | 0.786 | 0.929 | 0.857 | 0.845 | 2 |
| 8 | 4 | 26 | dense+rerank | 0.857 | 0.929 | 0.857 | 0.893 | 2 |
| 16 | 0 | 7 | dense | 0.429 | 0.786 | 0.750 | 0.571 | 4 |
| 16 | 0 | 7 | dense+rerank | 0.571 | 0.786 | 0.750 | 0.679 | 4 |
| 16 | 4 | 9 | dense | 0.500 | 0.929 | 0.893 | 0.708 | 0 |
| 16 | 4 | 9 | dense+rerank | 0.857 | 1.000 | 1.000 | 0.929 | 0 |
| 32 | 0 | 4 | dense | 0.357 | 0.929 | 0.929 | 0.583 | 1 |
| 32 | 0 | 4 | dense+rerank | 0.786 | 0.929 | 0.929 | 0.857 | 1 |
| 32 | 4 | 4 | dense | 0.429 | 1.000 | 1.000 | 0.655 | 0 |
| 32 | 4 | 4 | dense+rerank | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| 50 | 0 | 3 | dense | 0.571 | 1.000 | 0.964 | 0.774 | 1 |
| 50 | 0 | 3 | dense+rerank | 0.929 | 1.000 | 0.964 | 0.964 | 1 |
| 50 | 4 | 3 | dense | 0.357 | 0.929 | 0.929 | 0.643 | 1 |
| 50 | 4 | 3 | dense+rerank | 0.857 | 0.929 | 0.929 | 0.893 | 1 |
