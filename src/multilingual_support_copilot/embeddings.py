from sentence_transformers import SentenceTransformer

MODEL_NAME="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

model = SentenceTransformer(MODEL_NAME)

def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(texts)

    return embeddings.tolist()