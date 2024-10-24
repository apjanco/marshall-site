import srsly
import chromadb
from chromadb.utils import embedding_functions

def make_index(db_name:str, data_path:str):
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

    chroma_client = chromadb.PersistentClient(path=db_name)
    collection = chroma_client.get_or_create_collection(name=db_name, embedding_function=sentence_transformer_ef)

    data = list(srsly.read_jsonl(data_path))

    documents = [d['text'] for d in data]
    embeddings = sentence_transformer_ef(documents)
    metadatas = data
    ids = [d['image'] for d in data]
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

