
# customer_support_platform.py

from llama_cpp import Llama  # or your preferred LLaMA integration
import chromadb
from chromadb.config import Settings
from openai import OpenAIEmbeddings

# Initialize ChromaDB
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_storage"))
collection = chroma_client.get_or_create_collection(name="support_docs")

# Initialize OpenAI Embeddings
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

# Load LLaMA model (adjust based on actual integration)
llm = Llama(model_path="./llama_model.bin")

def embed_text(text):
    return embedder.embed_query(text)

def add_documents(docs):
    for i, doc in enumerate(docs):
        embedding = embed_text(doc["content"])
        collection.add(
            documents=[doc["content"]],
            metadatas=[doc.get("metadata", {})],
            ids=[f"doc_{i}"],
            embeddings=[embedding]
        )

def retrieve_relevant_docs(query, k=3):
    query_vec = embed_text(query)
    results = collection.query(query_embeddings=[query_vec], n_results=k)
    return [doc for doc in results["documents"][0]]

def generate_answer(query):
    relevant_docs = retrieve_relevant_docs(query)
    context = "\n".join(relevant_docs)
    prompt = f"Answer the following question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = llm(prompt)
    return response["choices"][0]["text"]

# Example usage
if __name__ == "__main__":
    docs = [
        {"content": "To reset your password, go to Settings and click on 'Forgot Password'.", "metadata": {"category": "account"}},
        {"content": "You can contact support via email or live chat.", "metadata": {"category": "support"}}
    ]
    add_documents(docs)
    user_query = "How do I reset my password?"
    answer = generate_answer(user_query)
    print("Answer:", answer)
