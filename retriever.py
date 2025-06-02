from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient

embeddings = SentenceTransformerEmbeddings(model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

url = "http://localhost:6333/dashboard"

client = QdrantClient(
    url=url,
    prefer_grpc=False
    )

print(client)

db = Qdrant(
    client=client,
    embeddings=embeddings,
    collection_name="vector_database"
)

print(db)
print("-------------------------------------")
query = "What are the common side effects of systemic therepeutical agents?"

docs = db.similarity_search_with_score(query=query, k=2)

for i in docs:
    doc, score = i
    print({"Score":score, "Content":doc.page_content, "metadata":doc.metadata})

