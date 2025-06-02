from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

print(embeddings)

loader = DirectoryLoader("Data/",glob="*.pdf", show_progress=True, loader_cls=PyPDFLoader)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

texts = text_splitter.split_documents(documents)

url = "http://localhost:6333/dashboard"

qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="vector_database",
)

print("Vector Database Created Successfully")