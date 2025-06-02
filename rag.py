from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import CTransformers
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
import os
import json

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

local_llm = "openhermes-2-mistral-7b.Q4_K_M.gguf"

config = {
'max_new_tokens': 1024,
'context_length': 2048,
'repetition_penalty': 1.1,
'temperature': 0.1,
'top_k': 50,
'top_p': 0.9,
'stream': True,
'threads': int(os.cpu_count() / 2)
}

llm = CTransformers(
    model=local_llm,
    model_type="llama",
    lib="avx2",
    **config
)

print("LLM Initialized....")

prompt_template = """You are a helpful medical assistant. Use the following pieces of information to provide a detailed and accurate answer to the user's question.
If you don't know the answer or if the context doesn't contain relevant information, clearly state that you don't have enough information to provide a complete answer.

Context: {context}
Question: {question}

Please provide a comprehensive answer that:
1. Directly addresses the question
2. Uses the provided context
3. Is well-structured and easy to understand
4. Includes relevant details from the context

Answer:
"""

embeddings = SentenceTransformerEmbeddings(model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

url = "http://localhost:6333" 

client = QdrantClient(
    url=url, prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_database")

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

retriever = db.as_retriever(search_kwargs={"k":1})

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )
    
    response = qa(query)
    print(response)
    
    answer = response.get('result', "No answer returned by model.")

    # Handle no source documents
    if not response.get("source_documents"):
        response_data = jsonable_encoder(json.dumps({
            "answer": "Sorry, I couldn't find any relevant information in the documents.",
            "source_document": [],
            "doc": "N/A"
        }))
        return Response(response_data)

    # If documents exist, structure response
    source_document = [
        {"content": doc.page_content, "source": doc.metadata.get('source', 'Unknown')}
        for doc in response['source_documents']
    ]

    doc = response['source_documents'][0].metadata.get('source', 'Unknown')

    response_data = jsonable_encoder(json.dumps({
        "answer": answer,
        "source_document": source_document,
        "doc": doc
    }))

    return Response(response_data)
