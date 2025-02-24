# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi",
#     "langchain",
#     "langchain-community",
#     "langchain-openai",
#     "numpy",
#     "pydantic",
#     "uvicorn",
# ]
# ///
from fastapi import FastAPI, Response, HTTPException, Header, Request, status
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)


class TextBlob(BaseModel):
    text: str


class SearchTextRequest(BaseModel):
    text: str
    k: int


# Initialize OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "https://typingmind.com",
    "typingmind.com",
    "http://typingmind.com",
    "https://www.typingmind.com",
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "k", "text"],  # Include custom headers here
)


@app.options("/search_text/")
async def search_text_options(response: Response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )


@app.post("/add_texts/")
async def add_texts(body: TextBlob, custom_header: str = Header(None)):
    text = body.text
    folder_path = "./faiss_index"
    print(f"text: {text}")
    texts = text_splitter.split_text(text)

    # Load existing vector store from disk if it exists, else initialize a new FAISS vector store
    if os.path.exists(folder_path):
        faiss = FAISS.load_local(folder_path=folder_path, embeddings=embeddings)
    else:
        faiss = FAISS(
            docstore=None,
            embedding_function=lambda x: x,
            index=None,
            index_to_docstore_id=embeddings,
        )  # Initialize FAISS with embeddings

    # Add texts (either to the newly created or the loaded FAISS vector store)
    faiss.add_texts(texts=texts)

    faiss.save_local(folder_path=folder_path)

    return {"message": "Text added successfully", "custom_header": custom_header}


@app.post("/search_text/")
async def search_text(request: SearchTextRequest):
    text = request.text
    k = request.k
    logging.info(f"Received search_text request with text={text} and k={k}")
    try:
        faiss = FAISS.load_local(folder_path="./faiss_index", embeddings=embeddings)
        retriever = faiss.as_retriever(
            search_type="mmr", search_kwargs={"k": k, "lambda_mult": 0.25}
        )

        results = retriever.get_relevant_documents(text)
        return {"matches": results}
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=3033)
