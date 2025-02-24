# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi",
#     "pydantic",
#     "uvicorn",
# ]
# ///
import os
import json
import logging

from fastapi import FastAPI, Response, HTTPException, Header, Request, status
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import google.generativeai as genai
import lancedb

# Set up logging
logging.basicConfig(level=logging.INFO)


class TextBlob(BaseModel):
    text: str


class SearchTextRequest(BaseModel):
    text: str
    k: int

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

@app.get("/")
async def root():
    return {"message": "Welcome to TypingMind"}

@app.post("/search_text/")
async def search_text(request: SearchTextRequest):
    text = request.text
    k = request.k
    logging.info(f"Received search_text request with text={text} and k={k}")
    try:
        # Generate embedding for query using Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = 'models/text-embedding-004'
        result = genai.embed_content(model=model, content=text, task_type="retrieval_query")
        query_embedding = result['embedding']

        # Perform vector search using LanceDB
        db = lancedb.connect(".lancedb")
        results = db['documents'].search(query_embedding).limit(k).to_list()

        # Filter results based on score threshold and convert to SearchResult objects
        return [
                dict(
                    content=r['text'],
                    metadata={'title': r['title']},
                    source=r['url']
                )
                for r in results
            ]
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=3033)
