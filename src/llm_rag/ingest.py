import argparse
import os
import asyncio
import hashlib
import re

import aiohttp
from bs4 import BeautifulSoup
import structlog
from langchain_core._api.beta_decorator import suppress_langchain_beta_warning
from langchain_text_splitters import HTMLSemanticPreservingSplitter
import google.generativeai as genai
import lancedb
from markdownify import MarkdownConverter

from .types import Document, SourceType

# Configure structlog
structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()

def compute_md5(content: str) -> str:
    """Compute MD5 hash of content.
    
    Args:
        content: Content to hash
        
    Returns:
        MD5 hash as hex string
    """
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def init_db() -> lancedb.db.LanceDBConnection:
    """Initialize LanceDB database and create hash table if needed.
    
    Returns:
        LanceDB database instance
    """
    db = lancedb.connect(".lancedb")
    
    # Create hash table if it doesn't exist
    if "content_hashes" not in db.table_names():
        db.create_table(
            "content_hashes",
            data=[{
                "hash": "",
            }],
            mode="overwrite"
        )
    db['content_hashes'].optimize()
    db['content_hashes'].create_scalar_index(column="hash")
    return db

def is_content_new(db: lancedb.db.LanceDBConnection, content_hash: str) -> bool:
    """Check if content hash exists in database.
    
    Args:
        db: LanceDB database instance
        content_hash: MD5 hash to check
        
    Returns:
        True if content is new, False if it already exists
    """
    hash_table = db["content_hashes"]
    results = hash_table.search().where(f"hash = '{content_hash}'").to_list()
    return len(results) == 0

async def process_url(url: str) -> Document:
    """Process a URL and return a Document.
    
    Args:
        url: URL to process
        
    Returns:
        Document containing HTML content and metadata
    """
    # Fetch HTML content
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            html_content = await response.text()
    
    # Parse HTML for metadata
    soup = BeautifulSoup(html_content, 'html.parser')
    metadata = {
        'url': url,
        'title': soup.title.string if soup.title else url,
        'source_type': 'url'
    }
    # Remove unneeded sections
    for data in soup(['style', 'script', 'svg', 'head', 'input', 'meta', 'td', 'footer', 'header']):
        data.decompose()
    for data in soup.find_all(class_=[re.compile(r'sidebar'), re.compile('nav')]):
        data.decompose()
    md = MarkdownConverter().convert_soup(soup)
    logger.info("processing_url", 
                url=url, 
                title=metadata['title'], 
                content_length=len(md))
    
    return Document(
        content=md,  # Use raw HTML
        metadata=metadata,
        source_type=SourceType.URL,
        source_path=url
    )

async def process_source(source: str, source_type: SourceType) -> list[Document]:
    """Process a source and return a list of Documents.
    
    Args:
        source: Source to process (URL or file path)
        source_type: Type of source ('url' or 'file')
        
    Returns:
        List of Documents
    """
    logger.info("processing_source", source=source, source_type=source_type)
    
    if source_type == SourceType.URL:
        doc = await process_url(source)
        
        # Check if content is new (document level)
        db = init_db()
        content_hash = compute_md5(doc.content)
        
        if not is_content_new(db, content_hash):
            logger.info("skipping_duplicate_source", source=source)
            return []
            
        # Add hash to database
        logger.info("calculated_hash", hash=content_hash)
        hash_table = db["content_hashes"]
        hash_table.add([{
            "hash": content_hash,
        }])
        
        return [doc]
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

def chunk_documents(documents: list[Document], chunk_size: int = 1000) -> list[Document]:
    """Chunk documents into smaller pieces using langchain's HTMLSemanticPreservingSplitter.
    
    Args:
        documents: List of documents to chunk
        chunk_size: Maximum chunk size in characters
        
    Returns:
        List of chunked documents
    """
    # HTMLSemanticPreservingSplitter doesn't take chunk_size directly
    # It uses RecursiveCharacterTextSplitter internally for large chunks
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(headers_to_split_on=["h1", "h2"])
    db = init_db()
    hash_table = db["content_hashes"]
    chunked_docs = []
    for doc in documents:
        # Split the HTML document
        chunks = splitter.split_text(doc.content)
        # Create new document for each chunk, preserving metadata
        for i, chunk in enumerate(chunks):
            content_hash = compute_md5(chunk.page_content)
            if not is_content_new(db, content_hash):
                logger.info("skipping_duplicate_chunk", chunk=chunk.page_content)
                continue
                
            # Add hash to database
            hash_table.add([{
                "hash": content_hash,
            }])

            chunked_docs.append(Document(
                content=chunk.page_content,
                metadata={
                    **doc.metadata,
                    'chunk_index': i
                },
                source_type=doc.source_type,
                source_path=doc.source_path
            ))
    
    avg_length = sum(len(doc.content) for doc in chunked_docs) / len(chunked_docs) if chunked_docs else 0
    logger.info("chunked_documents", 
                count=len(chunked_docs), 
                avg_chunk_length=avg_length)
    return chunked_docs

async def embed_single_document(
    doc: Document,
    model: str,
    semaphore: asyncio.Semaphore,
) -> tuple[list[float], dict]:
    """Generate embedding for a single document using Gemini.
    
    Args:
        doc: Document to embed
        model: Gemini model name
        semaphore: Semaphore for controlling concurrency
        
    Returns:
        Tuple of (embedding vector, document metadata)
    """
    async with semaphore:
        result = await genai.embed_content_async(
                model=model,
                content=doc.content,
                task_type="retrieval_document"
            )
        
        return (
            result['embedding'],
            {
                **doc.metadata,
                'content': doc.content  # Include content in metadata for storage
            }
        )

async def embed_documents_async(
    documents: list[Document],
    max_concurrency: int = 50,
) -> list[tuple[list[float], dict]]:
    """Generate embeddings for documents using Gemini asynchronously.
    
    Args:
        documents: List of documents to embed
        max_concurrency: Maximum number of concurrent embedding requests
        
    Returns:
        List of (embedding, metadata) tuples
    """
    # Configure Gemini
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    model = 'models/text-embedding-004'
    
    # Create concurrency controls
    semaphore = asyncio.Semaphore(max_concurrency)
    
    # Create embedding tasks
    tasks = [
        embed_single_document(doc, model, semaphore)
        for doc in documents
    ]
    
    # Run tasks concurrently and gather results
    return await asyncio.gather(*tasks)

async def embed_documents(
    documents: list[Document],
) -> list[tuple[list[float], dict]]:
    """Generate embeddings for documents using Gemini.
    
    Args:
        documents: List of documents to embed
        
    Returns:
        List of (embedding, metadata) tuples
    """
    return await embed_documents_async(documents)

def store_embeddings(
    table_name: str,
    embeddings: list[tuple[list[float], dict]],
) -> None:
    """Store embeddings in LanceDB.
    
    Args:
        table_name: Name of LanceDB table
        embeddings: List of (embedding, metadata) tuples
    """
    # Connect to LanceDB
    db = lancedb.connect(".lancedb")
    
    # Convert to list of dictionaries for LanceDB
    # Flatten metadata into separate columns
    data = [
        {
            "vector": embedding,
            "text": metadata.get("content", ""),  # Use consistent column names
            "url": metadata.get("url", ""),
            "title": metadata.get("title", ""),
            "chunk_index": metadata.get("chunk_index", 0)
        }
        for embedding, metadata in embeddings
    ]

    if len(data) > 0:
        table = db.create_table(table_name, data=data, exist_ok=True) 
        table.add(data)
        logger.info("embeddings_stored", num_documents=len(data))

async def main():
    parser = argparse.ArgumentParser(description="Ingest documents into LanceDB")
    parser.add_argument("--source", required=True, help="Source path or URL")
    parser.add_argument(
        "--type",
        required=True,
        choices=["code", "url", "pdf"],
        help="Type of source",
    )
    parser.add_argument(
        "--table",
        default="documents",
        help="LanceDB table name",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size in characters",
    )
    args = parser.parse_args()

    try:
        # Process source
        documents = await process_source(args.source, args.type)
        
        # Chunk documents
        logger.info("chunking_documents")
        chunked_docs = chunk_documents(documents, args.chunk_size)
        logger.info(f"created_chunks", count=len(chunked_docs))
        
        # Generate embeddings
        logger.info("generating_embeddings")
        embeddings = await embed_documents(chunked_docs)
        
        # Store in LanceDB
        store_embeddings(args.table, embeddings)
        
        logger.info("ingestion_complete")
        
    except Exception as e:
        logger.error(f"error_during_ingestion", error=str(e))
        raise

if __name__ == "__main__":
    asyncio.run(main())
