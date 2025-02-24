from typing import Optional
import argparse
import asyncio

import lancedb
import google.generativeai as genai
from mcp import server

from .types import SearchResult

class SearchServer:
    def __init__(self, db_path: str, table_name: str):
        """Initialize search server.
        
        Args:
            db_path: Path to LanceDB database
            table_name: Name of table containing embeddings
        """
        self.db = lancedb.connect(db_path)
        self.table = self.db[table_name]

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search for documents matching query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        # Generate embedding for query using Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = 'models/text-embedding-004'
        result = genai.embed_content(model=model, content=query, task_type="retrieval_query")
        query_embedding = result['embedding']

        # Perform vector search using LanceDB
        results = self.table.search(query_embedding).limit(limit).to_list()

        # Filter results based on score threshold and convert to SearchResult objects
        search_results = [
            SearchResult(
                content=r['text'],
                metadata={'url': r['url'], 'title': r['title']},
                score=r['_distance'],
                source=r['url']
            )
            for r in results
        ]

        return search_results

async def handle_request(request: Request, server: SearchServer) -> Response:
    """Handle incoming search requests."""
    try:
        query = request.json.get("query")
        if not query:
            return Response({"error": "Missing query"}, status=400)

        limit = request.json.get("limit", 10)

        results = await server.search(query, limit)
        return Response({"results": [r.model_dump() for r in results]})
    except Exception as e:
        return Response({"error": str(e)}, status=500)

def main():
    parser = argparse.ArgumentParser(description="Run search server")
    parser.add_argument("--db", default=".lancedb", help="Path to LanceDB database")
    parser.add_argument(
        "--table",
        default="documents",
        help="LanceDB table name",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind server to",
    )
    args = parser.parse_args()

    # Initialize search server
    server = SearchServer(args.db, args.table)

    # Start MCP server
    mcp_server = server.Server()
    mcp_server.handle("/search", lambda req: handle_request(req, server))
    asyncio.run(mcp_server.serve(args.host, args.port))

if __name__ == "__main__":
    main()
