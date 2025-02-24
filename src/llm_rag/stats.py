import argparse
from pathlib import Path
import structlog

import lancedb
from tabulate import tabulate

# Configure structlog
structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()

def get_table_stats(table_name: str) -> None:
    """Get statistics for a LanceDB table.
    
    Args:
        table_name: Name of the table to analyze
    """
    # Connect to LanceDB
    db = lancedb.connect(".lancedb")
    
    if table_name not in db.table_names():
        logger.error("table_not_found", table=table_name)
        return
    
    table = db[table_name]
    
    # Get basic stats
    count = len(table)
    
    # Get disk usage
    table_path = Path(".lancedb") / table_name
    size_bytes = sum(f.stat().st_size for f in table_path.rglob('*') if f.is_file())
    size_kb = size_bytes / 1024
    
    # Get schema
    schema = table.schema
    
    # Get example entry
    example = table.head(n=1).to_pydict() if count > 0 else None
    
    # Print statistics
    logger.info("table_statistics",
                table=table_name,
                total_entries=count,
                size_kb=round(size_kb, 2))
    
    # Print schema
    print("\nSchema:")
    schema_data = [[field.name, field.type] for field in schema]
    print(tabulate(schema_data, headers=['Field', 'Type'], tablefmt='grid'))
    
    # Print example entry
    if example:
        print("\nExample Entry:")
        if 'text' in example:
            print(f"Text:\n{example['text']}")
            del example['text']
        example_data = [[k, str(v)[:100] + '...' if len(str(v)) > 100 else v] 
                       for k, v in example.items()]
        print(tabulate(example_data, headers=['Field', 'Value'], tablefmt='grid'))

def main():
    parser = argparse.ArgumentParser(description="Get LanceDB table statistics")
    parser.add_argument("--table", default="documents",
                       help="Table name to analyze (default: documents)")
    args = parser.parse_args()
    
    get_table_stats(args.table)

if __name__ == "__main__":
    main()
