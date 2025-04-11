import logging
from db.mdb import MongoDBConnector
from vogayeai.vogaye_ai_embeddings import VogayeAIEmbeddings
import os
from dotenv import load_dotenv
from bson import ObjectId

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class VectorSearchService(MongoDBConnector):
    def __init__(self, uri=None, database_name: str = None, appname: str = None, collection_name: str = os.getenv("NEWS_COLLECTION")):
        """
        Service for performing vector search operations on Financial News data.

        Args:
            uri (str, optional): MongoDB URI. Defaults to None.
            database_name (str, optional): Database name. Defaults to None.
            appname (str, optional): Application name. Defaults to None.
            collection_name (str, optional): Collection name. Defaults to "financial_news".
        """
        super().__init__(uri, database_name, appname)
        self.collection_name = collection_name
        self.collection = self.get_collection(collection_name)
        self.embedding_model_id = os.getenv("EMBEDDINGS_MODEL_ID", "voyage-finance-2")
        self.vector_index_name = os.getenv("VECTOR_INDEX_NAME")
        self.vector_field = os.getenv("VECTOR_FIELD")
        logger.info("VectorSearchService initialized")
        
    def semantic_similarity_search(self, query: str, n: int) -> dict:
        """Performs a vector search on past issues in MongoDB Atlas.
        
        Args:
            query (str): The query string to search for.
            n (int): The number of results to return.

        Returns:
            dict: A dictionary containing the search results.
        """
        message = "[Tool] Performing MongoDB Atlas Vector Search"
        print("\n" + message)

        logger.info(f"Query: {query}")
        # Generate query vector using Voyage AI Embeddings
        ve = VogayeAIEmbeddings(api_key=os.getenv("VOYAGE_API_KEY"))
        query_vector = ve.get_embeddings(model_id=self.embedding_model_id, text=query)

        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.vector_index_name,
                        "path": self.vector_field,
                        "queryVector": query_vector,
                        "numCandidates": max(n * 3, 5),
                        "limit": n
                    }
                }
            ]
            results = list(self.collection.aggregate(pipeline))
            for result in results:
                if "_id" in result:
                    result["_id"] = str(result["_id"])
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            results = []

        # Return the results
        return {
            "articles": results
        }

    def get_all_article_ids(self):
        """Get all article IDs in the collection."""
        cursor = self.collection.find({}, {"_id": 1})
        return [str(doc["_id"]) for doc in cursor]


# Example usage
if __name__ == "__main__":
    # Initialize service
    vector_search_service = VectorSearchService()
    
    # Configuration
    ARTICLES_PER_ASSET = 50
    OUTPUT_FILE = "retained_article_ids.txt"
    DELETION_BATCH_SIZE = 100
    
    # Portfolio assets
    portfolio_assets = [
        "SPY (S&P 500 ETF)",
        "QQQ (Nasdaq ETF)",
        "EEM (Emerging Markets ETF)",
        "XLE (Energy Sector ETF)",
        "TLT (Long-Term Treasury Bonds)",
        "LQD (Investment-Grade Bonds)",
        "HYG (High-Yield Bonds)",
        "VNQ (Real Estate ETF)",
        "GLD (Gold ETF)",
        "USO (Oil ETF)",
    ]
    
    print("\n=== FINANCIAL NEWS DATABASE CLEANUP: ASSET-FOCUSED RETENTION ===")
    print(f"This tool will retain up to {ARTICLES_PER_ASSET} articles for each asset in your portfolio.")
    print(f"All other articles will be removed.")
    
    # Track all articles to keep (by ID)
    articles_to_keep = set()
    
    # Track headlines to avoid duplicates
    seen_headlines = set()
    
    # For each asset, find and keep relevant articles
    for asset in portfolio_assets:
        print(f"\n\nSearching for articles about: {asset}")
        
        # Create a more specific query for better results
        query = f"Financial articles and analysis specifically about {asset} performance, price movements, and investment insights"
        
        # Retrieve more articles than we need to account for duplicates
        search_size = ARTICLES_PER_ASSET * 3
        results = vector_search_service.semantic_similarity_search(query, search_size)
        
        # Track articles kept for this asset
        asset_articles_kept = 0
        asset_articles_seen = 0
        
        # Show sample of articles found
        print(f"\nFound {len(results['articles'])} potentially relevant articles.")
        
        # Process and deduplicate
        for article in results['articles']:
            headline = article.get('headline', '')
            asset_articles_seen += 1
            
            # Skip duplicates
            if headline in seen_headlines or headline == "":
                continue
            
            # Add to our collections
            articles_to_keep.add(article['_id'])
            seen_headlines.add(headline)
            asset_articles_kept += 1
            
            # Display some examples
            if asset_articles_kept <= 3:
                print(f"\n--- Sample {asset_articles_kept} ---")
                print(f"Headline: {headline}")
                print(f"Description: {article.get('description', 'No description')[:200]}...")
            
            # Stop if we have enough articles for this asset
            if asset_articles_kept >= ARTICLES_PER_ASSET:
                break
        
        print(f"Keeping {asset_articles_kept} unique articles about {asset}")
    
    # Get all article IDs
    all_article_ids = vector_search_service.get_all_article_ids()
    
    # Determine which articles to delete
    articles_to_delete = [id for id in all_article_ids if id not in articles_to_keep]
    
    # Final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total articles in database: {len(all_article_ids)}")
    print(f"Articles to keep: {len(articles_to_keep)}")
    print(f"Articles to delete: {len(articles_to_delete)}")
    
    # Save the IDs of articles we're keeping as backup
    with open(OUTPUT_FILE, "w") as f:
        for article_id in articles_to_keep:
            f.write(f"{article_id}\n")
    print(f"Retained article IDs saved to {OUTPUT_FILE}")
    
    # Final confirmation before deletion
    if articles_to_delete:
        final_confirmation = input(f"Are you ABSOLUTELY SURE you want to delete {len(articles_to_delete)} articles and keep only the most relevant {len(articles_to_keep)}? (type 'DELETE' to confirm): ")
        
        if final_confirmation == 'DELETE':
            # Delete in batches to avoid timeout
            total_deleted = 0
            total_batches = (len(articles_to_delete) + DELETION_BATCH_SIZE - 1) // DELETION_BATCH_SIZE
            
            for i in range(0, len(articles_to_delete), DELETION_BATCH_SIZE):
                batch = articles_to_delete[i:i+DELETION_BATCH_SIZE]
                object_ids = [ObjectId(id_str) for id_str in batch]
                result = vector_search_service.collection.delete_many({"_id": {"$in": object_ids}})
                total_deleted += result.deleted_count
                print(f"Deleted batch {i//DELETION_BATCH_SIZE + 1}/{total_batches}: {result.deleted_count} articles")
            
            print(f"\nCleanup complete! Retained {len(articles_to_keep)} relevant articles and removed {total_deleted} non-relevant articles.")
        else:
            print("Deletion cancelled.")
    else:
        print("No articles to delete.")