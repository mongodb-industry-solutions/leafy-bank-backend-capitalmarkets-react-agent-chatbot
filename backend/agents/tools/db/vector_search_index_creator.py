import logging
from pymongo.errors import OperationFailure

from mdb import MongoDBConnector

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class VectorSearchIndexCreator(MongoDBConnector):
    def __init__(self, collection_name: str, uri=None, database_name: str = None, appname: str = None):
        """ VectorSearchIndexCreator class to create a vector search index in MongoDB. If it already exists, it will not be created again."""
        super().__init__(uri, database_name, appname)
        self.collection_name = collection_name
        self.collection = self.get_collection(self.collection_name)
        logger.info("VectorSearchIndexCreator initialized")

    def create_index(self, index_name: str, vector_field: str, dimensions: int = 1024, similarity_metric: str = "cosine") -> dict:
        """
        Creates a vector search index on the MongoDB collection.

        Args:
            index_name (str, optional): Index name.
            vector_field (str, optional): Vector field name.
            dimensions (int, optional): Number of dimensions. Default is 1024.
            similarity_metric (str, optional): Similarity metric. Default is "cosine".

        Returns:
            dict: Index creation result
        """
        logger.info(f"Creating vector search index...")
        logger.info(f"Collection: {self.collection_name}")
        logger.info(f"Vector Field: {vector_field}")
        logger.info(f"Dimensions: {dimensions}")
        logger.info(f"Similarity Metric: {similarity_metric}")

        # Define the vector search index configuration
        index_config = {
            "name": index_name,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "path": vector_field,
                        "type": "vector",
                        "numDimensions": dimensions,
                        "similarity": similarity_metric
                    }
                ]
            }
        }

        try:
            # Create the index
            self.collection.create_search_index(index_config)
            logger.info(f"Vector search index '{index_name}' created successfully.")
            return {"status": "success", "message": f"Vector search index '{index_name}' created successfully."}
        except OperationFailure as e:
            if e.code == 68:  # IndexAlreadyExists error code
                logger.warning(f"Vector search index '{index_name}' already exists.")
                return {"status": "warning", "message": f"Vector search index '{index_name}' already exists."}
            else:
                logger.error(f"Error creating vector search index: {e}")
                return {"status": "error", "message": f"Error creating vector search index: {e}"}
        except Exception as e:
            logger.error(f"Error creating vector search index: {e}")
            return {"status": "error", "message": f"Error creating vector search index: {e}"}


# Example usage
if __name__ == "__main__":
    market_analysis_collection_name = os.getenv("REPORTS_COLLECTION_MARKET_ANALYSIS")
    market_news_collection_name = os.getenv("REPORTS_COLLECTION_MARKET_NEWS")
    market_analysis_vector_index_name = os.getenv("REPORT_MARKET_ANALISYS_VECTOR_INDEX_NAME")
    market_news_vector_index_name = os.getenv("REPORT_MARKET_NEWS_VECTOR_INDEX_NAME")
    report_vector_field = os.getenv("REPORT_VECTOR_FIELD")

    # Create vector search index for market analysis
    vs_idx = VectorSearchIndexCreator(collection_name=market_analysis_collection_name)
    result = vs_idx.create_index(
        index_name=market_analysis_vector_index_name,
        vector_field=report_vector_field
    )
    logger.info(result)

    # Create vector search index for market news
    vs_idx = VectorSearchIndexCreator(collection_name=market_news_collection_name)
    result = vs_idx.create_index(
        index_name=market_news_vector_index_name,
        vector_field=report_vector_field
    )
    logger.info(result)