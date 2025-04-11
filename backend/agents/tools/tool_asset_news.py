import logging
from agents.tools.db.mdb import MongoDBConnector
from agents.tools.vogayeai.vogaye_ai_embeddings import VogayeAIEmbeddings
from agents.tools.states.agent_market_news_state import MarketNewsAgentState, AssetNews
import os
from dotenv import load_dotenv
from typing import List
import random

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
NUMBER_OF_ARTICLES = 3

class AssetNewsTool(MongoDBConnector):
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
        self.sentiment_categories = ["Positive", "Negative", "Neutral"]
        self.embedding_model_id = os.getenv("EMBEDDINGS_MODEL_ID", "voyage-finance-2")
        self.vector_index_name = os.getenv("VECTOR_INDEX_NAME")
        self.vector_field = os.getenv("VECTOR_FIELD")
        logger.info("AssetNewsTool initialized")

            
    def vector_search_articles(self, query: str, n: int) -> dict:
        """Performs a vector search on historical financial news articles in MongoDB Atlas.
        Keep in mind that the collection name is set to "financial_news" in the .env file and its data is fixed.
        This method is used to find relevant articles based on a given query. It utilizes the Voyage AI Embeddings to generate a query vector
        and performs a vector search on the specified collection to find articles that are semantically similar to the query. 
        The results are returned as a dictionary containing the articles and their metadata.
        
        Args:
            query (str): The query string to search for.
            n (int): The number of results to return.

        Returns:
            dict: A dictionary containing the search results.
        """
        message = "[Action] Performing MongoDB Atlas Vector Search for Financial News"
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

            # Remove unnecessary fields from the results
            for result in results:
                if "_id" in result:
                    del result["_id"]
                if "article_embedding" in result:
                    del result["article_embedding"]
                if "article_string" in result:
                    del result["article_string"]
                if "extraction_timestamp_utc" in result:
                    del result["extraction_timestamp_utc"]
                if "synced" in result:
                    del result["synced"]
                if "ticker" in result:
                    del result["ticker"]
                if "posted" in result:
                    del result["posted"]
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            results = []

        # Return the results
        return {
            "articles": results
        }
    

    def sentiment_category_randomizer(self) -> str:
        """Randomly selects a sentiment category from the predefined list."""
        # NOTE: Considering that the financial_news collection data is fixed
        # For demo purposes, we are using a randomizer to select a sentiment category.
        # In a real-world scenario, you would implement your own sentiment analysis logic.
        # This emulates dynamism in the data and the report generation process.
        return random.choice(self.sentiment_categories)


    def transform_articles(self, asset_symbol: str, sentiment_category: str, articles: list) -> List[AssetNews]:
        """Transforms the articles into AssetNews objects.

        Args:
            articles (list): List of articles to transform.

        Returns:
            list: List of AssetNews objects.
        """
        transformed_articles = []
        for article in articles:

            # NOTE: This is simplified for demo purposes
            # Extract the maximum sentiment score value from the three categories
            sentiment_score = None
            if "sentiment_score" in article:
                # Find the maximum value among all sentiment categories
                sentiment_scores = article["sentiment_score"]
                if isinstance(sentiment_scores, dict):
                    max_score = max(sentiment_scores.values())
                    sentiment_score = round(max_score, 2)

            # NOTE: This is an emulation of posted time for demo purposes
            # In a real-world scenario, you would use the actual posted time from the article
            # Generate a random "hours ago" value between 2 and 24
            random_hours = random.randint(2, 24)
            posted_time = f"{random_hours} hours ago"

            # Create an AssetNews object for each article
            asset_news = AssetNews(
                asset=asset_symbol,
                headline=article.get("headline"),
                description=article.get("description"),
                source=article.get("source"),
                posted=posted_time,
                link=article.get("link"),
                sentiment_score=sentiment_score,
                sentiment_category=sentiment_category  # Randomly assign sentiment category
            )
            transformed_articles.append(asset_news)
        return transformed_articles
    

    def fetch_market_news(self, state: MarketNewsAgentState) -> dict:
        """
        Fetches financial news articles related to the assets in the portfolio allocation.
        """
        message = "[Tool] Asset News."
        logger.info(message)

        for allocation in state.portfolio_allocation:  # Iterate over the list of PortfolioAllocation objects
            symbol = allocation.asset  # Access the "asset" field of each PortfolioAllocation object
            description = allocation.description  # Access the "description" field of each PortfolioAllocation object

            # Get sentiment category from the randomizer
            sentiment_category = self.sentiment_category_randomizer()
            symbol_w_description = f"{symbol} ({description})"
            query = f"{sentiment_category} financial news articles related to {symbol_w_description}"
            n = NUMBER_OF_ARTICLES  # Number of articles to retrieve
            # Perform the vector search
            logger.info(f"[Action] Performing MongoDB Atlas Vector Search for {symbol_w_description}")
            results = self.vector_search_articles(query, n)
            # Transform the articles into AssetNews objects
            transformed_articles = self.transform_articles(symbol, sentiment_category, results["articles"])
            # Update the state with the asset trends
            state.report.asset_news.extend(transformed_articles)

        # Append the message to the updates list
        state.updates.append(message)

        # Set the next step in the state
        state.next_step = "asset_news_summary_node"

        return { "asset_news": state.report.asset_news, "updates": state.updates, "next_step": state.next_step }
    


# Initialize the AssetNewsTool
asset_news_obj = AssetNewsTool()

# Define tools
def fetch_market_news_tool(state: MarketNewsAgentState) -> dict:
    """
    Assess the trend of a given symbol by comparing its last closing price with its moving average.
    """
    return asset_news_obj.fetch_market_news(state=state)

# Example usage
if __name__ == "__main__":
    from states.agent_market_news_state import MarketNewsAgentState, PortfolioAllocation

    # Initialize the state with only the fields required for the first step
    state = MarketNewsAgentState(
        portfolio_allocation=[
            PortfolioAllocation(
                asset="SPY", description="S&P 500 ETF", allocation_percentage="25%"
            ),
            PortfolioAllocation(
                asset="QQQ", description="Nasdaq ETF", allocation_percentage="20%"
            )
        ],
        next_step="asset_news_node",  # Set the next step in the workflow
    )

    # Use the tool to calculate asset trends
    trends = fetch_market_news_tool(state)

    # Print the updated state
    print("\nUpdated State:")
    print(state)