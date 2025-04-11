import logging
from db.mdb import MongoDBConnector
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

class FinancialNewsDataService(MongoDBConnector):
    def __init__(self, uri=None, database_name: str = None, appname: str = None, collection_name: str = os.getenv("NEWS_COLLECTION")):
        """
        Service for manipulating Financial News data.

        Args:
            uri (str, optional): MongoDB URI. Defaults to None.
            database_name (str, optional): Database name. Defaults to None.
            appname (str, optional): Application name. Defaults to None.
            collection_name (str, optional): Collection name. Defaults to "financial_news".
        """
        super().__init__(uri, database_name, appname)
        self.collection_name = collection_name
        logger.info("FinancialNewsDataService initialized")

    def calc_overall_sentiment_for_all(self):
        """
        Calculate the overall sentiment score for all symbols.

        Returns:
            dict: A dictionary containing the symbol, overall sentiment score, category, and number of articles.
        """
        try:
            # The aggregation pipeline is used here to efficiently query and process the data within MongoDB.
            # This approach is optimal because:
            # 1. It reduces the amount of data transferred over the network by performing the computation on the server side.
            # 2. It leverages MongoDB's optimized aggregation framework, which is designed for high performance and scalability.
            # 3. It simplifies the code by allowing us to express complex data transformations in a declarative manner.

            # The pipeline consists of three stages:
            # 1. $group: Groups the documents by the "ticker" field and calculates the average sentiment scores and the count of articles.
            # 2. $project: Projects the required fields and calculates the overall sentiment score.
            # 3. $addFields: Adds the "category" field based on the overall sentiment score.
            pipeline = [
                {
                    "$group": {
                        "_id": "$ticker",
                        "avg_positive": {"$avg": "$sentiment_score.positive"},
                        "avg_negative": {"$avg": "$sentiment_score.negative"},
                        "avg_neutral": {"$avg": "$sentiment_score.neutral"},
                        "article_count": {"$sum": 1}
                    }
                },
                {
                    "$project": {
                        "symbol": "$_id",
                        "overall_sentiment_score": {
                            "$subtract": [
                                {"$add": ["$avg_positive", {
                                    "$multiply": ["$avg_neutral", 0.5]}]},
                                "$avg_negative"
                            ]
                        },
                        "article_count": 1
                    }
                },
                {
                    "$addFields": {
                        "category": {
                            "$switch": {
                                "branches": [
                                    {"case": {
                                        "$gte": ["$overall_sentiment_score", 0.6]}, "then": "Positive"},
                                    {"case": {
                                        "$gte": ["$overall_sentiment_score", 0.4]}, "then": "Neutral"}
                                ],
                                "default": "Negative"
                            }
                        }
                    }
                }
            ]

            # Execute the aggregation pipeline
            result = self.db[self.collection_name].aggregate(pipeline)

            # Process the result and construct the sentiment_scores dictionary
            sentiment_scores = {}
            for doc in result:
                symbol = doc["symbol"]
                sentiment_scores[symbol] = {
                    "overall_sentiment_score": doc["overall_sentiment_score"],
                    "category": doc["category"],
                    "article_count": doc["article_count"]
                }

            logger.info(
                f"Calculated overall sentiment for {len(sentiment_scores)} symbols")
            return sentiment_scores
        except Exception as e:
            logger.error(f"Error calculating overall sentiment: {e}")
            return {}

    def calc_overall_sentiment_for_symbol(self, symbol: str):
        """
        Calculate the overall sentiment score for a specific symbol.

        Args:
            symbol (str): The symbol for which to calculate the overall sentiment score.

        Returns:
            dict: A dictionary containing the symbol, overall sentiment score, category, and number of articles.
        """
        try:
            # The aggregation pipeline is used here to efficiently query and process the data within MongoDB.
            # This approach is optimal because:
            # 1. It reduces the amount of data transferred over the network by performing the computation on the server side.
            # 2. It leverages MongoDB's optimized aggregation framework, which is designed for high performance and scalability.
            # 3. It simplifies the code by allowing us to express complex data transformations in a declarative manner.

            # The pipeline consists of three stages:
            # 1. $match: Filters the documents to include only those with the specified symbol.
            # 2. $group: Groups the documents by the "ticker" field and calculates the average sentiment scores and the count of articles.
            # 3. $project: Projects the required fields and calculates the overall sentiment score.
            # 4. $addFields: Adds the "category" field based on the overall sentiment score.
            pipeline = [
                {
                    "$match": {"ticker": symbol}
                },
                {
                    "$group": {
                        "_id": "$ticker",
                        "avg_positive": {"$avg": "$sentiment_score.positive"},
                        "avg_negative": {"$avg": "$sentiment_score.negative"},
                        "avg_neutral": {"$avg": "$sentiment_score.neutral"},
                        "article_count": {"$sum": 1}
                    }
                },
                {
                    "$project": {
                        "symbol": "$_id",
                        "overall_sentiment_score": {
                            "$subtract": [
                                {"$add": ["$avg_positive", {
                                    "$multiply": ["$avg_neutral", 0.5]}]},
                                "$avg_negative"
                            ]
                        },
                        "article_count": 1
                    }
                },
                {
                    "$addFields": {
                        "category": {
                            "$switch": {
                                "branches": [
                                    {"case": {
                                        "$gte": ["$overall_sentiment_score", 0.6]}, "then": "Positive"},
                                    {"case": {
                                        "$gte": ["$overall_sentiment_score", 0.4]}, "then": "Neutral"}
                                ],
                                "default": "Negative"
                            }
                        }
                    }
                }
            ]

            # Execute the aggregation pipeline
            result = self.db[self.collection_name].aggregate(pipeline)

            # Process the result and construct the sentiment_score dictionary
            sentiment_score = {}
            for doc in result:
                sentiment_score = {
                    "symbol": doc["symbol"],
                    "overall_sentiment_score": doc["overall_sentiment_score"],
                    "category": doc["category"],
                    "article_count": doc["article_count"]
                }

            logger.info(
                f"Calculated overall sentiment for symbol {symbol}")
            return sentiment_score
        except Exception as e:
            logger.error(f"Error calculating overall sentiment for symbol {symbol}: {e}")
            return {}

if __name__ == "__main__":
    # Example usage
    financial_news_data_service = FinancialNewsDataService()
    sentiment_scores = financial_news_data_service.calc_overall_sentiment_for_all()
    for symbol, data in sentiment_scores.items():
        print(
            f"Symbol: {symbol}, Overall Sentiment Score: {data['overall_sentiment_score']}, Category: {data['category']}, Article Count: {data['article_count']}")

    # Example usage for a specific symbol
    symbol_sentiment = financial_news_data_service.calc_overall_sentiment_for_symbol("EEM")
    print(
        f"Symbol: {symbol_sentiment['symbol']}, Overall Sentiment Score: {symbol_sentiment['overall_sentiment_score']}, Category: {symbol_sentiment['category']}, Article Count: {symbol_sentiment['article_count']}")