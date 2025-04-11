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

class MarketDataService(MongoDBConnector):
    def __init__(self, uri=None, database_name: str = None, appname: str = None, collection_name: str = os.getenv("YFINANCE_TIMESERIES_COLLECTION")):
        """
        Service for manipulating Yahoo Finance market data.

        Args:
            uri (str, optional): MongoDB URI. Defaults to None.
            database_name (str, optional): Database name. Defaults to None.
            appname (str, optional): Application name. Defaults to None.
            collection_name (str, optional): Collection name. Defaults to "yfinanceMarketData".
        """
        super().__init__(uri, database_name, appname)
        self.collection_name = collection_name
        logger.info("MarketDataService initialized")

    def fetch_assets_close_price(self):
        """
        Get the latest close price for all assets.

        Returns:
            dict: A dictionary containing the assets and their close prices.
        """
        try:
            # The aggregation pipeline is used here to efficiently query and process the data within MongoDB.
            # This approach is optimal because:
            # 1. It reduces the amount of data transferred over the network by performing the computation on the server side.
            # 2. It leverages MongoDB's optimized aggregation framework, which is designed for high performance and scalability.
            # 3. It simplifies the code by allowing us to express complex data transformations in a declarative manner.

            # The pipeline consists of two stages:
            # 1. $sort: Sorts the documents by the "timestamp" field in descending order.
            # 2. $group: Groups the documents by the "symbol" field and selects the first document in each group,
            # which corresponds to the latest document due to the previous sorting stage.
            pipeline = [
                {
                    "$sort": {"timestamp": -1}
                },
                {
                    "$group": {
                        "_id": "$symbol",
                        "latest_close_price": {"$first": "$close"},
                        "latest_timestamp": {"$first": "$timestamp"}
                    }
                }
            ]

            # Execute the aggregation pipeline
            result = self.db[self.collection_name].aggregate(pipeline)

            # Process the result and construct the close_prices dictionary
            close_prices = {}
            for doc in result:
                symbol = doc["_id"]
                close_price = doc["latest_close_price"]
                timestamp = doc["latest_timestamp"]
                close_prices[symbol] = {
                    "close_price": close_price,
                    "timestamp": timestamp
                }

            logger.info(f"Retrieved close prices for {len(close_prices)} assets")
            return close_prices
        except Exception as e:
            logger.error(f"Error retrieving close prices: {e}")
            return {}

if __name__ == "__main__":
    # Example usage
    market_data_service = MarketDataService()
    close_prices = market_data_service.fetch_assets_close_price()
    for symbol, data in close_prices.items():
        print(f"Symbol: {symbol}, Close Price: {data['close_price']}, Timestamp: {data['timestamp']}")