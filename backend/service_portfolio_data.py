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

class PortfolioDataService(MongoDBConnector):
    def __init__(self, uri=None, database_name: str = None, appname: str = None, collection_name: str = os.getenv("PORTFOLIO_COLLECTION")):
        """
        Service for manipulating Portfolio Allocation data.

        Args:
            uri (str, optional): MongoDB URI. Defaults to None.
            database_name (str, optional): Database name. Defaults to None.
            appname (str, optional): Application name. Defaults to None.
            collection_name (str, optional): Collection name. Defaults to "portfolio_allocation".
        """
        super().__init__(uri, database_name, appname)
        self.collection_name = collection_name
        logger.info("PortfolioDataService initialized")

    def fetch_portfolio_allocation(self):
        """
        Get portfolio allocation data.

        Returns:
            dict: A dictionary containing the portfolio allocation data.
        """
        try:
            # Query to get all portfolio allocation data
            result = self.db[self.collection_name].find()

            # Process the result and construct the portfolio_allocation dictionary
            portfolio_allocation = {}
            for doc in result:
                symbol = doc["symbol"]
                allocation_data = {
                    "allocation_percentage": doc["allocation_percentage"],
                    "allocation_number": doc["allocation_number"],
                    "allocation_decimal": doc["allocation_decimal"],
                    "description": doc["description"]
                }
                portfolio_allocation[symbol] = allocation_data

            logger.info(f"Retrieved portfolio allocation for {len(portfolio_allocation)} assets")
            return portfolio_allocation
        except Exception as e:
            logger.error(f"Error retrieving portfolio allocation: {e}")
            return {}

if __name__ == "__main__":
    # Example usage
    portfolio_data_service = PortfolioDataService()
    portfolio_allocation = portfolio_data_service.fetch_portfolio_allocation()
    for symbol, data in portfolio_allocation.items():
        print(f"Symbol: {symbol}, Allocation: {data['allocation_percentage']}, Description: {data['description']}")