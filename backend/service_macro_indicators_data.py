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

class MacroIndicatorDataService(MongoDBConnector):
    def __init__(self, uri=None, database_name: str = None, appname: str = None, collection_name: str = os.getenv("PYFREDAPI_COLLECTION")):
        """
        Service for manipulating macroeconomic indicators data.

        Args:
            uri (str, optional): MongoDB URI. Defaults to None.
            database_name (str, optional): Database name. Defaults to None.
            appname (str, optional): Application name. Defaults to None.
            collection_name (str, optional): Collection name. Defaults to "pyfredapiMacroeconomicIndicators".
        """
        super().__init__(uri, database_name, appname)
        self.collection_name = collection_name
        logger.info("MacroIndicatorDataService initialized")

    def fetch_most_recent_macro_indicators(self):
        """
        Get the most recent macroeconomic indicators.

        Returns:
            dict: A dictionary containing the most recent macroeconomic indicators.
        """
        try:
            # The aggregation pipeline is used here to efficiently query and process the data within MongoDB.
            # This approach is optimal because:
            # 1. It reduces the amount of data transferred over the network by performing the computation on the server side.
            # 2. It leverages MongoDB's optimized aggregation framework, which is designed for high performance and scalability.
            # 3. It simplifies the code by allowing us to express complex data transformations in a declarative manner.

            # The pipeline consists of two stages:
            # 1. $sort: Sorts the documents by the "date" field in descending order.
            # 2. $group: Groups the documents by the "series_id" field and selects the first document in each group,
            #            which corresponds to the most recent document due to the previous sorting stage.
            pipeline = [
                {
                    "$sort": {"date": -1}
                },
                {
                    "$group": {
                        "_id": "$series_id",
                        "title": {"$first": "$title"},
                        "frequency": {"$first": "$frequency"},
                        "frequency_short": {"$first": "$frequency_short"},
                        "units": {"$first": "$units"},
                        "units_short": {"$first": "$units_short"},
                        "date": {"$first": "$date"},
                        "value": {"$first": "$value"}
                    }
                }
            ]

            # Execute the aggregation pipeline
            result = self.db[self.collection_name].aggregate(pipeline)

            # Process the result and construct the macro_indicators dictionary
            macro_indicators = {}
            for doc in result:
                series_id = doc["_id"]
                macro_indicators[series_id] = {
                    "title": doc["title"],
                    "frequency": doc["frequency"],
                    "frequency_short": doc["frequency_short"],
                    "units": doc["units"],
                    "units_short": doc["units_short"],
                    "date": doc["date"],
                    "value": doc["value"]
                }

            logger.info(f"Retrieved most recent macro indicators for {len(macro_indicators)} series")
            return macro_indicators
        except Exception as e:
            logger.error(f"Error retrieving most recent macro indicators: {e}")
            return {}

if __name__ == "__main__":
    # Example usage
    macro_indicator_data_service = MacroIndicatorDataService()
    most_recent_macro_indicators = macro_indicator_data_service.fetch_most_recent_macro_indicators()
    for series_id, data in most_recent_macro_indicators.items():
        print(f"Series ID: {series_id}, Title: {data['title']}, Date: {data['date']}, Value: {data['value']}")