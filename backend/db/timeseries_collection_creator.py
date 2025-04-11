from pymongo import ASCENDING
from pymongo.errors import CollectionInvalid
from bson.codec_options import CodecOptions
from bson.datetime_ms import DatetimeConversion
from mdb import MongoDBConnector

import logging

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


class TimeSeriesCollectionCreator(MongoDBConnector):
    def __init__(self, uri=None, database_name=None, appname=None):
        super().__init__(uri, database_name, appname)

    def create_timeseries_collection(self, collection_name: str, time_field: str, meta_field: str, granularity: str = "minutes", expire_after_seconds=None):
        """
        Create a time series collection if it doesn't exist.

        Args:
            collection_name (str): Collection name.
            time_field (str): Time field.
            meta_field (str): Meta field.
            granularity (str, optional): Granularity. Defaults to "minutes".
            expire_after_seconds (int, optional): Document expiration time in seconds. Defaults to None.
        """
        codec_options = CodecOptions(
            datetime_conversion=DatetimeConversion.DATETIME_AUTO)

        if collection_name in self.db.list_collection_names():
            logger.info(f"The '{collection_name}' collection already exists.")
            return

        try:
            collection_options = {
                'timeseries': {
                    'timeField': time_field,
                    'metaField': meta_field,
                    'granularity': granularity
                },
                'codec_options': codec_options
            }
            if expire_after_seconds is not None:
                collection_options['expireAfterSeconds'] = expire_after_seconds

            self.db.create_collection(
                collection_name,
                **collection_options
            )
            self.db[collection_name].create_index(
                [(meta_field, ASCENDING), (time_field, ASCENDING)]
            )
            logger.info(
                f"Time series collection '{collection_name}' and index created successfully.")
        except CollectionInvalid:
            logger.error(
                f"Time series collection '{collection_name}' already exists.")
        except Exception as e:
            logger.error(
                f"An error occurred while creating the time series collection: {e}")


if __name__ == "__main__":
    # Example usage
    market_data = TimeSeriesCollectionCreator()
    market_data.create_timeseries_collection(
        collection_name=os.getenv("YFINANCE_TIMESERIES_COLLECTION"),
        time_field="timestamp",
        meta_field="symbol",
        granularity="minutes"
    )