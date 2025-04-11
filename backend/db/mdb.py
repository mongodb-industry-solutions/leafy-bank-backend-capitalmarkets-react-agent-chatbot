import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class MongoDBConnector:
    """ A class to provide access to a MongoDB database.
    This class handles the connection to the database and provides methods to interact with collections and documents.

    Attributes:
        uri (str): The connection string URI for the MongoDB database.
        database_name (str): The name of the database to connect to.
        appname (str): The name of the application connecting to the database.
    """

    def __init__(self, uri=None, database_name=None, appname=None):
        """ Initialize the MongoDBConnector instance. """
        self.uri = uri or os.getenv("MONGODB_URI")
        self.database_name = database_name or os.getenv("DATABASE_NAME")
        self.appname = appname or os.getenv("APP_NAME")
        self.client = MongoClient(self.uri, appname=self.appname)
        self.db = self.client[self.database_name]

    def get_collection(self, collection_name):
        """Retrieve a collection."""
        if not collection_name:
            raise ValueError("Collection name must be provided.")
        return self.db[collection_name]

    def insert_one(self, collection_name, document):
        """Insert a single document into a collection."""
        collection = self.get_collection(collection_name)
        result = collection.insert_one(document)
        return result.inserted_id

    def insert_many(self, collection_name, documents):
        """Insert multiple documents into a collection."""
        collection = self.get_collection(collection_name)
        result = collection.insert_many(documents)
        return result.inserted_ids

    def find(self, collection_name, query={}, projection=None):
        """Retrieve documents from a collection."""
        collection = self.get_collection(collection_name)
        return list(collection.find(query, projection))

    def update_one(self, collection_name, query, update, upsert=False):
        """Update a single document in a collection."""
        collection = self.get_collection(collection_name)
        result = collection.update_one(query, update, upsert=upsert)
        return result.modified_count

    def update_many(self, collection_name, query, update, upsert=False):
        """Update multiple documents in a collection."""
        collection = self.get_collection(collection_name)
        result = collection.update_many(query, update, upsert=upsert)
        return result.modified_count

    def delete_one(self, collection_name, query):
        """Delete a single document from a collection."""
        collection = self.get_collection(collection_name)
        result = collection.delete_one(query)
        return result.deleted_count

    def delete_many(self, collection_name, query):
        """Delete multiple documents from a collection."""
        collection = self.get_collection(collection_name)
        result = collection.delete_many(query)
        return result.deleted_count