"""
Seed script for the Capital Markets Market Assistant.

Loads every collection under backend/db/collections/ into MongoDB Atlas. The
files are mongoexport-format Extended JSON (each _id is {"$oid": ...}, each
date is {"$date": ...}), so they're parsed with bson.json_util.loads rather
than the stdlib json module -- plain json.load would leave _id as a nested
object instead of a real ObjectId.

This does not create the three vector search indexes; run
backend/agent/db/vector_search_index_creator.py separately afterward (it
needs the collections to exist first).

Run once, from the backend/ directory:
    poetry run python db/seed_database.py

Safe to run again: any collection that already has documents is skipped.
"""
import logging
import os
from pathlib import Path

from bson import json_util
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
APP_NAME = os.getenv("APP_NAME")

COLLECTIONS_DIR = Path(__file__).parent / "collections"


def collection_name_from_filename(path: Path) -> str:
    """agentic_capital_markets.agent_profiles.json -> agent_profiles

    Falls back to the bare stem if the file wasn't exported with the
    <database>.<collection>.json naming convention mongoexport/Compass use.
    """
    stem = path.stem
    parts = stem.split(".", 1)
    return parts[1] if len(parts) == 2 else stem


def seed():
    if not MONGODB_URI or not DATABASE_NAME:
        raise SystemExit("MONGODB_URI and DATABASE_NAME must be set (see backend/.env.example)")

    client = MongoClient(MONGODB_URI, appname=APP_NAME)
    db = client[DATABASE_NAME]

    files = sorted(COLLECTIONS_DIR.glob("*.json"))
    if not files:
        logger.warning(f"No seed files found in {COLLECTIONS_DIR}")
        return

    for path in files:
        coll_name = collection_name_from_filename(path)
        existing = db[coll_name].count_documents({})
        if existing:
            logger.info(f"[skip] {coll_name}: already has {existing} document(s)")
            continue

        docs = json_util.loads(path.read_text())
        if not isinstance(docs, list):
            docs = [docs]
        if not docs:
            logger.info(f"[skip] {coll_name}: {path.name} is empty")
            continue

        result = db[coll_name].insert_many(docs)
        logger.info(f"[seed] {coll_name}: inserted {len(result.inserted_ids)} document(s) from {path.name}")

    logger.info("\n--- final counts ---")
    for path in files:
        coll_name = collection_name_from_filename(path)
        logger.info(f"{coll_name}: {db[coll_name].count_documents({})} docs")

    client.close()


if __name__ == "__main__":
    seed()
