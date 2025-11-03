import time
import logging
import datetime as dt
from datetime import timezone, datetime, timezone
from pymongo import MongoClient

from scheduler import Scheduler
import pytz

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

class CheckpointerMemoryJobs:
    def __init__(self):
        """
        CheckpointerJobs
        """
        # Load MongoDB configuration from environment variables
        self.mongodb_uri = os.getenv("MONGODB_URI")
        self.database_name = os.getenv("DATABASE_NAME")
        self.checkpoints_collection = os.getenv("CHECKPOINTS_AIO_COLLECTION", "checkpoints_aio")
        self.checkpoint_writes_collection = os.getenv("CHECKPOINTS_WRITES_AIO_COLLECTION", "checkpoint_writes_aio")
        
        # Initialize MongoDB client
        self.mongodb_client = MongoClient(self.mongodb_uri)

        self.utc = pytz.UTC
        self.scheduler = Scheduler(tzinfo=timezone.utc)
        logger.info("CheckpointerJobs initialized")

    def run_clear_old_memories(self):
        """
        Execute a clean up memory records (threads) that are not from the current day.
        """
        logger.info("Cleaning up memory records (threads) that are not from the current day!")
        try:
            deleted_count = self.cleanup_old_memories()
            logger.info(f"Cleaned up {deleted_count} records from previous days")
        except Exception as e:
            logger.error(f"Error cleaning up old memories: {str(e)}")
    
    def cleanup_threads(self, keep_threads=None):
        """
        Clean up old threads from the memory collections.
        
        Args:
            keep_threads (list): List of thread_ids to keep (if None, deletes all threads)
            
        Returns:
            int: Number of deleted documents
        """
        deleted_count = 0
        try:
            # Build the query to delete documents not in the keep_threads list
            query = {} if keep_threads is None else {"thread_id": {"$nin": keep_threads}}
            
            # Delete from the checkpoints collection
            result = self.mongodb_client[self.database_name][self.checkpoints_collection].delete_many(query)
            deleted_count += result.deleted_count
            
            # Delete from the checkpoint writes collection
            result = self.mongodb_client[self.database_name][self.checkpoint_writes_collection].delete_many(query)
            deleted_count += result.deleted_count
            
            return deleted_count
        
        except Exception as e:
            logger.error(f"Error cleaning up old threads: {str(e)}")
            return 0
    
    def cleanup_old_memories(self):
        """
        Clean up memory records (threads) that are not from the current day.
        Identifies thread_ids that don't contain today's date and removes them.
        
        Returns:
            int: Number of deleted documents
        """
        # Get today's date in YYYYMMDD format
        today_date = datetime.now(timezone.utc).strftime('%Y%m%d')
        today_pattern = f"thread_{today_date}_"
        
        try:
            # Get all distinct thread_ids from both collections
            checkpoints_threads = self.mongodb_client[self.database_name][self.checkpoints_collection].distinct("thread_id")
            writes_threads = self.mongodb_client[self.database_name][self.checkpoint_writes_collection].distinct("thread_id")
            # Combine and deduplicate all thread IDs
            all_threads = list(set(checkpoints_threads + writes_threads))
            # Keep only threads from today
            threads_to_keep = [thread for thread in all_threads if today_pattern in thread]
            # Delete all threads not from today
            deleted_count = self.cleanup_threads(keep_threads=threads_to_keep)
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old memories: {str(e)}")
            logger.exception("Full stack trace:")
            return 0
        
    def schedule_jobs(self):
        """
        Schedule jobs
        """
        # Get NODE_ENV to determine the scheduled time
        node_env = os.getenv("NODE_ENV", "").lower()
        
        # Set scheduled time based on environment
        if node_env == "prod":
            run_cleanup_old_memories_time = dt.time(hour=4, minute=5, tzinfo=timezone.utc)
        elif node_env == "staging":
            run_cleanup_old_memories_time = dt.time(hour=4, minute=10, tzinfo=timezone.utc)
        elif node_env == "dev":
            run_cleanup_old_memories_time = dt.time(hour=4, minute=15, tzinfo=timezone.utc)
        else:
            # Default fallback
            run_cleanup_old_memories_time = dt.time(hour=4, minute=20, tzinfo=timezone.utc)
        
        logger.info(f"Scheduling cleanup job at {run_cleanup_old_memories_time} (NODE_ENV={node_env})")
        self.scheduler.daily(run_cleanup_old_memories_time, self.run_clear_old_memories)
        logger.info("Scheduled jobs configured!")

    def start(self):
        """
        Starts the scheduler.
        """
        self.schedule_jobs()
        logger.info("Schedule Jobs overview:")
        logger.info(self.scheduler)
        while True:
            self.scheduler.exec_jobs()
            time.sleep(1)
