import os
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

from agents.tools.db.mdb import MongoDBConnector
from agents.tools.vogayeai.vogaye_ai_embeddings import VogayeAIEmbeddings
from agents.tools.states.agent_market_analysis_state import MarketAnalysisAgentState
from agents.tools.states.agent_market_news_state import MarketNewsAgentState

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class PersistReportInMongoDB(MongoDBConnector):
    def __init__(self, collection_name: str, uri=None, database_name=None):
        """
        Initialize the PersistReportInMongoDB class.
        This class is responsible for persisting data to a MongoDB collection. It inherits from the MongoDBConnector class.

        Args:
            collection_name (str): Collection name to be used in MongoDB
            uri (_type_, optional): MongoDB URI. Defaults to None. If None, it will take the value from parent class.
            database_name (_type_, optional): Database Name. Defaults to None. If None, it will take the value from parent class.
        """
        super().__init__(uri, database_name)
        self.collection_name = collection_name
        self.collection = self.get_collection(self.collection_name)
        self.embedding_model_id = os.getenv("EMBEDDINGS_MODEL_ID", "voyage-finance-2")
        self.ve = VogayeAIEmbeddings(api_key=os.getenv("VOYAGE_API_KEY"))
        logger.info(f"PersistReportInMongoDB initialized with collection: {self.collection_name}")

    def generate_news_report_embeddings(self, report):
        """
        Generate embeddings for a news report using VogayeAI.
        
        Args:
            report (dict): The news report data
            
        Returns:
            list: The embeddings vector
        """
        try:
            # Create a consolidated summary text for embedding
            summary_texts = []
            
            # Add asset news summaries - these contain the condensed insights
            if "asset_news_summary" in report:
                for summary in report.get("asset_news_summary", []):
                    asset = summary.get("asset", "")
                    summary_text = summary.get("summary", "")
                    category = summary.get("overall_sentiment_category", "")
                    score = summary.get("overall_sentiment_score", "")
                    summary_texts.append(f"{asset} ({category}, score: {score}): {summary_text}")
            
            # Add individual news items from asset_news, including only the specified fields
            if "asset_news" in report:
                for news_item in report.get("asset_news", []):
                    # Include only the requested fields (excluding posted, sentiment_score, and sentiment_category)
                    asset = news_item.get("asset", "")
                    headline = news_item.get("headline", "")
                    description = news_item.get("description", "")
                    source = news_item.get("source", "")
                    link = news_item.get("link", "")
                    
                    # Format the news item text with emphasis on the link
                    news_text = f"{asset} - {headline}: {description} (Source: {source}, Link: {link})"
                    summary_texts.append(news_text)
            
            # Add overall news diagnosis - this provides the big picture assessment
            if "overall_news_diagnosis" in report:
                summary_texts.append(f"OVERALL: {report['overall_news_diagnosis']}")
                    
            # Join all summary texts
            text_to_embed = " ".join(summary_texts)
            
            # Generate embeddings
            logger.info(f"Generating news embeddings using model: {self.embedding_model_id}")
            logger.info(f"Embedding content length: {len(text_to_embed)} characters")
            embeddings = self.ve.get_embeddings(model_id=self.embedding_model_id, text=text_to_embed)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating news embeddings: {e}")
            return None
    
    def generate_market_report_embeddings(self, report):
        """
        Generate embeddings for a market analysis report using VogayeAI.
        
        Args:
            report (dict): The market analysis report data
            
        Returns:
            list: The embeddings vector
        """
        try:
            # Use overall_diagnosis as the primary text for embedding
            if "overall_diagnosis" in report:
                text_to_embed = report.get("overall_diagnosis", "")
                logger.info("Using overall_diagnosis for market report embeddings")
            else:
                # Fallback to asset trends and other data if overall_diagnosis isn't available
                logger.info("overall_diagnosis not found, using asset trends as fallback")
                text_parts = []
                
                # Add asset trends diagnoses
                if "asset_trends" in report:
                    for trend in report.get("asset_trends", []):
                        asset = trend.get("asset", "")
                        diagnosis = trend.get("diagnosis", "")
                        text_parts.append(f"{asset}: {diagnosis}")
                
                # Add macro indicators
                if "macro_indicators" in report:
                    for indicator in report.get("macro_indicators", []):
                        macro = indicator.get("macro_indicator", "")
                        diagnosis = indicator.get("diagnosis", "")
                        text_parts.append(f"{macro}: {diagnosis}")
                
                # Add volatility index
                if "market_volatility_index" in report:
                    volatility = report.get("market_volatility_index", {})
                    index = volatility.get("volatility_index", "")
                    diagnosis = volatility.get("diagnosis", "")
                    text_parts.append(f"{index}: {diagnosis}")
                    
                # Join all parts
                text_to_embed = " ".join(text_parts)
            
            # Generate embeddings
            logger.info(f"Generating market embeddings using model: {self.embedding_model_id}")
            logger.info(f"Embedding content length: {len(text_to_embed)} characters")
            embeddings = self.ve.get_embeddings(model_id=self.embedding_model_id, text=text_to_embed)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating market embeddings: {e}")
            return None

    def clean_old_reports(self):
        """
        Maintain a rolling 30-day window of reports by removing older documents.
        This ensures the collection only contains the most recent 30 days of reports.
        Uses timestamp field for proper chronological sorting.
        """
        try:
            # Count the total number of documents in the collection
            total_documents = self.collection.count_documents({})
            logger.info(f"Total documents in {self.collection_name}: {total_documents}")
            
            # If we have more than 30 documents, remove the oldest ones
            if total_documents > 30:
                logger.info(f"Collection {self.collection_name} has {total_documents} documents. Cleaning to keep only the latest 30.")
                
                # Get the 30 most recent documents, sorted by timestamp (not date_string)
                recent_docs = self.collection.find({}, {"_id": 1, "timestamp": 1}).sort("timestamp", -1).limit(30)
                recent_ids = [doc["_id"] for doc in recent_docs]
                
                # Delete all documents that aren't in our list of 30 most recent
                if len(recent_ids) == 30:
                    delete_result = self.collection.delete_many({"_id": {"$nin": recent_ids}})
                    logger.info(f"Removed {delete_result.deleted_count} documents, keeping the 30 most recent by timestamp")
                else:
                    logger.warning(f"Expected 30 recent documents but found {len(recent_ids)}. Skipping cleanup.")
            else:
                logger.info(f"Collection {self.collection_name} has {total_documents} documents. No cleanup needed.")
                    
        except Exception as e:
            logger.error(f"Error cleaning old reports: {e}")

    def save_market_analysis_report(self, final_state):
        """
        Save the market analysis report to the MongoDB collection.
        This method takes the final state of the workflow, prepares the report data, and inserts it into the MongoDB collection.
        If a report for the current date already exists, it will be skipped entirely to save API tokens.
        After saving, maintains a rolling 30-day window of reports.

        Args:
            final_state: The final state of the workflow containing the report data.
        """
        try:
            # Get current date in UTC
            current_date = datetime.now(timezone.utc)
            date_string = current_date.strftime("%Y%m%d")  # Date in "YYYYMMDD" format
            
            # Check if a report for the current date already exists
            existing_report = self.collection.find_one({"date_string": date_string})
            
            if existing_report:
                # Skip the entire operation if a report already exists for today
                logger.info(f"Report for date {date_string} already exists. Skipping to save API tokens.")
                # Clean old reports to maintain only the latest 30 days
                self.clean_old_reports()
                return
            
            # Only proceed if no report exists for today
            logger.info("Saving market analysis report to MongoDB...")

            # Convert the final_state to a MarketAnalysisAgentState object if necessary
            if not isinstance(final_state, MarketAnalysisAgentState):
                final_state = MarketAnalysisAgentState.model_validate(final_state)

            # Prepare the report data
            report = final_state.report.model_dump()
            report_data = {
                "portfolio_allocation": [allocation.model_dump() for allocation in final_state.portfolio_allocation],
                "report": report,
                "updates": final_state.updates,
                "timestamp": current_date,
                "date_string": date_string
            }
            
            # Generate embeddings for the new report
            logger.info(f"Generating embeddings for new market report dated {date_string}")
            report_data["report_embedding"] = self.generate_market_report_embeddings(report)
            
            # Insert a new report
            self.collection.insert_one(report_data)
            logger.info(f"New market report for date {date_string} saved to MongoDB.")
            
            # Clean old reports to maintain only the latest 30 days
            self.clean_old_reports()
                
        except Exception as e:
            logger.error(f"Error saving market report to MongoDB: {e}")

    def save_market_news_report(self, final_state):
        """
        Save the market news report to the MongoDB collection.
        This method takes the final state of the workflow, prepares the report data, and inserts it into the MongoDB collection.
        If a report for the current date already exists, it will be skipped entirely to save API tokens.
        After saving, maintains a rolling 30-day window of reports.

        Args:
            final_state: The final state of the workflow containing the news report data.
        """
        try:
            # Get current date in UTC
            current_date = datetime.now(timezone.utc)
            date_string = current_date.strftime("%Y%m%d")  # Date in "YYYYMMDD" format
            
            # Check if a report for the current date already exists
            existing_report = self.collection.find_one({"date_string": date_string})
            
            if existing_report:
                # Skip the entire operation if a report already exists for today
                logger.info(f"News report for date {date_string} already exists. Skipping to save API tokens.")
                # Clean old reports to maintain only the latest 30 days
                self.clean_old_reports()
                return
            
            # Only proceed if no report exists for today
            logger.info("Saving market news report to MongoDB...")

            # Convert the final_state to a MarketNewsAgentState object if necessary
            if not isinstance(final_state, MarketNewsAgentState):
                final_state = MarketNewsAgentState.model_validate(final_state)

            # Prepare the report data
            report = final_state.report.model_dump()
            report_data = {
                "portfolio_allocation": [allocation.model_dump() for allocation in final_state.portfolio_allocation],
                "report": report,
                "updates": final_state.updates,
                "timestamp": current_date,
                "date_string": date_string
            }
            
            # Generate embeddings for the new report
            logger.info(f"Generating embeddings for new news report dated {date_string}")
            report_data["report_embedding"] = self.generate_news_report_embeddings(report)
            
            # Insert a new report
            self.collection.insert_one(report_data)
            logger.info(f"New news report for date {date_string} saved to MongoDB.")
            
            # Clean old reports to maintain only the latest 30 days
            self.clean_old_reports()
                
        except Exception as e:
            logger.error(f"Error saving news report to MongoDB: {e}")