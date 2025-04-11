import os
import logging
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from agents.tools.vogayeai.vogaye_ai_embeddings import VogayeAIEmbeddings
from agents.tools.db.mdb import MongoDBConnector

# Initialize dotenv to load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize search tool
tavily_search_tool = TavilySearchResults(max_results=3)

# Initialize embeddings
embedding_model_id = os.getenv("EMBEDDINGS_MODEL_ID", "voyage-finance-2")
ve = VogayeAIEmbeddings(api_key=os.getenv("VOYAGE_API_KEY"))

# Initialize MongoDB collections for reports
MARKET_COLLECTION_NAME = os.getenv("REPORTS_COLLECTION_MARKET_ANALYSIS", "reports_market_analysis")
NEWS_COLLECTION_NAME = os.getenv("REPORTS_COLLECTION_MARKET_NEWS", "reports_market_news")

# Initialize MongoDB connector
mongodb_connector = MongoDBConnector()

# Get the collections for market and news reports
market_reports_collection = mongodb_connector.get_collection(collection_name=MARKET_COLLECTION_NAME)
news_reports_collection = mongodb_connector.get_collection(collection_name=NEWS_COLLECTION_NAME)

# Getting environment variables for vector index names
REPORT_MARKET_ANALISYS_VECTOR_INDEX_NAME = os.getenv("REPORT_MARKET_ANALISYS_VECTOR_INDEX_NAME")
REPORT_MARKET_NEWS_VECTOR_INDEX_NAME = os.getenv("REPORT_MARKET_NEWS_VECTOR_INDEX_NAME")

# Getting environment variables for vector field names
REPORT_VECTOR_FIELD = os.getenv("REPORT_VECTOR_FIELD", "report_embedding")


@tool
def market_analysis_reports_vector_search_tool(query: str, k: int = 1):
    """
    Perform a vector similarity search on market analysis reports for the CURRENT PORTFOLIO.

    IMPORTANT: This tool provides market analysis ONLY for assets included in the current portfolio allocation.  
    If someone requests real-time data or live updates for assets outside the current portfolio, use the Tavily Search tool instead.

    Use this tool when you need:
    - Market trends and analysis for portfolio assets
    - Insights on recent portfolio performance
    - Macroeconomic factors affecting the current portfolio
    - Asset-specific diagnostics for portfolio holdings

    Args:
        query (str): The search query related to portfolio assets, market trends, etc.
        k (int, optional): The number of top results to return. Defaults to 1.

    Returns:
        dict: Contains relevant sections from the most recent market analysis report
              for the current portfolio.
    """
    try:
        logger.info(f"Searching portfolio market analysis for: {query}")
        
        # Get the most recent document for context information
        most_recent = market_reports_collection.find_one(
            {}, 
            sort=[("timestamp", -1)]
        )
        
        if not most_recent:
            return {"results": "No market analysis reports found for the current portfolio."}
        
        # Extract the date of the most recent report
        report_date = most_recent.get("date_string", "Unknown date")
        
        # Get portfolio assets list for context
        portfolio_assets = []
        try:
            for allocation in most_recent.get("portfolio_allocation", []):
                asset = allocation.get("asset", "Unknown")
                description = allocation.get("description", "")
                allocation_pct = allocation.get("allocation_percentage", "")
                portfolio_assets.append(f"{asset} ({description}): {allocation_pct}")
        except Exception as e:
            logger.error(f"Error extracting portfolio information: {e}")

        # Generate query embedding
        query_embedding = ve.get_embeddings(model_id=embedding_model_id, text=query)
        
        # Perform vector search across all documents
        pipeline = [
            {
                "$vectorSearch": {
                    "index": f"{REPORT_MARKET_ANALISYS_VECTOR_INDEX_NAME}",
                    "path": REPORT_VECTOR_FIELD,
                    "queryVector": query_embedding,
                    "numCandidates": 5,
                    "limit": k + 3  # Get more candidates for re-ranking
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "date_string": 1,
                    "report": 1,
                    "timestamp": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(market_reports_collection.aggregate(pipeline))
        
        # If no results from vector search, just return the most recent document
        if not results:
            # Return most recent document info
            overall_diagnosis = most_recent.get("report", {}).get("overall_diagnosis", "No diagnosis available")
            asset_trends = most_recent.get("report", {}).get("asset_trends", [])
            asset_insights = []
            
            for trend in asset_trends:
                asset = trend.get("asset", "Unknown")
                diagnosis = trend.get("diagnosis", "No diagnosis")
                asset_insights.append(f"{asset}: {diagnosis}")
            
            return {
                "report_date": report_date,
                "portfolio_assets": portfolio_assets,
                "overall_diagnosis": overall_diagnosis,
                "asset_insights": asset_insights,
                "note": "This is the most recent market analysis for your portfolio."
            }
        
        # Re-rank results by combining vector similarity score with recency
        import datetime
        now = datetime.datetime.utcnow()
        
        # Make sure most recent is always in the results
        most_recent_in_results = False
        most_recent_id_str = str(most_recent.get("_id", ""))
        
        for result in results:
            # Check if this is the most recent document
            if result.get("date_string") == most_recent.get("date_string"):
                most_recent_in_results = True
                
            # Calculate time difference in days (newer = higher score)
            result_timestamp = result.get("timestamp", now)
            if isinstance(result_timestamp, str):
                try:
                    result_timestamp = datetime.datetime.fromisoformat(result_timestamp.replace('Z', '+00:00'))
                except:
                    result_timestamp = now
                    
            days_old = (now - result_timestamp).total_seconds() / (24 * 3600) if hasattr(result_timestamp, 'total_seconds') else 30
            
            # Calculate recency score (1.0 for current, approaches 0 as it gets older)
            recency_score = 1.0 / (1.0 + days_old)
            
            # Get vector similarity score (normalize it to 0-1 range)
            similarity_score = result.get("score", 0.0)
            
            # Combined score with weights (adjust weights as needed)
            vector_weight = 0.3  # Reduce this to give less weight to semantic similarity
            recency_weight = 0.7  # Increase this to give more weight to recency
            
            # Calculate combined score
            combined_score = (vector_weight * similarity_score) + (recency_weight * recency_score)
            result["combined_score"] = combined_score
        
        # If most recent isn't in results, add it
        if not most_recent_in_results:
            most_recent["combined_score"] = 1.0  # Give it a high score
            most_recent["score"] = 0.5  # Neutral semantic score
            results.append(most_recent)
        
        # Sort results by combined score
        results = sorted(results, key=lambda x: x.get("combined_score", 0.0), reverse=True)
        
        # Process best result (highest combined score)
        report_data = results[0]
        overall_diagnosis = report_data.get("report", {}).get("overall_diagnosis", "No diagnosis available")
        
        # Format the return data
        return {
            "report_date": report_date,
            "portfolio_assets": portfolio_assets[:5],  # Show top 5 portfolio assets
            "overall_diagnosis": overall_diagnosis,
            "note": "This information is from the most relevant and recent portfolio market analysis."
        }
        
    except Exception as e:
        logger.error(f"Error searching portfolio market reports: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}
    
@tool
def market_news_reports_vector_search_tool(query: str, k: int = 1):
    """
    Perform a vector similarity search on market news reports for the CURRENT PORTFOLIO.

    IMPORTANT: This tool provides market news summaries and insights ONLY for assets included in the current portfolio allocation.  
    Note that it DOES NOT offer real-time data or live updates.  

    When possible, include links to the original news articles at the end of the summary for reference.  
    If someone requests real-time data or information on assets not in the current portfolio, use the Tavily Search tool instead.

    Use this tool when you need:
    - Recent news affecting portfolio assets
    - Sentiment analysis for portfolio holdings
    - News summaries for specific assets in the portfolio
    - An overview of the news impact on the current portfolio

    Args:
        query (str): The search query related to portfolio assets.
        k (int, optional): The number of top results to return. Defaults to 1.

    Returns:
        dict: Contains relevant news summaries from the most recent reports
              for the current portfolio.
    """
    try:
        logger.info(f"Searching portfolio news reports for: {query}")
        
        # Get the most recent document for context information only
        most_recent = news_reports_collection.find_one(
            {}, 
            sort=[("timestamp", -1)]
        )
        
        if not most_recent:
            return {"results": "No news reports found for the current portfolio."}
        
        # Generate query embedding
        query_embedding = ve.get_embeddings(model_id=embedding_model_id, text=query)
        
        # Extract the date of the most recent report
        report_date = most_recent.get("date_string", "Unknown date")
        
        # Get portfolio assets list for context
        portfolio_assets = []
        try:
            for allocation in most_recent.get("portfolio_allocation", []):
                asset = allocation.get("asset", "Unknown")
                description = allocation.get("description", "")
                allocation_pct = allocation.get("allocation_percentage", "")
                portfolio_assets.append(f"{asset} ({description}): {allocation_pct}")
        except Exception as e:
            logger.error(f"Error extracting portfolio information: {e}")
        
        # Perform vector search across all documents
        pipeline = [
            {
                "$vectorSearch": {
                    "index": f"{REPORT_MARKET_NEWS_VECTOR_INDEX_NAME}",
                    "path": REPORT_VECTOR_FIELD,
                    "queryVector": query_embedding,
                    "numCandidates": 5,
                    "limit": k + 3  # Get more candidates for re-ranking
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "date_string": 1,
                    "report": 1,
                    "timestamp": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(news_reports_collection.aggregate(pipeline))
        
        # If no results from vector search, just return the most recent document
        if not results:
            # Extract relevant information
            overall_diagnosis = most_recent.get("report", {}).get("overall_news_diagnosis", "No news diagnosis available")
            asset_news_summaries = most_recent.get("report", {}).get("asset_news_summary", [])
            
            # Format asset news summary information
            news_summaries = []
            for summary in asset_news_summaries:
                asset = summary.get("asset", "Unknown")
                summary_text = summary.get("summary", "No summary available")
                sentiment = summary.get("overall_sentiment_category", "Unknown")
                news_summaries.append(f"{asset} ({sentiment}): {summary_text}")
            
            return {
                "report_date": report_date,
                "portfolio_assets": portfolio_assets,
                "overall_diagnosis": overall_diagnosis,
                "news_summaries": news_summaries,
                "note": "This is the most recent news report for your portfolio."
            }
        
        # Re-rank results by combining vector similarity score with recency
        import datetime
        now = datetime.datetime.utcnow()
        
        # Make sure most recent is always in the results
        most_recent_in_results = False
        most_recent_id_str = str(most_recent.get("_id", ""))
        
        for result in results:
            # Check if this is the most recent document
            if result.get("date_string") == most_recent.get("date_string"):
                most_recent_in_results = True
                
            # Calculate time difference in days (newer = higher score)
            result_timestamp = result.get("timestamp", now)
            if isinstance(result_timestamp, str):
                try:
                    result_timestamp = datetime.datetime.fromisoformat(result_timestamp.replace('Z', '+00:00'))
                except:
                    result_timestamp = now
                    
            days_old = (now - result_timestamp).total_seconds() / (24 * 3600) if hasattr(result_timestamp, 'total_seconds') else 30
            
            # Calculate recency score (1.0 for current, approaches 0 as it gets older)
            recency_score = 1.0 / (1.0 + days_old)
            
            # Get vector similarity score (normalize it to 0-1 range)
            similarity_score = result.get("score", 0.0)
            
            # Combined score with weights (adjust weights as needed)
            vector_weight = 0.3  # Reduce this to give less weight to semantic similarity
            recency_weight = 0.7  # Increase this to give more weight to recency
            
            # Calculate combined score
            combined_score = (vector_weight * similarity_score) + (recency_weight * recency_score)
            result["combined_score"] = combined_score
        
        # If most recent isn't in results, add it
        if not most_recent_in_results:
            most_recent["combined_score"] = 1.0  # Give it a high score
            most_recent["score"] = 0.5  # Neutral semantic score
            results.append(most_recent)
        
        # Sort results by combined score
        results = sorted(results, key=lambda x: x.get("combined_score", 0.0), reverse=True)
        
        # Process best result (highest combined score)
        report_data = results[0]
        report = report_data.get("report", {})
        overall_diagnosis = report.get("overall_news_diagnosis", "No news diagnosis available")
        asset_news_summaries = report.get("asset_news_summary", [])
        
        # Format news summaries
        news_summaries = []
        for summary in asset_news_summaries:
            asset = summary.get("asset", "Unknown")
            summary_text = summary.get("summary", "No summary available")
            sentiment = summary.get("overall_sentiment_category", "Unknown")
            news_summaries.append(f"{asset} ({sentiment}): {summary_text}")
        
        # Format the return data
        return {
            "report_date": report_date,
            "portfolio_assets": portfolio_assets[:5],  # Show top 5 portfolio assets
            "overall_diagnosis": overall_diagnosis,
            "news_summaries": news_summaries,
            "note": "This information is from the most relevant and recent portfolio news reports."
        }
        
    except Exception as e:
        logger.error(f"Error searching portfolio news reports: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}