import os
import logging
import datetime
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_tavily import TavilySearch
from agent.vogayeai.vogaye_ai_embeddings import VogayeAIEmbeddings
from agent.db.mdb import MongoDBConnector

# Initialize dotenv to load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize search tool
tavily_search_tool = TavilySearch(max_results=3)

# Initialize embeddings
embedding_model_id = os.getenv("EMBEDDINGS_MODEL_ID", "voyage-finance-2")
ve = VogayeAIEmbeddings(api_key=os.getenv("VOYAGE_API_KEY"))

# Getting environment variables for MongoDB collections
MARKET_ANALYSIS_COLLECTION_NAME = os.getenv("REPORTS_COLLECTION_MARKET_ANALYSIS", "reports_market_analysis")
MARKET_NEWS_COLLECTION_NAME = os.getenv("REPORTS_COLLECTION_MARKET_NEWS", "reports_market_news")
MARKET_SM_COLLECTION_NAME = os.getenv("REPORTS_COLLECTION_MARKET_SM", "reports_market_sm")
PORTFOLIO_ALLOCATION_COLLECTION_NAME = os.getenv("PORTFOLIO_ALLOCATION_COLLECTION", "portfolio_allocation")
PORTFOLIO_PERFORMANCE_COLLECTION_NAME = os.getenv("PORTFOLIO_PERFORMANCE_COLLECTION", "portfolio_performance")

# Initialize MongoDB connector
mongodb_connector = MongoDBConnector()

# Get the collections for market and news reports
market_reports_collection = mongodb_connector.get_collection(collection_name=MARKET_ANALYSIS_COLLECTION_NAME)
news_reports_collection = mongodb_connector.get_collection(collection_name=MARKET_NEWS_COLLECTION_NAME)
market_sm_collection = mongodb_connector.get_collection(collection_name=MARKET_SM_COLLECTION_NAME)

# Get portfolio allocation and performance collections
portfolio_allocation_collection = mongodb_connector.get_collection(collection_name=PORTFOLIO_ALLOCATION_COLLECTION_NAME)
portfolio_performance_collection = mongodb_connector.get_collection(PORTFOLIO_PERFORMANCE_COLLECTION_NAME)

# Getting environment variables for vector index names
REPORT_MARKET_ANALISYS_VECTOR_INDEX_NAME = os.getenv("REPORT_MARKET_ANALISYS_VECTOR_INDEX_NAME")
REPORT_MARKET_NEWS_VECTOR_INDEX_NAME = os.getenv("REPORT_MARKET_NEWS_VECTOR_INDEX_NAME")
REPORT_MARKET_SM_VECTOR_INDEX_NAME = os.getenv("REPORT_MARKET_SM_VECTOR_INDEX_NAME")

# Getting environment variables for vector field names
REPORT_VECTOR_FIELD = os.getenv("REPORT_VECTOR_FIELD", "report_embedding")


@tool
def market_analysis_reports_vector_search_tool(query: str, k: int = 1):
    """
    Perform a vector similarity search on market analysis reports for the CURRENT PORTFOLIO.

    IMPORTANT: This tool should typically be used AFTER get_portfolio_allocation_tool
    to provide context-aware analysis for the user's actual holdings.

    COMMON USAGE PATTERNS:
    - After get_portfolio_allocation_tool for reallocation advice
    - After get_portfolio_allocation_tool for technical analysis questions
    - Part of comprehensive market analysis (with news and social tools)

    Use this tool when you need:
    - Market trends and momentum analysis for portfolio assets
    - Technical indicators for traditional assets (moving averages)
    - Market conditions to inform reallocation decisions
    - Asset-specific diagnostics for portfolio holdings
    - Market volatility analysis for traditional assets
    - Macroeconomic factors affecting current portfolio

    Args:
        query (str): The search query related to portfolio assets, market trends, insights, etc.
        k (int, optional): The number of top results to return. Defaults to 1.

    Returns:
        dict: Contains relevant sections from the most recent market analysis report for the current portfolio.
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
        now = datetime.datetime.now(datetime.timezone.utc)
        
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

    IMPORTANT: This tool should typically be used AFTER get_portfolio_allocation_tool.
    For sentiment questions, use this WITH market_social_media_reports_vector_search_tool.

    COMMON USAGE PATTERNS:
    - After get_portfolio_allocation_tool for sentiment analysis
    - Combined with market_social_media_reports_vector_search_tool for complete sentiment picture
    - Part of comprehensive market analysis

    Use this tool when you need:
    - Recent market news affecting portfolio assets
    - News sentiment analysis for portfolio holdings  
    - Media coverage impact on portfolio assets
    - News-driven market sentiment for traditional assets

    Args:
        query (str): The search query related to portfolio assets.
        k (int, optional): The number of top results to return. Defaults to 1.

    Returns:
        dict: Contains relevant news summaries from the most recent market news reports for the current portfolio.
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

@tool
def market_social_media_reports_vector_search_tool(query: str, k: int = 1):
    """
    Perform a vector similarity search on market social media reports for the CURRENT PORTFOLIO.

    IMPORTANT: This tool should typically be used AFTER get_portfolio_allocation_tool.
    For sentiment questions, use this WITH market_news_reports_vector_search_tool.

    COMMON USAGE PATTERNS:
    - After get_portfolio_allocation_tool for sentiment analysis
    - Combined with market_news_reports_vector_search_tool for complete sentiment picture
    - Part of comprehensive market analysis

    Use this tool when you need:
    - Social media sentiment for portfolio assets
    - Community discussions about portfolio holdings
    - Reddit/Twitter sentiment on traditional market assets
    - Social media buzz affecting portfolio assets

    Args:
        query (str): The search query related to portfolio assets.
        k (int, optional): The number of top results to return. Defaults to 1.

    Returns:
        dict: Contains relevant social media sentiment from the most recent market social media reports for the current portfolio.
    """
    try:
        logger.info(f"Searching portfolio social media reports for: {query}")
        
        # Get the most recent document for context information only
        most_recent = market_sm_collection.find_one(
            {}, 
            sort=[("timestamp", -1)]
        )
        
        if not most_recent:
            return {"results": "No social media reports found for the current portfolio."}
        
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
                    "index": f"{REPORT_MARKET_SM_VECTOR_INDEX_NAME}",
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
        
        results = list(market_sm_collection.aggregate(pipeline))
        
        # If no results from vector search, just return the most recent document
        if not results:
            # Extract relevant information
            overall_diagnosis = most_recent.get("report", {}).get("overall_news_diagnosis", "No social media diagnosis available")
            asset_sm_sentiments = most_recent.get("report", {}).get("asset_sm_sentiments", [])
            
            # Format asset social media sentiment information
            sm_sentiments = []
            for sentiment in asset_sm_sentiments:
                asset = sentiment.get("asset", "Unknown")
                sentiment_text = sentiment.get("sentiment", "Unknown")
                sm_sentiments.append(f"{asset}: {sentiment_text}")
            
            return {
                "report_date": report_date,
                "portfolio_assets": portfolio_assets,
                "overall_diagnosis": overall_diagnosis,
                "social_media_sentiments": sm_sentiments,
                "note": "This is the most recent social media report for your portfolio."
            }
        
        # Re-rank results by combining vector similarity score with recency
        now = datetime.datetime.now(datetime.timezone.utc)
        
        # Make sure most recent is always in the results
        most_recent_in_results = False
        
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
        overall_diagnosis = report.get("overall_news_diagnosis", "No social media diagnosis available")
        asset_sm_sentiments = report.get("asset_sm_sentiments", [])
        
        # Format social media sentiments
        sm_sentiments = []
        for sentiment in asset_sm_sentiments:
            asset = sentiment.get("asset", "Unknown")
            sentiment_text = sentiment.get("sentiment", "Unknown")
            sm_sentiments.append(f"{asset}: {sentiment_text}")
        
        # Format the return data
        return {
            "report_date": report_date,
            "portfolio_assets": portfolio_assets[:5],  # Show top 5 portfolio assets
            "overall_diagnosis": overall_diagnosis,
            "social_media_sentiments": sm_sentiments,
            "note": "This information is from the most relevant and recent portfolio social media reports."
        }
        
    except Exception as e:
        logger.error(f"Error searching portfolio social media reports: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}
    
@tool
def get_vix_closing_value_tool(query: str) -> str:
    """
    Get the most recent closing value of the VIX index.

    IMPORTANT: This tool provides the VIX index closing value.

    Use this tool when you need:
    - VIX index closing value OR VIX closing value today.

    Args:
        query (str): The search query related to VIX index closing value.
    
    Returns:
        str: VIX index closing value answer.
    """
    try:
        # Log the query
        logger.info(f"Fetching VIX index closing because of query: {query}")

        # Get the most recent document for context information
        most_recent = market_reports_collection.find_one(
            {}, 
            sort=[("timestamp", -1)]
        )
        
        if not most_recent:
            return "VIX closing value not available. No market analysis reports found."
        
        # Extract the date of the most recent report
        report = most_recent.get("report", {})
        vix_closing_value = report.get("market_volatility_index", {}).get("fluctuation_answer", "")
        
        # Extract only the VIX close price part
        if "VIX close price is" in vix_closing_value:
            # Parse just the "VIX close price is X.XX" part
            vix_part = vix_closing_value.split("(")[0].strip()
            return vix_part
        else:
            return "VIX closing value is not available in the expected format."

    except Exception as e:
        logger.error(f"Error fetching VIX index closing value: {str(e)}")
        return f"Unable to retrieve VIX closing value: {str(e)}"
    

@tool
def get_portfolio_allocation_tool(query: str) -> dict:
    """Get the most recent portfolio allocation.

    IMPORTANT: This tool provides the most recent portfolio allocation.

    Use this tool when you need:
    - Portfolio allocation for the current portfolio.
    - Asset distribution information
    - Current investment breakdown

    Args:
        query (str): The search query related to portfolio allocation.

    Returns:
        dict: Portfolio allocation showing assets, descriptions, and percentages.
    """
    try:
        # Log the query
        logger.info(f"Fetching portfolio allocation because of query: {query}")

        # Get the most recent document for context information
        most_recent = market_reports_collection.find_one(
            {}, 
            sort=[("timestamp", -1)]
        )
        
        if not most_recent:
            return {"error": "Portfolio allocation not available. No market analysis reports found."}
        
        # Extract the portfolio allocation data
        portfolio_allocation = most_recent.get("portfolio_allocation", [])
        
        if not portfolio_allocation:
            return {"error": "Portfolio allocation data not found in the most recent report."}
        
        # Return the simplified portfolio allocation data
        return {
            "portfolio_allocation": portfolio_allocation,
        }
    
    except Exception as e:
        logger.error(f"Error fetching portfolio allocation: {str(e)}")
        return {"error": f"Unable to retrieve portfolio allocation: {str(e)}"}
    

@tool
def get_portfolio_ytd_return_tool(query: str) -> str:
    """
    Get the Year-to-Date (YTD) rate of return for the portfolio.

    IMPORTANT: This tool provides the YTD performance of the current portfolio.

    Use this tool when you need:
    - Year-to-date return of the portfolio
    - YTD portfolio performance 
    - How the portfolio has performed since the beginning of this year

    Args:
        query (str): The search query related to portfolio YTD return.

    Returns:
        str: Portfolio YTD return percentage and information.
    """
    try:
        # Log the query
        logger.info(f"Calculating portfolio YTD return for query: {query}")
        
        # Get current date information
        current_date = datetime.datetime.now()
        current_year = current_date.year
        
        # Find the first trading day entry of the current year
        start_of_year = datetime.datetime(current_year, 1, 1)
        first_entry_of_year = portfolio_performance_collection.find_one(
            {"date": {"$gte": start_of_year}},
            sort=[("date", 1)]
        )
        
        if not first_entry_of_year:
            return "Could not find portfolio data for the current year."
        
        # Get the most recent entry
        latest_entry = portfolio_performance_collection.find_one(
            {}, 
            sort=[("date", -1)]
        )
        
        if not latest_entry:
            return "Could not find recent portfolio performance data."
            
        # Extract start and end dates for better context
        start_date = first_entry_of_year.get("date").strftime("%Y-%m-%d")
        end_date = latest_entry.get("date").strftime("%Y-%m-%d")
        
        # Extract cumulative returns
        start_cumulative_return = first_entry_of_year.get("percentage_of_cumulative_return", 0)
        end_cumulative_return = latest_entry.get("percentage_of_cumulative_return", 0)
        
        # Calculate YTD return
        ytd_return = end_cumulative_return - start_cumulative_return
        
        # Format the response with detailed information
        return (f"The portfolio's YTD return from {start_date} to {end_date} is {ytd_return:.2f}%. "
                f"(Starting cumulative return: {start_cumulative_return:.2f}%, "
                f"Current cumulative return: {end_cumulative_return:.2f}%)")
        
    except Exception as e:
        logger.error(f"Error calculating portfolio YTD return: {str(e)}")
        return f"Unable to calculate YTD return: {str(e)}"