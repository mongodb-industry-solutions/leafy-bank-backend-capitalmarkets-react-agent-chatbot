# Imports 
import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from agent.bedrock.client import BedrockClient
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console
from langchain.tools import tool
from agent.vogayeai.vogaye_ai_embeddings import VogayeAIEmbeddings
from agent.db.mdb import MongoDBConnector

from datetime import datetime, timezone

from pymongo import AsyncMongoClient
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

from langgraph.prebuilt import create_react_agent

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize dotenv to load environment variables
load_dotenv()

# Initialize Rich for better output formatting and visualization
rich = Console()

# Instantiate Bedrock client
bedrock_client = BedrockClient()._get_bedrock_client()

# Initialize ChatBedrock LLM
llm = ChatBedrockConverse(model=os.getenv("CHAT_COMPLETIONS_MODEL_ID"),
                client=bedrock_client,
                temperature=0)

# Initialize TavilySearch
tavily = TavilySearch(max_results=3)

# Initialize the VoyageAI embeddings generator
embedding_model_id = os.getenv("EMBEDDINGS_MODEL_ID", "voyage-finance-2")
ve = VogayeAIEmbeddings(api_key=os.getenv("VOYAGE_API_KEY"))

# Initialize MongoDB collections for reports
market_collection_name = os.getenv("REPORTS_COLLECTION_MARKET_ANALYSIS", "reports_market_analysis")
news_collection_name = os.getenv("REPORTS_COLLECTION_MARKET_NEWS", "reports_market_news")
market_sm_collection_name = os.getenv("REPORTS_COLLECTION_MARKET_SM", "reports_market_sm")
portfolio_allocation_collection_name = os.getenv("PORTFOLIO_ALLOCATION_COLLECTION", "portfolio_allocation")
portfolio_performance_collection_name = os.getenv("PORTFOLIO_PERFORMANCE_COLLECTION", "portfolio_performance")
mongodb_connector = MongoDBConnector()
market_collection = mongodb_connector.get_collection(market_collection_name)
news_collection = mongodb_connector.get_collection(news_collection_name)
market_sm_collection = mongodb_connector.get_collection(market_sm_collection_name)
portfolio_allocation_collection = mongodb_connector.get_collection(portfolio_allocation_collection_name)
portfolio_performance_collection = mongodb_connector.get_collection(portfolio_performance_collection_name)

# Getting environment variables for vector index names
REPORT_MARKET_ANALISYS_VECTOR_INDEX_NAME = os.getenv("REPORT_MARKET_ANALISYS_VECTOR_INDEX_NAME")
REPORT_MARKET_NEWS_VECTOR_INDEX_NAME = os.getenv("REPORT_MARKET_NEWS_VECTOR_INDEX_NAME")
REPORT_MARKET_SM_VECTOR_INDEX_NAME = os.getenv("REPORT_MARKET_SM_VECTOR_INDEX_NAME")

# Getting environment variables for vector field names
REPORT_VECTOR_FIELD = os.getenv("REPORT_VECTOR_FIELD", "report_embedding")


# Function to generate a new thread_id
def generate_thread_id():
    """Generate a unique thread_id based on current timestamp"""
    return f"thread_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

# Function to list available threads
async def list_available_threads(memory_collection, limit=3):
    """
    List the most recent thread_ids from the MongoDB memory store
    
    Args:
        memory_collection: MongoDB collection to query
        limit: Maximum number of threads to return (default: 3)
        
    Returns:
        List of the most recent thread_ids
    """
    threads = []
    
    try:
        # First approach: try to get distinct thread_ids directly
        try:
            distinct_threads = await memory_collection.distinct("thread_id")
            # Sort by extracting timestamp from thread_id format (thread_YYYYMMDD_HHMMSS)
            sorted_threads = sorted(
                [t for t in distinct_threads if t and t != "null"],
                key=lambda x: x.split('_')[1:] if len(x.split('_')) >= 3 else "",
                reverse=True
            )
            return sorted_threads[:limit]
        except Exception as e1:
            rich.print(f"[bright_red]Error with distinct query: {str(e1)}[/bright_red]")
        
        # Second approach: try using find() instead of aggregate
        try:
            cursor = memory_collection.find(
                {"thread_id": {"$ne": None}},
                {"thread_id": 1, "_id": 0}
            ).sort("_id", -1).limit(limit * 10)  # Get more than needed to account for duplicates
            
            thread_set = set()
            async for doc in cursor:
                thread_id = doc.get("thread_id")
                if thread_id and thread_id != "null" and thread_id not in thread_set:
                    thread_set.add(thread_id)
                    threads.append(thread_id)
                    if len(threads) >= limit:
                        break
            
            return threads
        except Exception as e2:
            rich.print(f"[bright_red]Error with find query: {str(e2)}[/bright_red]")
            
    except Exception as e:
        rich.print(f"[bright_red]Error retrieving thread IDs: {str(e)}[/bright_red]")
    
    # If all methods fail, return an empty list
    return threads

# Function to clean up old threads
async def cleanup_old_threads(mongodb_client, database_name, checkpoints_collection, checkpoint_writes_collection, keep_threads=None):
    """
    Clean up old threads from the memory collections
    
    Args:
        mongodb_client: AsyncMongoClient instance
        database_name: Database name
        checkpoints_collection: Name of the checkpoints collection
        checkpoint_writes_collection: Name of the checkpoint writes collection
        keep_threads: List of thread_ids to keep (if None, deletes all threads)
        
    Returns:
        Number of deleted documents
    """
    deleted_count = 0
    
    try:
        # Build the query to delete documents not in the keep_threads list
        query = {} if keep_threads is None else {"thread_id": {"$nin": keep_threads}}
        
        # Delete from the checkpoints collection
        result = await mongodb_client[database_name][checkpoints_collection].delete_many(query)
        deleted_count += result.deleted_count
        
        # Delete from the checkpoint writes collection
        result = await mongodb_client[database_name][checkpoint_writes_collection].delete_many(query)
        deleted_count += result.deleted_count
        
        return deleted_count
    
    except Exception as e:
        rich.print(f"[bright_red]Error cleaning up old threads: {str(e)}[/bright_red]")
        return 0

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
    - Market trends and insights analysis for portfolio assets
    - Technical indicators for traditional assets (moving averages)
    - Market conditions to inform reallocation decisions
    - Asset-specific diagnostics for portfolio holdings
    - Market volatility analysis for traditional assets

    Args:
        query (str): The search query related to portfolio assets, trends, insights, etc.
        k (int, optional): The number of top results to return. Defaults to 1.

    Returns:
        dict: Contains relevant sections from the most recent market analysis report for the current portfolio.
    """
    try:
        rich.print(f"\n[Action] Searching portfolio market analysis for: {query}")
        
        # Get the most recent document for context information only
        most_recent = market_collection.find_one(
            {}, 
            sort=[("timestamp", -1)]
        )
        
        if not most_recent:
            return {"results": "No market analysis reports found for the current portfolio."}
        
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
                    "index": f"{REPORT_MARKET_ANALISYS_VECTOR_INDEX_NAME}",
                    "path": REPORT_VECTOR_FIELD,
                    "queryVector": query_embedding,
                    "numCandidates": 5,
                    "limit": k
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "date_string": 1,
                    "report": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(market_collection.aggregate(pipeline))
        
        # If no results from vector search, just return the most recent document
        if not results:
            # Extract relevant information
            overall_diagnosis = most_recent.get("report", {}).get("overall_diagnosis", "No diagnosis available")
            asset_trends = most_recent.get("report", {}).get("asset_trends", [])
            asset_insights = []
            
            # Format asset trends information
            for trend in asset_trends[:5]:  # Limit to 5 assets for readability
                asset = trend.get("asset", "Unknown")
                diagnosis = trend.get("diagnosis", "No diagnosis")
                asset_insights.append(f"{asset}: {diagnosis}")
            
            return {
                "report_date": report_date,
                "portfolio_assets": portfolio_assets[:5],  # Show top 5 portfolio assets
                "overall_diagnosis": overall_diagnosis,
                "asset_insights": asset_insights,
                "note": "This is the most recent market analysis for your portfolio."
            }
        
        # Process vector search results
        report_data = results[0]
        overall_diagnosis = report_data.get("report", {}).get("overall_diagnosis", "No diagnosis available")
        
        # Format the return data
        return {
            "report_date": report_date,
            "portfolio_assets": portfolio_assets[:5],  # Show top 5 portfolio assets
            "overall_diagnosis": overall_diagnosis,
            "note": "This information is from relevant portfolio market analysis reports."
        }
        
    except Exception as e:
        rich.print(f"[bright_red]Error searching portfolio market reports: {str(e)}[/bright_red]")
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
        rich.print(f"\n[Action] Searching portfolio news reports for: {query}")
        
        # Get the most recent document for context information only
        most_recent = news_collection.find_one(
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
                    "limit": k
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "date_string": 1,
                    "report.asset_news_summary": 1,
                    "report.overall_news_diagnosis": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(news_collection.aggregate(pipeline))
        
        # If no results from vector search, just return the most recent document
        if not results:
            # Extract relevant information
            overall_diagnosis = most_recent.get("report", {}).get("overall_news_diagnosis", "No news diagnosis available")
            asset_news_summaries = most_recent.get("report", {}).get("asset_news_summary", [])
            
            # Format asset news summary information
            news_summaries = []
            for summary in asset_news_summaries[:3]:  # Limit to 3 assets for readability
                asset = summary.get("asset", "Unknown")
                summary_text = summary.get("summary", "No summary available")
                sentiment = summary.get("overall_sentiment_category", "Unknown")
                news_summaries.append(f"{asset} ({sentiment}): {summary_text}")
            
            return {
                "report_date": report_date,
                "portfolio_assets": portfolio_assets[:5],  # Show top 5 portfolio assets
                "overall_diagnosis": overall_diagnosis,
                "news_summaries": news_summaries,
                "note": "This is the most recent news report for your portfolio."
            }
        
        # Process vector search results
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
            "news_summaries": news_summaries[:3],  # Limit to 3 for readability
            "note": "This information is from relevant portfolio news reports."
        }
        
    except Exception as e:
        rich.print(f"[bright_red]Error searching portfolio news reports: {str(e)}[/bright_red]")
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
        rich.print(f"\n[Action] Searching portfolio social media reports for: {query}")
        
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
                    "limit": k
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "date_string": 1,
                    "report": 1,
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
            for sentiment in asset_sm_sentiments[:3]:  # Limit to 3 assets for readability
                asset = sentiment.get("asset", "Unknown")
                sentiment_text = sentiment.get("sentiment", "Unknown")
                sm_sentiments.append(f"{asset}: {sentiment_text}")
            
            return {
                "report_date": report_date,
                "portfolio_assets": portfolio_assets[:5],  # Show top 5 portfolio assets
                "overall_diagnosis": overall_diagnosis,
                "social_media_sentiments": sm_sentiments,
                "note": "This is the most recent social media report for your portfolio."
            }
        
        # Process vector search results
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
            "social_media_sentiments": sm_sentiments[:3],  # Limit to 3 for readability
            "note": "This information is from relevant portfolio social media reports."
        }
        
    except Exception as e:
        rich.print(f"[bright_red]Error searching portfolio social media reports: {str(e)}[/bright_red]")
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
        rich.print(f"\n[Action] Fetching VIX index closing value for query: {query}")

        # Get the most recent document for context information
        most_recent = market_collection.find_one(
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
        rich.print(f"[bright_red]Error fetching VIX index closing value: {str(e)}[/bright_red]")
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
        rich.print(f"\n[Action] Fetching portfolio allocation for query: {query}")

        # Get the most recent document for context information
        most_recent = market_collection.find_one(
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
        rich.print(f"[bright_red]Error fetching portfolio allocation: {str(e)}[/bright_red]")
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
        rich.print(f"\n[Action] Calculating portfolio YTD return for query: {query}")
        
        # Get current date information
        current_date = datetime.now()
        current_year = current_date.year
        
        # Find the first trading day entry of the current year
        start_of_year = datetime(current_year, 1, 1)
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
        rich.print(f"[bright_red]Error calculating portfolio YTD return: {str(e)}[/bright_red]")
        return f"Unable to calculate YTD return: {str(e)}"


# Define a function to process chunks from the agent
def process_chunks(chunk):
    """
    Processes a chunk from the agent and displays information about tool calls or the agent's answer.

    Parameters:
        chunk (dict): A dictionary containing information about the agent's messages.

    Returns:
        None

    This function processes a chunk of data to check for agent messages.
    It iterates over the messages and checks for tool calls.
    If a tool call is found, it extracts the tool name and query, then prints a formatted message using the Rich library.
    If no tool call is found, it extracts and prints the agent's answer using the Rich library.
    """

    # Check if the chunk contains an agent's message
    if "agent" in chunk:
        # Iterate over the messages in the chunk
        for message in chunk["agent"]["messages"]:
            # Check if the message contains tool calls
            if "tool_calls" in message.additional_kwargs:
                # If the message contains tool calls, extract and display an informative message with tool call details

                # Extract all the tool calls
                tool_calls = message.additional_kwargs["tool_calls"]

                # Iterate over the tool calls
                for tool_call in tool_calls:
                    # Extract the tool name
                    tool_name = tool_call["function"]["name"]

                    # Extract the tool query
                    tool_arguments = eval(tool_call["function"]["arguments"])
                    tool_query = tool_arguments.get("query", "")

                    # Display an informative message with tool call details
                    rich.print(
                        f"\nThe agent is calling the tool [on deep_sky_blue1]{tool_name}[/on deep_sky_blue1] with the query [on deep_sky_blue1]{tool_query}[/on deep_sky_blue1]. Please wait for the agent's answer[deep_sky_blue1]...[/deep_sky_blue1]",
                        style="deep_sky_blue1",
                    )
            else:
                # If the message doesn't contain tool calls, extract and display the agent's answer

                # Extract the agent's answer
                agent_answer = message.content

                # Display the agent's answer
                rich.print(f"\nAgent:\n{agent_answer}", style="black on white")


# Define an async function to process checkpoints from the memory
async def process_checkpoints(checkpoints):
    """
    Asynchronously processes a list of checkpoints and displays relevant information.
    Each checkpoint consists of a tuple where the first element is the index and the second element is an object
    containing various details about the checkpoint. The function distinguishes between messages from the user
    and the agent, displaying them accordingly.

    Parameters:
        checkpoints (list): A list of checkpoint tuples to process.

    Returns:
        None

    This function processes a list of checkpoints asynchronously.
    It iterates over the checkpoints and displays the following information for each checkpoint:
    - Timestamp
    - Checkpoint ID
    - Messages associated with the checkpoint
    """

    rich.print("\n==========================================================\n")

    # Initialize an empty list to store the checkpoints
    checkpoints_list = []

    # Iterate over the checkpoints and add them to the list in an async way
    async for checkpoint_tuple in checkpoints:
        checkpoints_list.append(checkpoint_tuple)

    # Iterate over the list of checkpoints
    for idx, checkpoint_tuple in enumerate(checkpoints_list):
        # Extract key information about the checkpoint
        checkpoint = checkpoint_tuple.checkpoint
        messages = checkpoint["channel_values"].get("messages", [])

        # Display checkpoint information
        rich.print(f"[white]Checkpoint:[/white]")
        rich.print(f"[black]Timestamp: {checkpoint['ts']}[/black]")
        rich.print(f"[black]Checkpoint ID: {checkpoint['id']}[/black]")

        # Display checkpoint messages
        for message in messages:
            if isinstance(message, HumanMessage):
                rich.print(
                    f"[bright_magenta]User: {message.content}[/bright_magenta] [bright_cyan](Message ID: {message.id})[/bright_cyan]"
                )
            elif isinstance(message, AIMessage):
                rich.print(
                    f"[bright_magenta]Agent: {message.content}[/bright_magenta] [bright_cyan](Message ID: {message.id})[/bright_cyan]"
                )

        rich.print("")

    rich.print("==========================================================")


# Define an async function to chat with the agent
async def main():
    """
    Entry point of the application that manages MongoDB chat memory and user interactions.
    """
    # Load MongoDB configuration from environment variables
    MONGODB_URI = os.getenv("MONGODB_URI")
    DATABASE_NAME = os.getenv("DATABASE_NAME")
    CHECKPOINTS_AIO_COLLECTION = os.getenv("CHECKPOINTS_AIO_COLLECTION", "checkpoints_aio")
    CHECKPOINTS_WRITES_AIO_COLLECTION = os.getenv("CHECKPOINTS_WRITES_AIO_COLLECTION", "checkpoint_writes_aio")

    # Initialize the async MongoDB client
    async_mongodb_client = AsyncMongoClient(MONGODB_URI)
    
    # Initialize the async MongoDB collection for memory
    async_mongodb_memory_collection = async_mongodb_client[DATABASE_NAME][CHECKPOINTS_AIO_COLLECTION]

    # Initialize persistent chat memory
    memory = AsyncMongoDBSaver(client=async_mongodb_client, db_name=DATABASE_NAME)

    # Ask user if they want a new thread, continue an existing one, or clear memory
    rich.print("\nWelcome to the Market Assistant!\n", style="bold green")
    rich.print("Choose an option:")
    rich.print("1. Start a new chat session")
    rich.print("2. Continue an existing chat session")
    rich.print("3. Clear all chat memory")
    
    choice = input("\nEnter your choice (1, 2, or 3): ")
    
    if choice == "3":
        # Confirm with the user before deleting all memory
        confirm = input("Are you sure you want to clear ALL chat memory? This cannot be undone. (yes/no): ")
        if confirm.lower() == "yes":
            deleted_count = await cleanup_old_threads(
                async_mongodb_client,
                DATABASE_NAME,
                CHECKPOINTS_AIO_COLLECTION,
                CHECKPOINTS_WRITES_AIO_COLLECTION
            )
            rich.print(f"Deleted {deleted_count} memory records. All chat history has been cleared.", style="yellow")
        else:
            rich.print("Memory cleanup canceled.", style="yellow")
        
        # After cleaning up, start a new chat
        thread_id = generate_thread_id()
    elif choice == "2":
        # Get only the 3 most recent threads
        threads = await list_available_threads(async_mongodb_memory_collection, limit=3)
        
        # Clean up older threads
        if threads:
            deleted_count = await cleanup_old_threads(
                async_mongodb_client,
                DATABASE_NAME,
                CHECKPOINTS_AIO_COLLECTION,
                CHECKPOINTS_WRITES_AIO_COLLECTION,
                keep_threads=threads
            )
            if deleted_count > 0:
                rich.print(f"Cleaned up {deleted_count} older memory records.", style="dim")
        
        if not threads:
            rich.print("No existing chat sessions found. Starting a new one.", style="yellow")
            thread_id = generate_thread_id()
        else:
            # Display available threads
            rich.print("\nAvailable chat sessions:", style="bold")
            for i, thread in enumerate(threads, 1):
                # Try to extract the datetime from the thread_id for a more readable format
                try:
                    date_part = thread.split('_')[1]
                    time_part = thread.split('_')[2]
                    formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
                    rich.print(f"{i}. Session from {formatted_date} (ID: {thread})")
                except:
                    rich.print(f"{i}. {thread}")
            
            # Let user select a thread
            selection = input("\nEnter the number of the session you want to continue (or 'n' for a new session): ")

            if selection.lower() == 'n':
                thread_id = generate_thread_id()
            else:
                # First check if selection is one of the actual thread_ids in the list
                if selection in threads:
                    thread_id = selection
                else:
                    try:
                        idx = int(selection) - 1
                        if 0 <= idx < len(threads):
                            thread_id = threads[idx]
                        else:
                            rich.print("Invalid selection. Starting a new session.", style="yellow")
                            thread_id = generate_thread_id()
                    except ValueError:
                        rich.print("Invalid input. Starting a new session.", style="yellow")
                        thread_id = generate_thread_id()
    else:
        # Generate a new thread_id for a new chat session
        thread_id = generate_thread_id()
        
        # Clean up older threads, keeping only the 2 most recent ones (plus this new one)
        threads = await list_available_threads(async_mongodb_memory_collection, limit=2)
        if threads:
            threads.append(thread_id)  # Also keep the new thread
            deleted_count = await cleanup_old_threads(
                async_mongodb_client,
                DATABASE_NAME,
                CHECKPOINTS_AIO_COLLECTION,
                CHECKPOINTS_WRITES_AIO_COLLECTION,
                keep_threads=threads
            )
            if deleted_count > 0:
                rich.print(f"Cleaned up {deleted_count} older memory records.", style="dim")
    
    rich.print(f"\nChat session ID: {thread_id}", style="cyan")

    # Create a LangGraph agent with the tools
    langgraph_agent = create_react_agent(
        model=llm, 
        tools=[
            tavily, 
            market_analysis_reports_vector_search_tool, 
            market_news_reports_vector_search_tool,
            market_social_media_reports_vector_search_tool,
            get_vix_closing_value_tool,
            get_portfolio_allocation_tool,
            get_portfolio_ytd_return_tool
        ], 
        checkpointer=memory
    )

    # Loop until the user chooses to quit the chat
    while True:
        # Get the user's question and display it in the terminal
        user_question = input("\nUser:\n")

        # Check if the user wants to quit the chat
        if user_question.lower() == "quit":
            rich.print(
                "\nAgent:\nHave a nice day! :wave:\n", style="black on white"
            )
            break
            
        # Check if the user wants to clear all memory
        if user_question.lower() == "clear all memory":
            confirm = input("Are you sure you want to clear ALL chat memory? This cannot be undone. (yes/no): ")
            if confirm.lower() == "yes":
                deleted_count = await cleanup_old_threads(
                    async_mongodb_client,
                    DATABASE_NAME,
                    CHECKPOINTS_AIO_COLLECTION,
                    CHECKPOINTS_WRITES_AIO_COLLECTION
                )
                rich.print(f"Deleted {deleted_count} memory records. All chat history has been cleared.", style="yellow")
                thread_id = generate_thread_id()  # Generate a new thread ID after clearing memory
                rich.print(f"Starting a new chat session with ID: {thread_id}", style="cyan")
            else:
                rich.print("Memory cleanup canceled.", style="yellow")
            continue

        # Use the async stream method of the LangGraph agent to get the agent's answer
        async for chunk in langgraph_agent.astream(
                {"messages": [HumanMessage(content=user_question)]},
                {"configurable": {"thread_id": thread_id}},  # Use the correct format with configurable
            ):
            # Process the chunks from the agent
            process_chunks(chunk)

            # Use the async list method of the memory to list all checkpoints that match the current thread_id
            checkpoints = memory.alist({"configurable": {"thread_id": thread_id}})
            # Process the checkpoints from the memory in an async way
            await process_checkpoints(checkpoints)

