import os
from datetime import datetime, timezone
import logging
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from agent.bedrock.client import BedrockClient
from pymongo import AsyncMongoClient

# How to use MongoDB checkpointer for persistence:
# https://langchain-ai.github.io/langgraph/how-tos/persistence_mongodb/
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

# How to use the pre-built ReAct agent:
# https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# Tools
from agent.react_agent_tools import tavily_search_tool, market_analysis_reports_vector_search_tool, market_news_reports_vector_search_tool

# Initialize dotenv to load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MarketAssistantReactAgent:
    """
    Service class for the Market Assistant React Agent.
    Provides methods for creating and managing chat sessions with the agent.
    """
    
    def __init__(self):
        """Initialize the Market Assistant React Agent service."""
        # Load MongoDB configuration from environment variables
        self.mongodb_uri = os.getenv("MONGODB_URI")
        self.database_name = os.getenv("DATABASE_NAME")
        self.checkpoints_collection = os.getenv("CHECKPOINTS_AIO_COLLECTION", "checkpoints_aio")
        self.checkpoint_writes_collection = os.getenv("CHECKPOINTS_WRITES_AIO_COLLECTION", "checkpoint_writes_aio")
        
        # Instantiate Bedrock client
        bedrock_client = BedrockClient()._get_bedrock_client()

        # Initialize LLM
        self.chat_completion_model_id = os.getenv("CHAT_COMPLETIONS_MODEL_ID")
        self.llm = ChatBedrock(model=self.chat_completion_model_id,
                client=bedrock_client,
                temperature=0)
        
        # Initialize async MongoDB client
        self.async_mongodb_client = AsyncMongoClient(self.mongodb_uri)
        self.async_mongodb_memory_collection = self.async_mongodb_client[self.database_name][self.checkpoints_collection]
        self.memory = AsyncMongoDBSaver(client=self.async_mongodb_client, db_name=self.database_name)
        
        # Create the agent with tools - done at initialization to avoid recreation
        self.langgraph_agent = create_react_agent(
            model=self.llm, 
            tools=[
                tavily_search_tool,
                market_analysis_reports_vector_search_tool,
                market_news_reports_vector_search_tool
            ],
            checkpointer=self.memory
        )
        
        logger.info("MarketAssistantReactAgent initialized with LangGraph agent and tools")
    
    async def cleanup_old_threads(self, keep_threads=None):
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
            result = await self.async_mongodb_client[self.database_name][self.checkpoints_collection].delete_many(query)
            deleted_count += result.deleted_count
            
            # Delete from the checkpoint writes collection
            result = await self.async_mongodb_client[self.database_name][self.checkpoint_writes_collection].delete_many(query)
            deleted_count += result.deleted_count
            
            return deleted_count
        
        except Exception as e:
            logger.error(f"Error cleaning up old threads: {str(e)}")
            return 0
    
    async def clear_all_memory(self):
        """
        Clear all chat memory.
        
        Returns:
            dict: Information about the operation
        """
        deleted_count = await self.cleanup_old_threads()
        
        return {
            "status": "success",
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} memory records. All chat history has been cleared."
        }
        
    async def process_user_message(self, thread_id, message_content):
        """
        Process a user message and get the agent's response.
        
        Args:
            thread_id (str): The ID of the thread for this conversation
            message_content (str): The content of the user's message
            
        Returns:
            dict: The agent's response and any tool calls made
        """
        # Check for special commands
        if message_content.lower() == "clear all memory":
            result = await self.clear_all_memory()
            return {
                "status": "success",
                "message_type": "system",
                "final_answer": result["message"],
                "thread_id": thread_id
            }
        
        # Process the message with the agent
        tool_calls = []
        final_answer = ""
        
        try:
            # Log that we're about to invoke the agent with tools
            logger.info(f"Processing message with agent, thread_id: {thread_id}")
            
            # Use the async stream method of the LangGraph agent to get the agent's answer
            async for chunk in self.langgraph_agent.astream(
                {"messages": [HumanMessage(content=message_content)]},
                {"configurable": {"thread_id": thread_id}}
            ):
                # Log full chunk for debugging
                logger.debug(f"Chunk: {chunk}")
                
                # Check if this is the final answer
                if "agent" in chunk and chunk.get("agent", {}).get("is_final", False):
                    for message in chunk["agent"]["messages"]:
                        final_answer = message.content
                        logger.info(f"Final answer captured: {final_answer[:100]}...")
                
                # Track intermediate steps - this is where ReAct's tool usage appears
                if "intermediate_steps" in chunk:
                    for step in chunk["intermediate_steps"]:
                        # Extract tool call
                        if hasattr(step, "action"):
                            tool_name = step.action.tool
                            tool_input = step.action.tool_input
                            
                            # Handle different input formats
                            if isinstance(tool_input, dict) and "query" in tool_input:
                                tool_query = tool_input["query"]
                            else:
                                tool_query = str(tool_input)
                            
                            tool_calls.append({
                                "tool_name": tool_name,
                                "query": tool_query
                            })
                            
                            logger.info(f"Tool called: {tool_name} with query: {tool_query[:100]}...")
                
                # Process chunks to extract the response and any tool calls
                if "agent" in chunk:
                    for message in chunk["agent"]["messages"]:
                        # Check for direct tool_calls attribute
                        if hasattr(message, "tool_calls") and message.tool_calls:
                            for tool_call in message.tool_calls:
                                tool_name = tool_call.get("name")
                                tool_args = tool_call.get("args", {})
                                tool_query = tool_args.get("query", "") if isinstance(tool_args, dict) else str(tool_args)
                                
                                tool_calls.append({
                                    "tool_name": tool_name,
                                    "query": tool_query,
                                    "id": tool_call.get("id", "")
                                })
                                
                                logger.info(f"Tool called (direct): {tool_name} with query: {tool_query[:50]}...")
                        
                        # Also check the additional_kwargs approach as fallback
                        elif hasattr(message, "additional_kwargs") and "tool_calls" in message.additional_kwargs:
                            tool_calls_data = message.additional_kwargs["tool_calls"]
                            for tool_call in tool_calls_data:
                                tool_name = tool_call["function"]["name"] if "function" in tool_call else tool_call.get("name", "")
                                
                                # Handle different formats of tool call arguments
                                if "function" in tool_call and "arguments" in tool_call["function"]:
                                    try:
                                        tool_arguments = eval(tool_call["function"]["arguments"])
                                        tool_query = tool_arguments.get("query", "")
                                    except:
                                        tool_query = tool_call["function"]["arguments"]
                                elif "args" in tool_call:
                                    tool_args = tool_call["args"]
                                    tool_query = tool_args.get("query", "") if isinstance(tool_args, dict) else str(tool_args)
                                else:
                                    tool_query = ""
                                    
                                tool_calls.append({
                                    "tool_name": tool_name,
                                    "query": tool_query,
                                    "id": tool_call.get("id", "")
                                })
                                
                                logger.info(f"Tool called (kwargs): {tool_name} with query: {tool_query[:100]}...")
                        # If message doesn't have tool_calls, capture its content as potential final_answer
                        elif not final_answer and message.content and message.content.strip():
                            final_answer = message.content
            
            # Find the final answer in the last step if not already captured
            if not final_answer:
                async for checkpoint_tuple in self.memory.alist({"configurable": {"thread_id": thread_id}}):
                    if "channel_values" in checkpoint_tuple.checkpoint:
                        messages = checkpoint_tuple.checkpoint["channel_values"].get("messages", [])
                        if messages and hasattr(messages[-1], "content"):
                            final_answer = messages[-1].content
            
            # Return the response with tool calls
            return {
                "status": "success",
                "message_type": "agent",
                "final_answer": final_answer,
                "tool_calls": tool_calls,
                "thread_id": thread_id,
            }
                    
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            logger.exception("Full stack trace:")
            return {
                "status": "error",
                "message": f"An error occurred processing your message: {str(e)}",
                "thread_id": thread_id
            }