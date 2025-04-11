from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
from agents.react_agent_market_assistant import MarketAssistantReactAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize the service
market_assistant_service = MarketAssistantReactAgent()

# Create the router
router = APIRouter(prefix="/market-assistant", tags=["Market Assistant"])

# Models for request/response
class MessageRequest(BaseModel):
    thread_id: str
    message: str

class ThreadRequest(BaseModel):
    thread_selection: Optional[str] = None

class MessageResponse(BaseModel):
    status: str
    thread_id: str
    message_type: str
    final_answer: str
    tool_calls: Optional[List[dict]] = None

class ThreadsResponse(BaseModel):
    status: str
    available_threads: Optional[List[dict]] = None
    thread_id: Optional[str] = None
    message: str
    deleted_count: Optional[int] = None

@router.post("/new-session", response_model=ThreadsResponse)
async def create_new_session():
    """Create a new chat session."""
    try:
        result = await market_assistant_service.create_new_session()
        return result
    except Exception as e:
        logger.error(f"Error creating new session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/continue-session", response_model=ThreadsResponse)
async def continue_session(request: ThreadRequest):
    """Continue an existing chat session or get a list of available sessions."""
    try:
        result = await market_assistant_service.continue_session(request.thread_selection)
        return result
    except Exception as e:
        logger.error(f"Error continuing session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear-memory", response_model=ThreadsResponse)
async def clear_memory():
    """Clear all chat memory."""
    try:
        result = await market_assistant_service.clear_all_memory()
        return result
    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/send-message", response_model=MessageResponse)
async def send_message(request: MessageRequest):
    """Send a message to the agent and get a response."""
    try:
        result = await market_assistant_service.process_user_message(
            request.thread_id, 
            request.message
        )
        return result
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))