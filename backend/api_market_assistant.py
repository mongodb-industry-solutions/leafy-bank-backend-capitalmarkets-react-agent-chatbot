from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging
from agent.react_agent import MarketAssistantReactAgent

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
    thread_id: Optional[str] = None
    message: str

class MessageResponse(BaseModel):
    status: str
    thread_id: str
    message_type: str
    final_answer: str
    tool_calls: Optional[List[dict]] = None

@router.post("/send-message", response_model=MessageResponse)
async def send_message(request: MessageRequest):
    """Send a message to the agent and get a response."""
    try:
        # NOTE: If the thread_id is None, the agent will create a new thread.
        # If the thread_id is not None, the agent will continue the conversation in that thread.
        if request.thread_id is None:
            result = await market_assistant_service.process_user_message(
                message_content=request.message
            )
        else:
            result = await market_assistant_service.process_user_message(
                message_content=request.message,
                thread_id=request.thread_id
            )
        return result
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))