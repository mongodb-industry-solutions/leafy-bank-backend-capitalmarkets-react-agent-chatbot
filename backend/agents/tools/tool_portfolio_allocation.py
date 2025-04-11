from agents.tools.db.mdb import MongoDBConnector
from typing import Union, TypeVar
import os
import logging
from dotenv import load_dotenv

# Import both state types
from agents.tools.states.agent_market_analysis_state import MarketAnalysisAgentState
from agents.tools.states.agent_market_news_state import MarketNewsAgentState

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Type variable for state
StateType = TypeVar('StateType', MarketAnalysisAgentState, MarketNewsAgentState)

class PortfolioAllocationTool(MongoDBConnector):
    def __init__(self, uri=None, database_name=None, collection_name=None):
        super().__init__(uri, database_name)
        self.collection_name = collection_name or os.getenv("PORTFOLIO_COLLECTION", "portfolio_allocation")
        self.collection = self.get_collection(self.collection_name)
        logger.info("PortfolioAllocationTool initialized")

    def check_portfolio_allocation(self, state: Union[MarketAnalysisAgentState, MarketNewsAgentState]) -> dict:
        """Query the portfolio_allocation collection"""
        message = "[Tool] Check portfolio allocation."
        logger.info(message)

        # Query the collection
        results = list(self.collection.find({}, {"symbol": 1, "description": 1, "allocation_percentage": 1, "_id": 0}))
        
        # Transform the results into the required format
        portfolio_allocation = [
            {
                "asset": result["symbol"],
                "description": result["description"],
                "allocation_percentage": result["allocation_percentage"]
            }
            for result in results
        ]

        # Update the state with the portfolio allocation
        # Get the correct PortfolioAllocation class based on state type
        if isinstance(state, MarketAnalysisAgentState):
            from agents.tools.states.agent_market_analysis_state import PortfolioAllocation
            next_node = "asset_trends_node"
        else:  # MarketNewsAgentState
            from agents.tools.states.agent_market_news_state import PortfolioAllocation
            next_node = "fetch_market_news_node"
            
        # Apply to state using the correct type
        state.portfolio_allocation = [
            PortfolioAllocation(**allocation) for allocation in portfolio_allocation
        ]
        
        # Append the message to the updates list
        state.updates.append(message)

        # Set the next step based on state type
        state.next_step = next_node

        return {"portfolio_allocation": portfolio_allocation, "updates": state.updates, "next_step": state.next_step}

# Initialize the PortfolioAllocationTool
portfolio_allocation_tool = PortfolioAllocationTool()

# Define tools - this is the function used by both workflows
def check_portfolio_allocation_tool(state: Union[MarketAnalysisAgentState, MarketNewsAgentState]) -> dict:
    """Query the portfolio_allocation collection for any supported state type"""
    return portfolio_allocation_tool.check_portfolio_allocation(state=state)

if __name__ == "__main__":
    # Example usage with both state types
    from agents.tools.states.agent_market_analysis_state import MarketAnalysisAgentState
    from agents.tools.states.agent_market_news_state import MarketNewsAgentState
    
    # Test with analysis state
    analysis_state = MarketAnalysisAgentState()
    analysis_result = check_portfolio_allocation_tool(analysis_state)
    print("\nAnalysis State Next Step:", analysis_state.next_step)
    # Print the analysis result
    print("Analysis Result:", analysis_result)
    # Print the analysis state
    print("Analysis State:", analysis_state)
    
    # Test with news state
    news_state = MarketNewsAgentState()
    news_result = check_portfolio_allocation_tool(news_state)
    print("\nNews State Next Step:", news_state.next_step)
    # Print the news result
    print("News Result:", news_result)
    # Print the news state
    print("News State:", news_state)