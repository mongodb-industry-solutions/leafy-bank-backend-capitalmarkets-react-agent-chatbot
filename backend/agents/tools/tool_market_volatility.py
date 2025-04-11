from agents.tools.db.mdb import MongoDBConnector
from agents.tools.states.agent_market_analysis_state import MarketAnalysisAgentState, MarketVolatilityIndex
import os
import logging
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MarketVolatilityTool(MongoDBConnector):
    def __init__(self, uri=None, database_name=None, collection_name=None):
        super().__init__(uri, database_name)
        self.collection_name = collection_name or os.getenv("YFINANCE_TIMESERIES_COLLECTION", "yfinanceMarketData")
        self.collection = self.get_collection(self.collection_name)
        logger.info("MarketVolatilityTool initialized")

    def get_most_recent_value(self):
        """
        Get the most recent value for the VIX.
        """
        result = list(self.collection.find({"symbol": "VIX"}).sort("timestamp", -1).limit(1))
        return result[0] if len(result) > 0 else None

    def get_previous_value(self, current_date):
        """
        Get the previous value for the VIX before the given date.
        """
        # Convert current_date to datetime
        current_datetime = datetime.combine(current_date, datetime.min.time())
        result = list(self.collection.find({
            "symbol": "VIX",
            "timestamp": {"$lt": current_datetime}
        }).sort("timestamp", -1).limit(1))
        return result[0] if len(result) > 0 else None

    def assess_vix(self, state: MarketAnalysisAgentState) -> dict:
        """
        Assess the VIX and update the state with the fluctuation and diagnosis.
        """
        logger.info("[Tool] Market Volatility - VIX")
        logger.info("[Action] Assessing VIX...")
        vix_data = self.get_most_recent_value()
        if not vix_data:
            logger.warning("No VIX data available.")
            vix_assess = MarketVolatilityIndex(
                volatility_index=None,
                fluctuation_answer="No VIX data available.",
                diagnosis="No change"
            )
            state.report.market_volatility_index = vix_assess
            state.updates.append("[Action] Assessed VIX: No VIX data available.")
            return { "market_volatility_index": vix_assess, "updates": state.updates, "next_step": state.next_step }

        # Extract the current VIX value and date
        current_vix_value = round(vix_data["close"], 2)
        current_vix_date = vix_data["timestamp"].strftime("%Y-%m-%d")

        # Get the previous VIX value
        previous_vix_data = self.get_previous_value(vix_data["timestamp"].date())
        if not previous_vix_data:
            logger.warning("Not enough VIX data to assess.")
            vix_assess = MarketVolatilityIndex(
                volatility_index=str(current_vix_value),
                fluctuation_answer="Not enough VIX data to assess.",
                diagnosis="No change"
            )
            state.report.market_volatility_index = vix_assess
            state.updates.append("[Action] Assessed VIX: Not enough VIX data to assess.")
            return { "market_volatility_index": vix_assess, "updates": state.updates, "next_step": state.next_step }

        # Extract the previous VIX value and date
        previous_vix_value = round(previous_vix_data["close"], 2)
        previous_vix_date = previous_vix_data["timestamp"].strftime("%Y-%m-%d")

        # Calculate fluctuation and percentage change
        fluctuation = round(current_vix_value - previous_vix_value, 2)
        percentage_change = round((fluctuation / previous_vix_value) * 100, 2)

        # Determine the diagnosis based on the VIX value
        if current_vix_value > 20:
            diagnosis = "Reduce Equity assets"
        elif current_vix_value < 12:
            diagnosis = "Increase Equity assets"
        else:
            diagnosis = "No change"

        # Create the fluctuation answer
        fluctuation_answer = (
            f"VIX close price is {current_vix_value:.2f} (reported on: {current_vix_date}), "
            f"previous close price value was: {previous_vix_value:.2f} (reported on: {previous_vix_date}), "
            f"percentage change: {percentage_change:.2f}%."
        )

        # Create a MarketVolatilityIndex object
        vix_assess = MarketVolatilityIndex(
            volatility_index="VIX",
            fluctuation_answer=fluctuation_answer,
            diagnosis=diagnosis
        )

        # Update the state
        state.report.market_volatility_index = vix_assess
        state.updates.append(f"[Action] Assessed VIX")

        # Set the next step in the state
        state.next_step = "portfolio_overall_diagnosis_node"

        return { "market_volatility_index": vix_assess, "updates": state.updates, "next_step": state.next_step }


# Initialize the MarketVolatilityTool
market_volatility_tool = MarketVolatilityTool()

# Define tools
def assess_vix_tool(state: MarketAnalysisAgentState) -> dict:
    """
    Assess the VIX and update the state with the fluctuation and diagnosis.
    """
    return market_volatility_tool.assess_vix(state=state)

if __name__ == "__main__":
    from states.agent_market_analysis_state import MarketAnalysisAgentState

    # Initialize the state
    state = MarketAnalysisAgentState()

    # Use the tool to assess the VIX
    vix_assess = assess_vix_tool(state)

    # Print the updated state
    print("\nUpdated State:")
    print(state)
