from typing import List
from agents.tools.db.mdb import MongoDBConnector
from agents.tools.states.agent_market_analysis_state import MarketAnalysisAgentState, MacroIndicator
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MacroIndicatorsTool(MongoDBConnector):
    def __init__(self, uri=None, database_name=None, collection_name=None):
        super().__init__(uri, database_name)
        self.collection_name = collection_name or os.getenv("PYFREDAPI_COLLECTION", "pyfredapiMacroeconomicIndicators")
        self.collection = self.get_collection(self.collection_name)
        logger.info("MacroIndicatorsTool initialized")

    def get_most_recent_value(self, series_id: str):
        """
        Get the most recent value for a given series_id.
        """
        result = list(self.collection.find({"series_id": series_id}).sort("date", -1).limit(1))
        return result[0] if len(result) > 0 else None

    def assess_macro_indicator(self, series_id: str, indicator_name: str, state: MarketAnalysisAgentState, rules: dict) -> MacroIndicator:
        """
        Assess a macroeconomic indicator and update the state with the diagnosis.
        """
        logger.info(f"[Action] Assessing {indicator_name}...")
        current_data = self.get_most_recent_value(series_id)
        if not current_data:
            logger.warning(f"No {indicator_name} data available.")
            return MacroIndicator(
                macro_indicator=indicator_name,
                fluctuation_answer=f"No {indicator_name} data available.",
                diagnosis="No change"
            )

        # Get the previous value
        previous_data = list(self.collection.find({"series_id": series_id, "date": {"$lt": current_data["date"]}}).sort("date", -1).limit(1))
        if len(previous_data) == 0:
            logger.warning(f"Not enough {indicator_name} data to assess.")
            return MacroIndicator(
                macro_indicator=indicator_name,
                fluctuation_answer=f"Not enough {indicator_name} data to assess.",
                diagnosis="No change"
            )

        # Calculate fluctuation
        previous_value = round(previous_data[0]["value"], 2)
        current_value = round(current_data["value"], 2)
        fluctuation = round(current_value - previous_value, 2)

        # Determine the fluctuation answer
        if current_value > previous_value:
            fluctuation_answer = f"{indicator_name} is up by +{fluctuation:.2f} with respect to the previous period."
            diagnosis = rules["up"]
        elif current_value < previous_value:
            fluctuation_answer = f"{indicator_name} is down by -{abs(fluctuation):.2f} with respect to the previous period."
            diagnosis = rules["down"]
        else:
            fluctuation_answer = f"{indicator_name} is neutral with respect to the previous period."
            diagnosis = rules["neutral"]

        # Create a MacroIndicator object
        macro_indicator = MacroIndicator(
            macro_indicator=indicator_name,
            fluctuation_answer=fluctuation_answer,
            diagnosis=diagnosis
        )

        # Update the state
        state.report.macro_indicators.append(macro_indicator)
        state.updates.append(f"[Action] Assessed {indicator_name}")

        return macro_indicator

    def assess_macro_indicators(self, state: MarketAnalysisAgentState) -> dict:
        """
        Assess all macroeconomic indicators and update the state.
        """
        message = "[Tool] Assess macroeconomic indicators."
        logger.info(message)

        # Append the message to the updates list
        state.updates.append(message)

        # Define rules for each macroeconomic indicator
        rules = {
            "GDP": {
                "up": "Increase Equity assets",
                "down": "Increase Bond assets",
                "neutral": "No change"
            },
            "REAINTRATREARAT10Y": {  # Interest Rate
                "up": "Increase Bond assets",
                "down": "Increase Real Estate assets",
                "neutral": "No change"
            },
            "UNRATE": {  # Unemployment Rate
                "up": "Reduce Equity assets",
                "down": "Increase Equity assets",
                "neutral": "No change"
            }
        }

        # Assess each macroeconomic indicator
        indicators = [
            {"series_id": "GDP", "name": "GDP"},
            {"series_id": "REAINTRATREARAT10Y", "name": "Interest Rate"},
            {"series_id": "UNRATE", "name": "Unemployment Rate"}
        ]

        macro_indicators = []
        for indicator in indicators:
            macro_indicator = self.assess_macro_indicator(
                series_id=indicator["series_id"],
                indicator_name=indicator["name"],
                state=state,
                rules=rules[indicator["series_id"]]
            )
            macro_indicators.append(macro_indicator)

        # Set the next step in the state
        state.next_step = "market_volatility_node"

        return { "macro_indicators": macro_indicators, "updates": state.updates, "next_step": state.next_step }


# Initialize the MacroIndicatorsTool
macro_indicators_tool = MacroIndicatorsTool()

# Define tools
def assess_macro_indicators_tool(state: MarketAnalysisAgentState) -> dict:
    """
    Assess all macroeconomic indicators and update the state.
    """
    return macro_indicators_tool.assess_macro_indicators(state=state)

if __name__ == "__main__":

    # Initialize the state
    state = MarketAnalysisAgentState()

    # Use the tool to assess macroeconomic indicators
    macro_indicators = assess_macro_indicators_tool(state)

    # Print the updated state
    print("\nUpdated State:")
    print(state)

    # Print the macro indicators
    print("\nMacro Indicators:")
    for indicator in macro_indicators:
        print(indicator)