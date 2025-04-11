from agents.tools.db.mdb import MongoDBConnector
from agents.tools.states.agent_market_analysis_state import MarketAnalysisAgentState, AssetTrend
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

# Constants
# MA_PERIOD is the moving average period
MA_PERIOD = int(os.getenv("MA_PERIOD", 50))  # Default to 50 if not set in the environment

class AssetTrendsTool(MongoDBConnector):
    def __init__(self, uri=None, database_name=None, collection_name=None):
        super().__init__(uri, database_name)
        self.collection_name = collection_name or os.getenv("YFINANCE_TIMESERIES_COLLECTION", "yfinanceMarketData")
        self.collection = self.get_collection(self.collection_name)
        logger.info("AssetTrendsTool initialized")

    def calculate_moving_average(self, symbol: str, period: int = MA_PERIOD) -> float:
        """
        Calculate the moving average for a given symbol and period.
        """
        pipeline = [
            {"$match": {"symbol": symbol}},
            {"$sort": {"timestamp": -1}},
            {"$limit": period},
            {"$group": {
                "_id": None,
                "moving_average": {"$avg": "$close"}
            }}
        ]
        result = list(self.collection.aggregate(pipeline))
        return result[0]["moving_average"] if result else None

    def get_last_closing_price(self, symbol: str) -> float:
        """
        Get the last closing price for a given symbol.
        """
        result = list(self.collection.find({"symbol": symbol}).sort("timestamp", -1).limit(1))
        return result[0]["close"] if result else None

    def calculate_asset_trends(self, state: MarketAnalysisAgentState) -> dict:
        """
        Assess the trend of all symbols in the portfolio by comparing their last closing price with their moving average.
        """
        message = "[Tool] Calculate asset trends."
        logger.info(message)

        asset_trends = []
        for allocation in state.portfolio_allocation:  # Iterate over the list of PortfolioAllocation objects
            symbol = allocation.asset  # Access the "asset" field of each PortfolioAllocation object

            # Calculate the moving average
            moving_average = self.calculate_moving_average(symbol, MA_PERIOD)
            if moving_average is None:
                logger.warning(f"Not enough data to calculate the MA{MA_PERIOD} for {symbol}.")
                continue

            # Get the last closing price
            last_closing_price = self.get_last_closing_price(symbol)
            if last_closing_price is None:
                logger.warning(f"Not enough data to retrieve the last closing price for {symbol}.")
                continue

            # Compare the last closing price with the moving average
            trend = "uptrend" if last_closing_price > moving_average else "downtrend"

            # Create an AssetTrend object
            asset_trend = AssetTrend(
                asset=symbol,
                fluctuation_answer=f"{symbol} close price is {last_closing_price:.2f}, and its MA{MA_PERIOD} is {moving_average:.2f}.",
                diagnosis=f"It may indicate an {trend}."
            )
            asset_trends.append(asset_trend)

        # Update the state with the asset trends
        state.report.asset_trends = asset_trends  # Update the nested `asset_trends` field in `state.report`

        # Append the message to the updates list
        state.updates.append(message)

        # Set the next step in the state
        state.next_step = "macro_indicators_node"

        return { "asset_trends": asset_trends, "updates": state.updates, "next_step": state.next_step }


# Initialize the AssetTrendsTool
asset_trends_tool = AssetTrendsTool()

# Define tools
def calculate_asset_trends_tool(state: MarketAnalysisAgentState) -> dict:
    """
    Assess the trend of a given symbol by comparing its last closing price with its moving average.
    """
    return asset_trends_tool.calculate_asset_trends(state=state)

if __name__ == "__main__":
    from states.agent_market_analysis_state import MarketAnalysisAgentState, PortfolioAllocation

    # Initialize the state with only the fields required for the first step
    state = MarketAnalysisAgentState(
        portfolio_allocation=[
            PortfolioAllocation(
                asset="SPY", description="S&P 500 ETF", allocation_percentage="25%"
            ),
            PortfolioAllocation(
                asset="QQQ", description="Nasdaq ETF", allocation_percentage="20%"
            ),
            PortfolioAllocation(
                asset="EEM", description="Emerging Markets ETF", allocation_percentage="8%"
            ),
            PortfolioAllocation(
                asset="XLE", description="Energy Sector ETF", allocation_percentage="5%"
            ),
            PortfolioAllocation(
                asset="TLT", description="Long-Term Treasury Bonds", allocation_percentage="10%"
            ),
            PortfolioAllocation(
                asset="LQD", description="Investment-Grade Bonds", allocation_percentage="7%"
            ),
            PortfolioAllocation(
                asset="HYG", description="High-Yield Bonds", allocation_percentage="5%"
            ),
            PortfolioAllocation(
                asset="VNQ", description="Real Estate ETF", allocation_percentage="6%"
            ),
            PortfolioAllocation(
                asset="GLD", description="Gold ETF", allocation_percentage="8%"
            ),
            PortfolioAllocation(
                asset="USO", description="Oil ETF", allocation_percentage="6%"
            )
        ],
        next_step="macro_indicators_node",  # Set the next step in the workflow
    )

    # Use the tool to calculate asset trends
    trends = calculate_asset_trends_tool(state)

    # Print the updated state
    print("\nUpdated State:")
    print(state)