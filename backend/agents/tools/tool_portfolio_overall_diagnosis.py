from agents.tools.states.agent_market_analysis_state import MarketAnalysisAgentState
from agents.tools.bedrock.anthropic_chat_completions import BedrockAnthropicChatCompletions
from agents.tools.agent_profiles import AgentProfiles
from typing import Optional
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

class PortfolioOverallDiagnosisTool:
    def __init__(self, chat_completions_model_id: Optional[str] = os.getenv("CHAT_COMPLETIONS_MODEL_ID"), agent_id: Optional[str] = "MARKET_ANALYSIS_AGENT"):
        """
        PortfolioOverallDiagnosisTool class to generate overall diagnosis for the portfolio.
        This class uses the BedrockAnthropicChatCompletions model to generate a comprehensive diagnosis based on the portfolio context.
        
        Args:
            chat_completions_model_id (str): Model ID for chat completions. Default is os.getenv("CHAT_COMPLETIONS_MODEL_ID").
            agent_id (str): Agent ID. Default is "MARKET_ANALYSIS_AGENT".
        """
        self.chat_completions_model_id = chat_completions_model_id
        self.agent_id = agent_id
        logger.info("PortfolioOverallDiagnosisTool initialized")

    def generate_overall_diagnosis(self, state: MarketAnalysisAgentState) -> dict:
        """
        Generate the overall diagnosis for the portfolio using the portfolio context and LLM.

        Args:
            state (MarketAnalysisAgentState): The current state of the agent.

        Returns:
            str: The overall diagnosis of the portfolio.
        """
        logger.info("[Tool] Generate overall diagnosis for the portfolio.")

        # Retrieve the MARKET_ANALYSIS_AGENT profile
        profiler = AgentProfiles()
        agent_profile = profiler.get_agent_profile(self.agent_id)
        if not agent_profile:
            logger.error(f"Agent profile not found for agent ID: {self.agent_id}")
            state.updates.append("Unable to generate overall diagnosis due to missing agent profile.")
            return { "overall_diagnosis": "Error while retrieving agent profile!", "updates": state.updates, "next_step": state.next_step }
        else:
            # Log the agent profile
            state.updates.append(f"[Action] Using agent profile: {self.agent_id} - {agent_profile['role']}")

        # Extract portfolio context from the state
        asset_trends = state.report.asset_trends
        macro_indicators = state.report.macro_indicators
        market_volatility = state.report.market_volatility_index
        portfolio_allocation = {allocation.asset: allocation.description for allocation in state.portfolio_allocation}

        # Build the context for the LLM
        context_parts = []
        if asset_trends:
            context_parts.append("Asset Trends Analysis:")
            for trend in asset_trends:
                description = portfolio_allocation.get(trend.asset, "No description available")
                context_parts.append(f"- {trend.asset} ({description}): {trend.fluctuation_answer} Diagnosis: {trend.diagnosis}")

        if macro_indicators:
            context_parts.append("Macroeconomic Indicators Analysis:")
            for indicator in macro_indicators:
                context_parts.append(f"- {indicator.macro_indicator}: {indicator.fluctuation_answer} Diagnosis: {indicator.diagnosis}")

        if market_volatility and market_volatility.diagnosis:
            context_parts.append("Market Volatility Analysis:")
            context_parts.append(f"- VIX: {market_volatility.fluctuation_answer} Diagnosis: {market_volatility.diagnosis}")

        portfolio_context = "\n".join(context_parts)

        # Generate the LLM prompt
        llm_prompt = (
            f"You are an AI assistant for a market analysis agent. "
            f"Your task is to provide a comprehensive overall diagnosis of the portfolio based on the following context:\n\n"
            f"Role: {agent_profile['role']}\n"
            f"Kind of Data: {agent_profile['kind_of_data']}\n"
            f"Instructions: {agent_profile['instructions']}\n\n"
            f"Rules: {agent_profile['rules']}\n\n"
            f"Portfolio Context:\n{portfolio_context}\n\n"
            f"Based on the above context, provide a comprehensive overall diagnosis of the portfolio."
        )

        logger.info("LLM Prompt for Overall Diagnosis:")
        logger.info(llm_prompt)

        # Simulate LLM response (replace this with actual LLM integration)
        try:
            # Instantiate the chat completion model
            chat_completions = BedrockAnthropicChatCompletions(model_id=self.chat_completions_model_id)
            # Generate a chain of thought based on the prompt
            overall_diagnosis = chat_completions.predict(llm_prompt)
            # Log the LLM response
            if not overall_diagnosis:
                overall_diagnosis = "No diagnosis generated."
            logger.info("LLM Response for Overall Diagnosis:")
            logger.info(overall_diagnosis)
        except Exception as e:
            logger.error(f"Error generating overall diagnosis: {e}")
            overall_diagnosis = "Unable to generate overall diagnosis at this time."

        # Update the state with the overall diagnosis
        state.report.overall_diagnosis = overall_diagnosis
        state.updates.append("[Tool] Generated overall diagnosis for the portfolio.")
        state.next_step = "__end__"
        
        return { "overall_diagnosis": overall_diagnosis, "updates": state.updates, "next_step": state.next_step }

# Initialize the PortfolioOverallDiagnosisTool
portfolio_overall_diagnosis_tool = PortfolioOverallDiagnosisTool()

# Define tools
def generate_overall_diagnosis_tool(state: MarketAnalysisAgentState) -> dict:
    """
    Generate the overall diagnosis for the portfolio and update the state.

    Args:
        state (MarketAnalysisAgentState): The current state of the agent.

    Returns:
        str: The overall diagnosis of the portfolio.
    """
    return portfolio_overall_diagnosis_tool.generate_overall_diagnosis(state=state)

if __name__ == "__main__":
    from states.agent_market_analysis_state import MarketAnalysisAgentState, PortfolioAllocation, AssetTrend, MacroIndicator, MarketVolatilityIndex

    # Example usage with realistic portfolio context
    state = MarketAnalysisAgentState(
        portfolio_allocation=[
            PortfolioAllocation(asset="SPY", description="S&P 500 ETF", allocation_percentage="25%"),
            PortfolioAllocation(asset="QQQ", description="Nasdaq ETF", allocation_percentage="20%"),
        ],
        report={
            "asset_trends": [
                AssetTrend(asset="SPY", fluctuation_answer="SPY close price is 537.64, and its MA50 is 538.97.", diagnosis="It may indicate a downtrend."),
                AssetTrend(asset="QQQ", fluctuation_answer="QQQ close price is 450.73, and its MA50 is 451.88.", diagnosis="It may indicate a downtrend."),
            ],
            "macro_indicators": [
                MacroIndicator(macro_indicator="US GDP", fluctuation_answer="US GDP is up by +4.21 with respect to the previous period.", diagnosis="Increase Equity assets"),
                MacroIndicator(macro_indicator="US Interest Rate", fluctuation_answer="US Interest Rate is down by -0.17 with respect to the previous period.", diagnosis="Increase Real Estate assets"),
            ],
            "market_volatility_index": MarketVolatilityIndex(
                volatility_index="29.96",
                fluctuation_answer="VIX close price is 29.96 (reported on: 2025-04-03), previous close price value was: 21.51 (reported on: 2025-04-02), percentage change: 39.28%.",
                diagnosis="Reduce Equity assets"
            ),
            "overall_diagnosis": None,
        },
        updates=["[Tool] Calculate asset trends.", "[Tool] Assessed macroeconomic indicators.", "[Tool] Assessed VIX."]
    )

    # Use the tool to generate the overall diagnosis
    overall_diagnosis = generate_overall_diagnosis_tool(state)

    # Print the updated state
    print("\nUpdated State:")
    print(state.model_dump_json(indent=4))

    # Print the overall diagnosis
    print("\nOverall Diagnosis:")
    print(overall_diagnosis)