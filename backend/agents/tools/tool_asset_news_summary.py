import logging
from agents.tools.db.mdb import MongoDBConnector
from agents.tools.states.agent_market_news_state import MarketNewsAgentState, AssetNewsSummary
from agents.tools.bedrock.anthropic_chat_completions import BedrockAnthropicChatCompletions
from agents.tools.agent_profiles import AgentProfiles
import os
from dotenv import load_dotenv
from typing import Optional, Dict, List
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class AssetNewsSummaryTool(MongoDBConnector):
    def __init__(self, chat_completions_model_id: Optional[str] = os.getenv("CHAT_COMPLETIONS_MODEL_ID"), agent_id: Optional[str] = "MARKET_NEWS_AGENT"):
        """
        AssetNewsSummaryTool class to generate summaries and calculate sentiment metrics for news articles.
        This class uses the BedrockAnthropicChatCompletions model to generate concise summaries.
        
        Args:
            chat_completions_model_id (str): Model ID for chat completions. Default is os.getenv("CHAT_COMPLETIONS_MODEL_ID").
            agent_id (str): Agent ID. Default is "MARKET_NEWS_AGENT".
        """
        self.chat_completions_model_id = chat_completions_model_id
        self.agent_id = agent_id
        logger.info("AssetNewsSummaryTool initialized")
    
    def group_news_by_asset(self, state: MarketNewsAgentState) -> Dict[str, List]:
        """Group news articles by asset symbol."""
        asset_news_groups = defaultdict(list)
        
        for news in state.report.asset_news:
            if news.asset:
                asset_news_groups[news.asset].append(news)
        
        return asset_news_groups
    
    def calculate_sentiment_metrics(self, news_group: List) -> dict:
        """Calculate sentiment metrics for a group of news articles."""
        if not news_group:
            return {
                "overall_sentiment_score": 0.0,
                "overall_sentiment_category": "Neutral",
                "article_count": 0
            }
        
        # Calculate the average sentiment score
        sentiment_scores = [news.sentiment_score for news in news_group if news.sentiment_score is not None]
        avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        
        # Determine the overall sentiment category (should be consistent for all articles in the group)
        sentiment_categories = [news.sentiment_category for news in news_group if news.sentiment_category]
        if sentiment_categories:
            # Use the most common sentiment category
            from collections import Counter
            counter = Counter(sentiment_categories)
            overall_category = counter.most_common(1)[0][0]
        else:
            overall_category = "Neutral"
        
        return {
            "overall_sentiment_score": round(avg_score, 2),
            "overall_sentiment_category": overall_category,
            "article_count": len(news_group)
        }
    
    def generate_asset_summary(self, asset: str, description: str, news_group: List, agent_profile: dict) -> str:
        """Generate a summary for an asset's news articles using LLM."""
        # Prepare news articles content for the LLM
        news_content = []
        for i, news in enumerate(news_group, 1):
            news_content.append(f"Article {i}:")
            news_content.append(f"Headline: {news.headline}")
            news_content.append(f"Description: {news.description}")
            news_content.append(f"Sentiment: {news.sentiment_category} (Score: {news.sentiment_score})")
            news_content.append("")
        
        news_context = "\n".join(news_content)
        
        # Generate the LLM prompt
        llm_prompt = (
            f"You are an AI assistant for a market news agent. "
            f"Your task is to provide a concise summary of recent news about {asset} ({description}).\n\n"
            f"Role: {agent_profile['role']}\n"
            f"Instructions: {agent_profile['instructions']}\n\n"
            f"News Articles:\n{news_context}\n\n"
            f"Generate a concise summary (maximum 30 words) of these news articles for {asset} ({description}). "
            f"Focus on key insights and implications for investors. Be objective and factual."
        )

        logger.info(f"LLM Prompt for {asset} news summary:")
        logger.info(llm_prompt)
        
        try:
            # Instantiate the chat completion model
            chat_completions = BedrockAnthropicChatCompletions(model_id=self.chat_completions_model_id)
            # Generate summary
            summary = chat_completions.predict(llm_prompt)
            
            # Truncate if necessary to ensure it's under 50 words
            words = summary.split()
            if len(words) > 50:
                summary = " ".join(words[:50]) + "..."
                
            return summary
        except Exception as e:
            logger.error(f"Error generating summary for {asset}: {e}")
            return f"Recent news for {asset} indicates {news_group[0].sentiment_category.lower()} market sentiment."
    
    def generate_news_summaries(self, state: MarketNewsAgentState) -> dict:
        """
        Generate summaries for news articles grouped by asset.
        
        Args:
            state (MarketNewsAgentState): The current state of the market news agent.
            
        Returns:
            dict: Updated state with asset news summaries.
        """
        message = "[Tool] Generating asset news summaries."
        logger.info(message)
        
        # Retrieve the MARKET_NEWS_AGENT profile
        profiler = AgentProfiles()
        agent_profile = profiler.get_agent_profile(self.agent_id)
        if not agent_profile:
            logger.error(f"Agent profile not found for agent ID: {self.agent_id}")
            state.updates.append("Unable to generate news summaries due to missing agent profile.")
            return {"asset_news_summary": [], "updates": state.updates, "next_step": state.next_step}
        
        state.updates.append(f"[Action] Using agent profile: {self.agent_id} - {agent_profile['role']}")
        
        # Group news by asset
        asset_news_groups = self.group_news_by_asset(state)
        
        # Create asset description lookup
        asset_descriptions = {allocation.asset: allocation.description for allocation in state.portfolio_allocation}
        
        # Process each asset group
        asset_news_summaries = []
        for asset, news_group in asset_news_groups.items():
            # Calculate sentiment metrics
            metrics = self.calculate_sentiment_metrics(news_group)
            
            # Get asset description
            description = asset_descriptions.get(asset, "")
            
            # Generate summary using LLM
            summary_text = self.generate_asset_summary(asset, description, news_group, agent_profile)
            
            # Create AssetNewsSummary object
            asset_summary = AssetNewsSummary(
                asset=asset,
                summary=summary_text,
                overall_sentiment_score=metrics["overall_sentiment_score"],
                overall_sentiment_category=metrics["overall_sentiment_category"],
                article_count=metrics["article_count"]
            )
            asset_news_summaries.append(asset_summary)
        
        # Update the state with the summaries
        state.report.asset_news_summary = asset_news_summaries
        
        # Generate overall news diagnosis
        if asset_news_summaries:
            positive_count = sum(1 for s in asset_news_summaries if s.overall_sentiment_category == "Positive")
            negative_count = sum(1 for s in asset_news_summaries if s.overall_sentiment_category == "Negative")
            neutral_count = sum(1 for s in asset_news_summaries if s.overall_sentiment_category == "Neutral")
            
            if positive_count > negative_count and positive_count > neutral_count:
                overall_diagnosis = "Overall, the portfolio assets are receiving predominantly positive news coverage."
            elif negative_count > positive_count and negative_count > neutral_count:
                overall_diagnosis = "Overall, the portfolio assets are receiving predominantly negative news coverage."
            else:
                overall_diagnosis = "Overall, the portfolio assets are receiving mixed or neutral news coverage."
            
            state.report.overall_news_diagnosis = overall_diagnosis
        
        # Update state with message
        state.updates.append(message)

        # Set the next step in the state
        state.next_step = "__end__"

        return {
            "asset_news_summary": state.report.asset_news_summary,
            "overall_news_diagnosis": state.report.overall_news_diagnosis,
            "updates": state.updates,
            "next_step": state.next_step
        }

# Initialize the AssetNewsSummaryTool
asset_news_summary_obj = AssetNewsSummaryTool()

# Define tools
def generate_news_summaries_tool(state: MarketNewsAgentState) -> dict:
    """Generate summaries for news articles grouped by asset."""
    return asset_news_summary_obj.generate_news_summaries(state=state)

# Example usage
if __name__ == "__main__":
    from states.agent_market_news_state import MarketNewsAgentState, PortfolioAllocation, AssetNews, Report
    
    # Initialize the state with sample data
    state = MarketNewsAgentState(
        portfolio_allocation=[
            PortfolioAllocation(
                asset="SPY", description="S&P 500 ETF", allocation_percentage="25%"
            ),
            PortfolioAllocation(
                asset="QQQ", description="Nasdaq ETF", allocation_percentage="20%"
            )
        ],
        report=Report(
            asset_news=[
                AssetNews(asset="SPY", headline="Market Update", description="The S&P 500 rose by 1% today.", 
                          source="Financial Times", posted="2 hours ago", sentiment_score=0.8, sentiment_category="Positive"),
                AssetNews(asset="SPY", headline="Economic Outlook", description="Positive economic indicators support market growth.", 
                          source="Bloomberg", posted="5 hours ago", sentiment_score=0.7, sentiment_category="Positive"),
                AssetNews(asset="QQQ", headline="Tech Stocks Decline", description="Tech sector faces challenges amid regulatory concerns.", 
                          source="CNBC", posted="3 hours ago", sentiment_score=0.3, sentiment_category="Negative"),
            ]
        ),
        next_step="asset_news_node",
    )
    
    # Generate summaries
    result = generate_news_summaries_tool(state)

    # Print the updated state
    print("\nUpdated State:")
    print(state.model_dump_json(indent=4))