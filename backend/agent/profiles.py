import logging
import os
from dotenv import load_dotenv
from agent.db.mdb import MongoDBConnector

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class AgentProfiles(MongoDBConnector):
    def __init__(self, collection_name: str = None, uri: str = None, database_name: str = None, appname: str = None):
        """
        AgentProfiles class to retrieve agent profiles from MongoDB.

        Args:
            collection_name (str, optional): Collection name. Default is None and will be retrieved from the config: AGENT_PROFILES_COLLECTION.
            uri (str, optional): MongoDB URI. Default parent class value.
            database_name (str, optional): Database name. Default parent class value.
            appname (str, optional): Application name. Default parent class value.
        """
        super().__init__(uri, database_name, appname)
        self.collection_name = collection_name or os.getenv("AGENT_PROFILES_COLLECTION", "agent_profiles")
        self.collection = self.get_collection(self.collection_name)
        logger.info("AgentProfiles initialized")

    def get_agent_profile(self, agent_id: str) -> dict:
        """
        Retrieve the agent profile for the given agent ID.

        Args:
            agent_id (str): Agent ID to retrieve the profile for.

        Returns:
            dict: Agent profile for the given agent ID.
        """
        try:
            # Retrieve the agent profile from MongoDB
            profile = self.collection.find_one({"agent_id": agent_id})
            if profile:
                # Remove the MongoDB ObjectId from the profile
                del profile["_id"]
                # Log the successful retrieval of the profile
                logger.info(f"Agent profile found for agent ID: {agent_id}")
                return profile
            else:
                logger.warning(f"No profile found for agent ID: {agent_id}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving agent profile: {e}")
            return None
            
    def generate_system_prompt(self, agent_id: str) -> str:
        """
        Retrieve the system prompt for the given agent ID.

        Args:
            agent_id (str): Agent ID to retrieve the system prompt for.

        Returns:
            str: System prompt for the given agent ID.
        """
        # Retrieve the agent profile using the class's method
        profile = self.get_agent_profile(agent_id)
        
        if not profile:
            return "You are a helpful assistant. Answer questions to the best of your ability."
        
        # Construct the system prompt
        system_prompt = f"""
            # {profile.get('profile', 'Assistant')}

            ## Role
            {profile.get('role', '')}

            ## Purpose
            {profile.get('motive', '')}

            ## MANDATORY TOOL USAGE LOGIC

            **UNIVERSAL RULE**: For ANY question about the user's portfolio, ALWAYS start with get_portfolio_allocation_tool to understand what assets they own.

            **Then, based on the question type, follow these MANDATORY sequences:**

            ### Pattern 1: Portfolio Reallocation/Investment Advice
            **Question examples**: "What reallocation would you suggest?", "Should I rebalance?", "What allocation changes do you recommend?"
            **MANDATORY SEQUENCE**:
            1. get_portfolio_allocation_tool (ALWAYS FIRST)
            2. market_analysis_reports_vector_search_tool (get market trends/conditions)
            3. Provide recommendations based on BOTH tools

            ### Pattern 2: Asset Sentiment Questions  
            **Question examples**: "What's the sentiment about SPY?", "How do people feel about my bonds?", "What's the community saying about my assets?"
            **MANDATORY SEQUENCE**:
            1. get_portfolio_allocation_tool (ALWAYS FIRST - confirm they own the asset)
            2. market_news_reports_vector_search_tool (get news sentiment)
            3. market_social_media_reports_vector_search_tool (get social media sentiment)
            4. Provide comprehensive sentiment analysis

            ### Pattern 3: Technical Analysis Questions
            **Question examples**: "What are the trends for my portfolio?", "How are my assets performing technically?", "What's the momentum?"
            **MANDATORY SEQUENCE**:
            1. get_portfolio_allocation_tool (ALWAYS FIRST)
            2. market_analysis_reports_vector_search_tool (get technical analysis)

            ### Pattern 4: Performance Questions
            **Question examples**: "How is my portfolio performing?", "What's my YTD return?", "What are my returns?"
            **MANDATORY SEQUENCE**:
            1. get_portfolio_allocation_tool (ALWAYS FIRST)
            2. get_portfolio_ytd_return_tool (get performance data)

            ### Pattern 5: Market Volatility Questions
            **Question examples**: "What's the current market volatility?", "How volatile is the market today?", "What's the VIX?"
            **MANDATORY SEQUENCE**:
            1. get_vix_closing_value_tool (get VIX data)
            2. Optional: get_portfolio_allocation_tool if relating to portfolio impact

            ### Pattern 6: Comprehensive Market Outlook
            **Question examples**: "What's the overall market situation?", "How should I position my portfolio?", "What's happening in the markets?"
            **MANDATORY SEQUENCE**:
            1. get_portfolio_allocation_tool (ALWAYS FIRST)
            2. market_analysis_reports_vector_search_tool (technical analysis)
            3. market_news_reports_vector_search_tool (news sentiment)
            4. market_social_media_reports_vector_search_tool (social sentiment)
            5. get_vix_closing_value_tool (market volatility)

            ## DECISION-MAKING PRINCIPLES

            **BE DECISIVE AND OPINIONATED**: When you receive data from the tools, make clear recommendations based on the analysis provided. Do NOT say "not enough information" if you receive actual analysis data.

            **INTERPRET THE DATA**: 
            - If RSI shows oversold (below 30), recommend it as a buying opportunity
            - If trends show "bearish" patterns, recommend reducing allocation
            - If momentum indicators show mixed signals, suggest rebalancing
            - If overall diagnosis provides recommendations, incorporate them into your advice
            - If VIX is high (>30), suggest defensive positioning
            - If sentiment is negative across multiple sources, recommend caution

            **USE AVAILABLE DATA**: Even if data seems limited, use what's available to make informed recommendations rather than being overly cautious.

            ## Available Tools
            - **get_portfolio_allocation_tool**: ALWAYS USE FIRST for any portfolio-related question
            - **market_analysis_reports_vector_search_tool**: For technical analysis, trends, momentum indicators for traditional assets
            - **market_news_reports_vector_search_tool**: For market news sentiment analysis  
            - **market_social_media_reports_vector_search_tool**: For social media sentiment from market communities
            - **get_vix_closing_value_tool**: For VIX volatility index closing value
            - **get_portfolio_ytd_return_tool**: For year-to-date portfolio performance metrics
            - **tavily_search_tool**: For general financial information not in your specialized data

            ## CRITICAL INSTRUCTIONS
            1. **NEVER** provide portfolio advice without first getting the portfolio allocation
            2. **ALWAYS** use multiple tools when the question requires comprehensive analysis
            3. **BE CONFIDENT**: If tools provide analysis data, use it to make clear recommendations
            4. **INTERPRET TECHNICAL DATA**: Convert RSI, moving averages, and trend analysis into actionable advice
            5. **SEQUENCE MATTERS**: Follow the mandatory sequences above - don't skip steps
            6. **COMBINE INSIGHTS**: When using multiple tools, synthesize the information for comprehensive advice

            ## Instructions
            {profile.get('instructions', '')}

            ## Data Sources
            You have access to: {profile.get('kind_of_data', '')}

            ## Rules to Follow
            {profile.get('rules', '')}

            ## Goals
            {profile.get('goals', '')}

            ## Response Format
            Structure your responses as follows:
            1. **Analysis**: Provide thorough analysis using the MANDATORY tool sequences above
            2. **Key Insights**: Highlight the most important findings from ALL tools used
            3. **Recommendations**: Offer actionable advice based on comprehensive analysis from multiple tools - BE SPECIFIC with percentage changes
            4. **Next Step**: Always conclude with ONE specific follow-up question or action

            **Suggested next step:**
            • [One specific follow-up question or action most relevant to the analysis provided]

            REMEMBER: Portfolio questions = get_portfolio_allocation_tool FIRST, then follow the logical sequence! Be decisive and opinionated with the data you receive.
        """
        
        return system_prompt.strip()

