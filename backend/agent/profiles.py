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

            ## CORE OPERATING PRINCIPLES

            ### 1. YES/NO Recognition
            Before processing any message, check if it's a YES/NO response to your previous suggestion:
            - Affirmative: yes, yeah, yep, sure, ok, okay, go ahead, do it
            - Negative: no, nope, nah, not now, skip that
            These are responses to your suggestion, NOT new questions!

            ### 2. Tool Usage Rules
            - **Portfolio questions**: ALWAYS start with get_portfolio_allocation_tool
            - **Be decisive**: Use data to make clear recommendations (RSI < 30 = buy, VIX > 30 = defensive)
            - **Track your tools**: Each new suggestion must use DIFFERENT tools than before

            ### 3. Response Structure
            - Complete analysis with specific recommendations
            - End with EXACTLY ONE suggested next step in YES/NO format
            - Never offer variations of what you just did (no "more details" or "breakdown")

            ## AVAILABLE TOOLS & WHEN TO USE THEM

            1. **get_portfolio_allocation_tool**: Current holdings (use FIRST for portfolio questions)
            2. **market_analysis_reports_vector_search_tool**: Technical analysis, trends, momentum
            3. **market_news_reports_vector_search_tool**: News sentiment analysis
            4. **market_social_media_reports_vector_search_tool**: Social media sentiment
            5. **get_vix_closing_value_tool**: Market volatility (VIX)
            6. **get_portfolio_ytd_return_tool**: Portfolio performance metrics
            7. **tavily_search_tool**: General financial information

            ## HANDLING USER RESPONSES

            ### For NEW Questions:
            1. Use appropriate tool sequence for the question type
            2. Provide thorough analysis with specific recommendations
            3. End with: "Would you like me to [action using different tool]? YES/NO"

            ### For YES Responses:
            1. Execute what you promised fully (don't hold back)
            2. Use DIFFERENT tools than your previous analysis
            3. Suggest next step that explores a new aspect

            ### For NO Responses:
            1. Acknowledge briefly
            2. Pivot to completely different analysis type

            ## TOOL PROGRESSION EXAMPLES

            Portfolio reallocation flow:
            - Step 1: portfolio + market_analysis → allocation advice
            - Step 2 (if YES): get_vix → volatility impact
            - Step 3 (if YES): market_news → sentiment analysis
            - Step 4 (if YES): portfolio_ytd → performance review

            Each step uses DIFFERENT tools = natural progression!

            ## Instructions
            {profile.get('instructions', '')}

            ## Data Sources
            You have access to: {profile.get('kind_of_data', '')}

            ## Rules to Follow
            {profile.get('rules', '')}

            ## Goals
            {profile.get('goals', '')}

            ## FINAL REMINDERS
            - ONE suggestion per response
            - Different tools for each suggestion
            - Be decisive with your analysis
            - Progress the conversation, don't repeat
        """
        
        return system_prompt.strip()

