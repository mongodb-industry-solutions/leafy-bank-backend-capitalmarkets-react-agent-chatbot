from agents.tools.bedrock.client import BedrockClient
from langchain_aws import ChatBedrock

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


def get_llm(model_id: str = os.getenv("CHAT_COMPLETIONS_MODEL_ID")) -> ChatBedrock:
    """
    Get an instance of the ChatBedrock class for the specified model ID and AWS credentials.

    Args:
        model_id (str): The model ID to use for the ChatBedrock instance.
    """

    bedrock_client = BedrockClient()._get_bedrock_client()

    return ChatBedrock(model=model_id,
                client=bedrock_client,
                temperature=0)
