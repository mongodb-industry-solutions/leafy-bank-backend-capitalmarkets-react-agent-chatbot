import boto3
from botocore.config import Config
import os

from typing import Optional

import os
from dotenv import load_dotenv

load_dotenv()


class BedrockClient:
    """Implementation of BedrockClient class."""
    
    def __init__(self, region_name: Optional[str] = os.getenv("AWS_REGION"), aws_access_key: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID"), 
                 aws_secret_key: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY"), assumed_role: Optional[str] = None) -> None:
        """Initialize BedrockClient class.
        
        Args:
            region_name (str): AWS region name. Default is os.getenv("AWS_REGION").
            aws_access_key (str): AWS access key. Default is os.getenv("AWS_ACCESS_KEY_ID").
            aws_secret_key (str): AWS secret key. Default is os.getenv("AWS_SECRET_ACCESS_KEY").
            assumed_role (str): AWS assumed role. Default is None.
        """
        self.region_name = region_name
        self.assumed_role = assumed_role
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
    
    def _get_bedrock_client(
            self,
            runtime: Optional[bool] = True,
    ):
        """Create a boto3 client for Amazon Bedrock, with optional configuration overrides."""
        if self.region_name is None:
            target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
        else:
            target_region = self.region_name
        session_kwargs = {"region_name": target_region}
        client_kwargs = {**session_kwargs}
        
        profile_name = os.environ.get("AWS_PROFILE")
        
        if profile_name:
            session_kwargs["profile_name"] = profile_name
        
        retry_config = Config(
            region_name=target_region,
            retries={
                "max_attempts": 10,
                "mode": "standard",
            },
        )
        session = boto3.Session(**session_kwargs)
        
        if self.assumed_role:
            sts = session.client("sts")
            response = sts.assume_role(
                RoleArn=str(self.assumed_role),
                RoleSessionName="bedrock-admin"
            )
            client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
            client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
            client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]
        
        if self.aws_access_key and self.aws_secret_key:
            client_kwargs["aws_access_key_id"] = self.aws_access_key
            client_kwargs["aws_secret_access_key"] = self.aws_secret_key
        
        service_name = 'bedrock-runtime' if runtime else 'bedrock'
        
        bedrock_client = session.client(
            service_name=service_name,
            config=retry_config,
            **client_kwargs
        )
    
        return bedrock_client
    
    def _close_bedrock(self):
        """Close Bedrock client."""
        if hasattr(self, 'bedrock') and self.bedrock:
            self.bedrock.close()
    
    def __del__(self):
        """Destructor."""
        self._close_bedrock()


# Example usage of the BedrockClient class.
if __name__ == '__main__':

    # If you are not going to use BedrockClient and its models, you might remove the packages boto3 and botocore. If so:
    # Open a Terminal and run the following commands:
    # 1. cd backend ---> (Make sure to be in the backend directory)
    # 2. poetry remove boto3 botocore ---> (This will remove the packages from the project)

    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region_name = os.getenv("AWS_REGION")

    # Example usage of the BedrockClient class.
    client = BedrockClient(
        aws_access_key=aws_access_key,
        aws_secret_key=aws_secret_key,
        region_name=region_name
    )._get_bedrock_client()

    print(type(client))
    print(client)