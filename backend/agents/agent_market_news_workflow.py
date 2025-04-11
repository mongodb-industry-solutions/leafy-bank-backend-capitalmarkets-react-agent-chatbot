from agent_market_news_graph import create_workflow_graph
from agents.tools.states.agent_market_news_state import MarketNewsAgentState
from agents.tools.persist_report import PersistReportInMongoDB

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if __name__ == '__main__':
    # Initial state for the workflow
    initial_state = MarketNewsAgentState(
        portfolio_allocation=[],  # Initialize as an empty list
        report={
            "asset_news": [],  # Initialize as an empty list
            "asset_news_summary": [],  # Initialize as an empty list
            "overall_news_diagnosis": None  # No diagnosis at the start
        },
        next_step="portfolio_allocation_node",  # Start with the portfolio allocation node
        updates=["Starting the market news workflow."]  # Initial update message
    )
    
    # Create the workflow graph
    graph = create_workflow_graph()
    final_state = graph.invoke(input=initial_state)

    # Print the final state
    print("\nFinal State:")
    print(final_state)

    # Add this after processing the workflow and obtaining final_state
    # Get the collection name from environment variables
    reports_market_news_coll = os.getenv("REPORTS_COLLECTION_MARKET_NEWS", "reports_market_news")

    # Persist the final state to MongoDB
    # Initialize the PersistReportInMongoDB class
    persist_data = PersistReportInMongoDB(collection_name=reports_market_news_coll)

    # Save the market news report
    persist_data.save_market_news_report(final_state)
    print("Market news report saved to MongoDB.")
