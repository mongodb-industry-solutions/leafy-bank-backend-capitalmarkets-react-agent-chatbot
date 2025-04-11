from agent_market_analysis_graph import create_workflow_graph
from agents.tools.states.agent_market_analysis_state import MarketAnalysisAgentState
from agents.tools.persist_report import PersistReportInMongoDB

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if __name__ == '__main__':
    # Initial state for the workflow
    initial_state = MarketAnalysisAgentState(
        portfolio_allocation=[],  # Initialize as an empty list
        report={
            "asset_trends": [],  # Initialize as an empty list
            "macro_indicators": [],  # Initialize as an empty list
            "market_volatility_index": {},  # Initialize as an empty MarketVolatilityIndex
            "overall_diagnosis": None  # No diagnosis at the start
        },
        next_step="portfolio_allocation_node",  # Start with the portfolio allocation node
        updates=["Starting the market analysis workflow."]  # Initial update message
    )
    
    # Create the workflow graph
    graph = create_workflow_graph()
    final_state = graph.invoke(input=initial_state)

    # Print the final state
    print("\nFinal State:")
    print(final_state)

    reports_market_analysis_coll = os.getenv("REPORTS_COLLECTION_MARKET_ANALYSIS", "reports_market_analysis")

    # Persist the final state to MongoDB
    # Initialize the PersistReportInMongoDB class
    persist_data = PersistReportInMongoDB(collection_name=reports_market_analysis_coll)
    # Save the market analysis report
    persist_data.save_market_analysis_report(final_state)
    print("Market analysis report saved to MongoDB.")
