from langgraph.graph import END, StateGraph

from agents.tools.states.agent_market_news_state import  MarketNewsAgentState
from agents.tools.tool_portfolio_allocation import check_portfolio_allocation_tool
from agents.tools.tool_asset_news import fetch_market_news_tool
from agents.tools.tool_asset_news_summary import generate_news_summaries_tool


# --- Create LangGraph StateGraph ---
def create_workflow_graph(checkpointer=None):
    """
    Create a workflow graph for the Market Analysis Agent.
    This graph defines the sequence of operations that the agent will perform to analyze the market and generate insights.
    """
    # Define the state of the agent
    graph = StateGraph(MarketNewsAgentState)

    # Define the nodes
    graph.add_node("portfolio_allocation_node", check_portfolio_allocation_tool)
    graph.add_node("fetch_market_news_node", fetch_market_news_tool)
    graph.add_node("asset_news_summary_node", generate_news_summaries_tool)
    
    # Define the edges
    graph.add_edge("portfolio_allocation_node", "fetch_market_news_node")
    graph.add_edge("fetch_market_news_node", "asset_news_summary_node")
    graph.add_edge("asset_news_summary_node", END)

    # Set the entry point
    graph.set_entry_point("portfolio_allocation_node")
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    else:
        return graph.compile()


if __name__ == '__main__':

    # Graph Compiliation and visualisation
    graph = create_workflow_graph()

    # Print the graph in ASCII format
    ascii_graph = graph.get_graph().draw_ascii()
    print(ascii_graph)