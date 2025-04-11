import logging
from fastapi import FastAPI, Request, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from scheduled_agents import ScheduledAgents
import threading

from service_market_data import MarketDataService
from service_portfolio_data import PortfolioDataService
from service_macro_indicators_data import MacroIndicatorDataService
from service_financial_news_data import FinancialNewsDataService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

# Add these imports
from api_market_assistant import router as market_assistant_router

# Add this router to your app
app.include_router(market_assistant_router)

@app.get("/")
async def read_root(request: Request):
    return {"message": "Server is running"}

# Initialize services
scheduled_agents_service = ScheduledAgents()
market_data_service = MarketDataService()
portfolio_data_service = PortfolioDataService()
macro_indicator_data_service = MacroIndicatorDataService()
financial_news_data_service = FinancialNewsDataService()


@app.post("/execute-market-analysis-workflow")
async def execute_market_analysis_workflow():
    """
    Execute the market analysis workflow.

    Returns:
        dict: A dictionary containing the status of the workflow execution.
    """
    try:
        return scheduled_agents_service.run_agent_market_analysis_workflow()
    except Exception as e:
        logging.error(f"Error executing market analysis workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute-market-news-workflow")
async def execute_market_news_workflow():
    """
    Execute the market news workflow.

    Returns:
        dict: A dictionary containing the status of the workflow execution.
    """
    try:
        return scheduled_agents_service.run_agent_market_news_workflow()
    except Exception as e:
        logging.error(f"Error executing market news workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fetch-assets-close-price")
async def fetch_assets_close_price():
    """
    Fetch the latest close price for all assets.

    Returns:
        dict: A dictionary containing the assets and their close prices.
    """
    try:
        close_prices = market_data_service.fetch_assets_close_price()
        return close_prices
    except Exception as e:
        logging.error(f"Error fetching assets close price: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fetch-portfolio-allocation")
async def fetch_portfolio_allocation():
    """
    Fetch portfolio allocation data.

    Returns:
        dict: A dictionary containing the portfolio allocation data.
    """
    try:
        portfolio_allocation = portfolio_data_service.fetch_portfolio_allocation()
        return portfolio_allocation
    except Exception as e:
        logging.error(f"Error fetching portfolio allocation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fetch-most-recent-macro-indicators")
async def fetch_most_recent_macro_indicators():
    """
    Fetch the most recent macroeconomic indicators.

    Returns:
        dict: A dictionary containing the most recent macroeconomic indicators.
    """
    try:
        macro_indicators = macro_indicator_data_service.fetch_most_recent_macro_indicators()
        return macro_indicators
    except Exception as e:
        logging.error(f"Error fetching most recent macro indicators: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calc-overall-sentiment-for-all")
async def calc_overall_sentiment_for_all():
    """
    Calculate the overall sentiment score for all symbols.

    Returns:
        dict: A dictionary containing the symbol, overall sentiment score, category, and number of articles.
    """
    try:
        sentiment_scores = financial_news_data_service.calc_overall_sentiment_for_all()
        return sentiment_scores
    except Exception as e:
        logging.error(f"Error calculating overall sentiment for all symbols: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class SymbolRequest(BaseModel):
    symbol: str

@app.post("/calc-overall-sentiment-for-symbol")
async def calc_overall_sentiment_for_symbol(request: SymbolRequest):
    """
    Calculate the overall sentiment score for a specific symbol.

    Args:
        request (SymbolRequest): The request body containing the symbol.

    Returns:
        dict: A dictionary containing the symbol, overall sentiment score, category, and number of articles.
    """
    try:
        sentiment_score = financial_news_data_service.calc_overall_sentiment_for_symbol(request.symbol)
        return sentiment_score
    except Exception as e:
        logging.error(f"Error calculating overall sentiment for symbol {request.symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def start_scheduler():
    scheduler.start()

scheduler = ScheduledAgents()
scheduler_thread = threading.Thread(target=start_scheduler)
scheduler_thread.start()