from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class PortfolioAllocation(BaseModel):
    asset: Optional[str] = Field(None, description="The asset symbol (e.g., SPY, QQQ).")
    description: Optional[str] = Field(None, description="A description of the asset.")
    allocation_percentage: Optional[str] = Field(None, description="The allocation percentage of the asset.")


class AssetNews(BaseModel):
    asset: Optional[str] = Field(None, description="The asset symbol (e.g., SPY, QQQ).")
    headline: Optional[str] = Field(None, description="The headline of the news article.")
    description: Optional[str] = Field(None, description="A brief description of the news article.")
    source: Optional[str] = Field(None, description="The source of the news article.")
    posted: Optional[str] = Field(None, description="When the news article was posted.")
    link: Optional[str] = Field(None, description="The link to the news article.")
    sentiment_score: Optional[float] = Field(None, description="The sentiment score of the article between 0 and 1.")
    sentiment_category: Optional[str] = Field(None, description="The category of the sentiment (e.g., positive, negative, neutral).")


class AssetNewsSummary(BaseModel):
    asset: Optional[str] = Field(None, description="The asset symbol (e.g., SPY, QQQ).")
    summary: Optional[str] = Field(None, description="A summary of the news articles related to the asset.")
    overall_sentiment_score: Optional[float] = Field(None, description="The overall sentiment score of the asset.")
    overall_sentiment_category: Optional[str] = Field(None, description="The overall sentiment category of the asset.")
    article_count: Optional[int] = Field(None, description="The number of articles related to the asset.")


class Report(BaseModel):
    asset_news: List[AssetNews] = Field(default_factory=list, description="A list of news articles related to the assets.")
    asset_news_summary: List[AssetNewsSummary] = Field(default_factory=list, description="A summary of news articles related to the assets.")
    overall_news_diagnosis: Optional[str] = Field(None, description="The overall news diagnosis for the portfolio.")


class MarketNewsAgentState(BaseModel):
    portfolio_allocation: List[PortfolioAllocation] = Field(default_factory=list, description="The portfolio allocation details.")
    report: Report = Field(default_factory=Report, description="The report containing analysis results.")
    next_step: Literal["__start__", "portfolio_allocation_node", "fetch_market_news_node", "asset_news_summary_node", "__end__"] = Field(None, description="The next step in the workflow (e.g., 'portfolio_allocation_node').")
    updates: List[str] = Field(default_factory=list, description="A list of updates or messages for the workflow.")