from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class PortfolioAllocation(BaseModel):
    asset: Optional[str] = Field(None, description="The asset symbol (e.g., SPY, QQQ).")
    description: Optional[str] = Field(None, description="A description of the asset.")
    allocation_percentage: Optional[str] = Field(None, description="The allocation percentage of the asset.")


class AssetTrend(BaseModel):
    asset: Optional[str] = Field(None, description="The asset symbol (e.g., SPY, QQQ).")
    fluctuation_answer: Optional[str] = Field(None, description="A description of the asset's fluctuation.")
    diagnosis: Optional[str] = Field(None, description="A diagnosis or recommendation based on the trend.")


class MacroIndicator(BaseModel):
    macro_indicator: Optional[str] = Field(None, description="The macroeconomic indicator symbol.")
    fluctuation_answer: Optional[str] = Field(None, description="A description of the macroeconomic indicator's fluctuation.")
    diagnosis: Optional[str] = Field(None, description="A diagnosis or recommendation based on the macroeconomic indicator.")


class MarketVolatilityIndex(BaseModel):
    volatility_index: Optional[str] = Field(None, description="The market volatility index value.")
    fluctuation_answer: Optional[str] = Field(None, description="A description of the market volatility.")
    diagnosis: Optional[str] = Field(None, description="A diagnosis or recommendation based on the volatility index.")


class Report(BaseModel):
    asset_trends: List[AssetTrend] = Field(default_factory=list, description="A list of asset trends.")
    macro_indicators: List[MacroIndicator] = Field(default_factory=list, description="A list of macroeconomic indicators and their values.") 
    market_volatility_index: MarketVolatilityIndex = Field(default_factory=MarketVolatilityIndex, description="Details about the market volatility index.")
    overall_diagnosis: Optional[str] = Field(None, description="A general diagnosis of the portfolio.")


class MarketAnalysisAgentState(BaseModel):
    portfolio_allocation: List[PortfolioAllocation] = Field(default_factory=list, description="The portfolio allocation details.")
    report: Report = Field(default_factory=Report, description="The report containing analysis results.")
    next_step: Literal["__start__", "portfolio_allocation_node", "asset_trends_node","macro_indicators_node", "market_volatility_node", "portfolio_overall_diagnosis_node", "__end__"] = Field(None, description="The next step in the workflow (e.g., 'portfolio_allocation_node').")
    updates: List[str] = Field(default_factory=list, description="A list of updates or messages for the workflow.")