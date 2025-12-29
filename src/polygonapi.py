"""
Polygon.io Options Chain API

This module provides a class-based API for fetching options chain data from Polygon.io
for use in hedge strategy analysis.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from polygon import RESTClient
from dataclasses import asdict, is_dataclass
from typing import Optional
import logging

logger = logging.getLogger('thaita.server')


class PolygonOptionsChainAPI:
    """
    API for fetching options chain data from Polygon.io.

    This class provides methods to fetch complete options chains with pricing data
    for a given ticker symbol.
    """

    def __init__(self, api_key: str):
        """
        Initialize the Polygon Options Chain API.

        Args:
            api_key: Polygon.io API key
        """
        self.polygon_client = RESTClient(api_key)

    def get_latest_stock_price(self, ticker: str) -> float:
        """
        Get the latest stock price for a ticker using Polygon API.
        Falls back to yfinance if Polygon fails.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Latest stock price, or 0 if unable to fetch
        """
        # Try Polygon first (more reliable and we already have the API key)
        try:
            logger.info(f"Fetching stock price for {ticker} from Polygon.io")
            # Get the latest trade for the stock
            aggs = list(self.polygon_client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=datetime.now() - timedelta(days=5),  # Get last 5 days
                to=datetime.now(),
                limit=5
            ))
            
            if aggs:
                # Get the most recent close price
                price = float(aggs[-1].close)
                if price > 0:
                    logger.info(f"Got price for {ticker} from Polygon: ${price}")
                    return price
        except Exception as e:
            logger.warning(f"Polygon price fetch failed for {ticker}: {e}, trying yfinance...")
        
        # Fallback to yfinance
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Try multiple possible keys for the current price
            price_keys = ['regularMarketPrice', 'currentPrice', 'price', 'previousClose']
            for key in price_keys:
                if key in info and info[key] is not None:
                    price = float(info[key])
                    if price > 0:
                        logger.info(f"Got price for {ticker} from yfinance key '{key}': ${price}")
                        return price
            
            # If no price found in info, try using history
            logger.warning(f"Price not found in info dict for {ticker}, trying history...")
            hist = stock.history(period='1d')
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                logger.info(f"Got price for {ticker} from yfinance history: ${price}")
                return price
            
            logger.error(f"Unable to fetch price for {ticker} - no valid price sources")
            return 0
        except Exception as e:
            logger.error(f"Error fetching stock price for {ticker} from both Polygon and yfinance: {e}")
            return 0

    def flatten_dict(self, d: dict, parent_key: str = '', sep: str = '.') -> dict:
        """
        Recursively flattens nested dictionaries (including dataclasses).

        Args:
            d: Dictionary to flatten
            parent_key: Parent key for nested items
            sep: Separator between nested keys

        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if is_dataclass(v):
                v = asdict(v)
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def records_to_dataframe(self, records: list) -> pd.DataFrame:
        """
        Converts list of OptionContractSnapshots into a DataFrame (each item is a row).

        Args:
            records: List of option contract snapshots from Polygon

        Returns:
            DataFrame with flattened option data
        """
        flattened_records = []
        for record in records:
            if is_dataclass(record):
                record = asdict(record)
            flattened = self.flatten_dict(record)
            flattened_records.append(flattened)

        df = pd.DataFrame(flattened_records)
        return df

    def fetch_options_chain(
        self,
        ticker: str,
        timeframe_days: int = 30,
        min_days_to_expiry: int = 0,
        strike_price_min_pct: float = 0.6,
        strike_price_max_pct: float = 1.1
    ) -> pd.DataFrame:
        """
        Fetch complete options chain for a ticker with pricing data.

        Args:
            ticker: Stock ticker symbol
            timeframe_days: User-selected timeframe in days (determines max_days_to_expiry)
            min_days_to_expiry: Minimum days until expiration
            strike_price_min_pct: Minimum strike price as % of current price (default: 0.6)
            strike_price_max_pct: Maximum strike price as % of current price (default: 1.1)

        Returns:
            DataFrame with options chain data including pricing and Greeks
        """
        # Calculate min and max days to expiry based on timeframe
        # This ensures we only download options in the relevant date range
        if timeframe_days <= 7:
            # Immediate: Get only the NEXT available expiration
            # Download 1-10 days, then filter to nearest expiration
            min_days_to_expiry = 1
            max_days_to_expiry = 10
        elif timeframe_days <= 30:
            # Soon: Get options expiring within the next 30 days
            min_days_to_expiry = 1
            max_days_to_expiry = 30
        else:  # timeframe_days == 75 (near future: 2-3 months)
            # Near Future: Get options expiring in 60-90 days
            min_days_to_expiry = 60
            max_days_to_expiry = 90

        logger.info(f"Fetching options chain for {ticker} from Polygon.io")
        logger.info(f"Timeframe: {timeframe_days} days → downloading expirations {min_days_to_expiry}-{max_days_to_expiry} days out")

        # Get current stock price
        current_price = self.get_latest_stock_price(ticker)
        if current_price == 0:
            logger.error(f"Unable to fetch stock price for {ticker}")
            return pd.DataFrame()

        logger.info(f"Current stock price for {ticker}: ${current_price}")

        # Calculate date range
        today = datetime.now()
        min_expiry = (today + timedelta(days=min_days_to_expiry)).strftime('%Y-%m-%d')
        max_expiry = (today + timedelta(days=max_days_to_expiry)).strftime('%Y-%m-%d')

        # Calculate strike price range
        min_strike = current_price * strike_price_min_pct
        max_strike = current_price * strike_price_max_pct

        logger.info(f"Fetching options: expiry {min_expiry} to {max_expiry}, strikes ${min_strike:.2f} to ${max_strike:.2f}")

        # Fetch options chain with snapshots (includes pricing data)
        options_chain = []
        try:
            for o in self.polygon_client.list_snapshot_options_chain(
                ticker,
                params={
                    "expiration_date.gte": min_expiry,
                    "expiration_date.lte": max_expiry,
                    "strike_price.gte": min_strike,
                    "strike_price.lte": max_strike,
                },
            ):
                options_chain.append(o)
        except Exception as e:
            logger.error(f"Error calling Polygon API for {ticker}: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            # Try to get more details
            if hasattr(e, 'status_code'):
                logger.error(f"Status code: {e.status_code}")
            if hasattr(e, 'response'):
                logger.error(f"Response: {e.response}")
            return pd.DataFrame()

        logger.info(f"Fetched {len(options_chain)} option contracts from Polygon")

        if not options_chain:
            # FALLBACK: If no options found in target range, try a wider range
            # This handles cases where stocks don't have long-dated options
            if timeframe_days > 30 and min_days_to_expiry > 30:
                logger.warning(f"No options found in {min_days_to_expiry}-{max_days_to_expiry} day range")
                logger.info(f"Trying fallback: fetching options 1-{max_days_to_expiry} days out")

                # Retry with wider range starting from 1 day
                min_expiry_fallback = (today + timedelta(days=1)).strftime('%Y-%m-%d')

                try:
                    options_chain = []
                    for o in self.polygon_client.list_snapshot_options_chain(
                        ticker,
                        params={
                            "expiration_date.gte": min_expiry_fallback,
                            "expiration_date.lte": max_expiry,
                            "strike_price.gte": min_strike,
                            "strike_price.lte": max_strike,
                        },
                    ):
                        options_chain.append(o)

                    logger.info(f"Fallback: Fetched {len(options_chain)} option contracts")

                    if not options_chain:
                        logger.warning(f"No options found even with fallback range")
                        logger.warning(f"  1. No options contracts exist for this ticker")
                        logger.warning(f"  2. Market is closed and no snapshot data available")
                        logger.warning(f"  3. Polygon API rate limit exceeded")
                        return pd.DataFrame()

                except Exception as e:
                    logger.error(f"Error in fallback fetch: {e}")
                    return pd.DataFrame()
            else:
                logger.warning(f"No options found for {ticker} - this could mean:")
                logger.warning(f"  1. No options contracts exist in the specified range")
                logger.warning(f"  2. Market is closed and no snapshot data available")
                logger.warning(f"  3. Polygon API rate limit exceeded")
                logger.warning(f"  4. Invalid ticker symbol")
                return pd.DataFrame()

        # Convert to DataFrame
        options_chain_df = self.records_to_dataframe(options_chain)
        
        if options_chain_df.empty:
            logger.error(f"DataFrame is empty after conversion for {ticker}")
            return pd.DataFrame()

        # Log available columns for debugging
        logger.info(f"Available columns in options_chain_df: {list(options_chain_df.columns)[:10]}...")

        # Process data with error handling
        try:
            if 'details.expiration_date' in options_chain_df.columns:
                options_chain_df['details.expiration_date'] = pd.to_datetime(options_chain_df['details.expiration_date'])
            
            # Calculate spread if quote data exists
            if 'last_quote.ask' in options_chain_df.columns and 'last_quote.bid' in options_chain_df.columns:
                options_chain_df['spread'] = options_chain_df['last_quote.ask'] - options_chain_df['last_quote.bid']
            else:
                logger.warning(f"Missing quote data for {ticker}, setting spread to 0")
                options_chain_df['spread'] = 0
            
            options_chain_df['created_at'] = int(pd.Timestamp.now().timestamp())

            # Fill NaN values
            options_chain_df = options_chain_df.fillna(0)

            # Convert types with error handling
            if 'day.volume' in options_chain_df.columns:
                options_chain_df['day.volume'] = options_chain_df['day.volume'].astype(int)
            
            options_chain_df['day.last_updated'] = 1
            
            if 'last_trade.exchange' in options_chain_df.columns:
                options_chain_df['last_trade.exchange'] = options_chain_df['last_trade.exchange'].astype(int)
            
            options_chain_df['last_trade.sip_timestamp'] = 1
            
            if 'last_trade.size' in options_chain_df.columns:
                options_chain_df['last_trade.size'] = options_chain_df['last_trade.size'].astype(int)

            # Convert datetime columns to strings
            datetime_cols = options_chain_df.select_dtypes(include=['datetime64[ns]']).columns
            for col in datetime_cols:
                options_chain_df[col] = options_chain_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

            logger.info(f"Processed options chain DataFrame with shape {options_chain_df.shape}")
            
            # Verify we have required columns
            required_cols = ['details.contract_type', 'details.strike_price']
            missing_cols = [col for col in required_cols if col not in options_chain_df.columns]
            if missing_cols:
                logger.error(f"Missing required columns for {ticker}: {missing_cols}")
                return pd.DataFrame()

            # Special filtering for "immediately" timeframe - keep only the next expiration
            if timeframe_days <= 7 and 'details.expiration_date' in options_chain_df.columns:
                # Convert expiration dates to datetime for comparison
                exp_dates = pd.to_datetime(options_chain_df['details.expiration_date'])
                nearest_expiration = exp_dates.min()

                # Filter to keep only options with the nearest expiration date
                options_chain_df = options_chain_df[exp_dates == nearest_expiration].copy()

                logger.info(f"'Immediately' timeframe: filtered to next expiration {nearest_expiration.strftime('%Y-%m-%d')}")
                logger.info(f"Remaining contracts after filtering: {len(options_chain_df)}")

            return options_chain_df
            
        except Exception as e:
            logger.error(f"Error processing options data for {ticker}: {e}")
            logger.error(f"DataFrame shape: {options_chain_df.shape}")
            logger.error(f"Columns: {list(options_chain_df.columns)}")
            return pd.DataFrame()


# For standalone testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv('POLYGON_API_KEY')
    api = PolygonOptionsChainAPI(api_key)

    ticker = 'NVDA'
    df = api.fetch_options_chain(ticker)
    print(f"Options chain shape: {df.shape}")
    print(f"\nSample data:")
    print(df.head())