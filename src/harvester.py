import os
import pandas as pd
import simfin as sf
from polygon import RESTClient
from dotenv import load_dotenv
from typing import List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class DataHarvester:
    """
    DataHarvester handles fetching financial data from SimFin and Polygon APIs.
    """

    def __init__(self):
        self.polygon_key = os.getenv("POLYGON_API_KEY")
        self.simfin_key = os.getenv("SIMFIN_API_KEY")
        
        if self.polygon_key:
            self.polygon_client = RESTClient(self.polygon_key)
        else:
            logger.warning("POLYGON_API_KEY not found in environment.")

        if self.simfin_key:
            sf.set_api_key(self.simfin_key)
            # Set local directory for simfin data
            sf.set_data_dir(os.path.join(os.getcwd(), 'data', 'simfin_cache'))
        else:
            logger.warning("SIMFIN_API_KEY not found in environment.")

    def fetch_sp500_tickers(self) -> List[str]:
        """
        Fetch the current list of S&P 500 tickers from Wikipedia.
        """
        logger.info("Fetching S&P 500 ticker list from Wikipedia...")
        try:
            import requests
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers)
            tables = pd.read_html(response.text)
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            # Clean tickers (Wikipedia uses '.' sometimes instead of '-' or vice versa)
            tickers = [t.replace('.', '-') for t in tickers]
            logger.info(f"Successfully fetched {len(tickers)} S&P 500 tickers.")
            return tickers
        except Exception as e:
            logger.error(f"Error fetching S&P 500 tickers: {e}")
            return []

    def fetch_simfin_data(self, tickers: List[str] = None):
        """
        Fetch base financial statements and metadata from SimFin.
        Since the 'derived' dataset is not available (500 error), we load
        the base statements (income, balance, cashflow) and will calculate
        derived metrics in the FeatureEngineer.
        """
        logger.info(f"Fetching SimFin data...")
        try:
            # Load basic company info (metadata)
            companies = sf.load_companies(market='us')
            logger.info(f"Loaded {len(companies)} companies")

            # Load base financial statements instead of derived dataset
            logger.info("Loading income statements...")
            income = sf.load_income(variant='annual', market='us')
            logger.info(f"Loaded income statements: {income.shape}, index: {income.index.names}")

            logger.info("Loading balance sheets...")
            balance = sf.load_balance(variant='annual', market='us')
            logger.info(f"Loaded balance sheets: {balance.shape}, index: {balance.index.names}")

            logger.info("Loading cash flow statements...")
            cashflow = sf.load_cashflow(variant='annual', market='us')
            logger.info(f"Loaded cash flow statements: {cashflow.shape}, index: {cashflow.index.names}")

            # Filter by tickers if specified
            if tickers:
                logger.info(f"Filtering for {len(tickers)} tickers...")
                # Ticker is the index of companies, not a column
                companies = companies[companies.index.isin(tickers)]
                # Filter financial statements by ticker (index level 0)
                # The index is a MultiIndex with ('Ticker', 'Report Date')
                income = income[income.index.get_level_values('Ticker').isin(tickers)]
                balance = balance[balance.index.get_level_values('Ticker').isin(tickers)]
                cashflow = cashflow[cashflow.index.get_level_values('Ticker').isin(tickers)]
                logger.info(f"After filtering: companies={len(companies)}, income={income.shape}, balance={balance.shape}, cashflow={cashflow.shape}")

            # Combine all statements into a single structure for easier access
            financial_data = {
                'income': income,
                'balance': balance,
                'cashflow': cashflow
            }

            return financial_data, companies
        except Exception as e:
            logger.error(f"Error fetching SimFin data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None

    def fetch_polygon_price_history(self, ticker: str, multiplier: int, timespan: str, from_date: str, to_date: str):
        """
        Fetch historical price data from Polygon.
        """
        if not self.polygon_client:
            logger.error("Polygon client not initialized.")
            return None
        
        logger.info(f"Fetching Polygon price history for {ticker}...")
        try:
            aggs = []
            for a in self.polygon_client.get_aggs(ticker, multiplier, timespan, from_date, to_date):
                aggs.append(a)
            return pd.DataFrame(aggs)
        except Exception as e:
            logger.error(f"Error fetching Polygon price history for {ticker}: {e}")
            return None

    def fetch_options_snapshot(self, ticker: str):
        """
        Fetch options market snapshot for a ticker from Polygon.
        """
        if not self.polygon_client:
            logger.error("Polygon client not initialized.")
            return None

        logger.info(f"Fetching options snapshot for {ticker}...")
        try:
            # Use snapshot or universal snapshot for options
            # This is a placeholder for the actual complex options logic
            # needed to calculate Vanna, Charm, GEX etc.
            return self.polygon_client.list_snapshot_options_chain(ticker)
        except Exception as e:
            logger.error(f"Error fetching options snapshot for {ticker}: {e}")
            return None

if __name__ == "__main__":
    # Simple test if keys are present
    harvester = DataHarvester()
    # df = harvester.fetch_simfin_fundamentals(['AAPL', 'MSFT'])
    # print(df.head())
