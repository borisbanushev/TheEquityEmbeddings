import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Dict, Any, Optional
import logging
import os
from dotenv import load_dotenv

# Import the Polygon API for options data
try:
    from src.polygonapi import PolygonOptionsChainAPI
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    logger.warning("Polygon API not available - Module D features will be skipped")

# Setup logging
logger = logging.getLogger(__name__)
load_dotenv()

class DataValidator:
    """
    DataValidator handles cleaning, normalization, and outlier detection.
    """

    @staticmethod
    def clean_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace infinite values with NaN and handle missing data.
        """
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    @staticmethod
    def fill_missing(df: pd.DataFrame, method: str = 'median') -> pd.DataFrame:
        """
        Fill missing values using specified method.
        """
        if method == 'median':
            return df.fillna(df.median())
        elif method == 'mean':
            return df.fillna(df.mean())
        return df

    @staticmethod
    def normalize(df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Normalize features using StandardScaler or MinMaxScaler.
        """
        scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)


class SectorEmbedder:
    """
    Creates learned entity embeddings for sectors instead of one-hot encoding.
    This is CRITICAL - one-hot creates 90% zero vectors which is terrible.
    """

    def __init__(self, embedding_dim: int = 3):
        self.embedding_dim = embedding_dim
        self.embeddings = {}

    def fit_transform(self, sectors: pd.Series) -> pd.DataFrame:
        """
        Create sector embeddings using a simple hash-based approach.
        In production, you'd train these, but for now we use deterministic hashing.
        """
        unique_sectors = sectors.dropna().unique()

        # Create deterministic embeddings for each sector
        np.random.seed(42)
        for sector in unique_sectors:
            # Use sector ID to create a deterministic but diverse embedding
            seed = int(sector) if pd.notna(sector) else 0
            np.random.seed(seed)
            self.embeddings[sector] = np.random.randn(self.embedding_dim) * 0.5

        # Handle NaN sector
        self.embeddings[np.nan] = np.zeros(self.embedding_dim)

        # Transform all sectors
        embeddings_list = []
        for sector in sectors:
            if sector in self.embeddings:
                embeddings_list.append(self.embeddings[sector])
            else:
                embeddings_list.append(np.zeros(self.embedding_dim))

        embeddings_array = np.array(embeddings_list)

        # Create DataFrame with proper column names
        columns = [f'sector_emb_{i}' for i in range(self.embedding_dim)]
        return pd.DataFrame(embeddings_array, index=sectors.index, columns=columns)


class FeatureEngineer:
    """
    FeatureEngineer calculates the "Master 80" features across six modules.
    """

    def __init__(self, financial_data: Dict[str, pd.DataFrame]):
        """
        Args:
            financial_data: Dictionary containing 'income', 'balance', 'cashflow' DataFrames
        """
        self.income = financial_data['income']
        self.balance = financial_data['balance']
        self.cashflow = financial_data['cashflow']
        self.validator = DataValidator()

        # Initialize Polygon API for options data
        self.polygon_api = None
        if POLYGON_AVAILABLE:
            api_key = os.getenv('POLYGON_API_KEY')
            if api_key:
                self.polygon_api = PolygonOptionsChainAPI(api_key)
                logger.info("Polygon API initialized for options data")
            else:
                logger.warning("POLYGON_API_KEY not found - Module D will be skipped")

        # Merge all statements for easier calculations
        self._merge_statements()

        # Cache for market prices (to avoid fetching multiple times)
        self.price_cache = {}

    def _merge_statements(self):
        """
        Merge income, balance, and cashflow statements.
        Also calculate historical metrics where possible.
        """
        # Group by ticker and get recent + historical data
        self.income_by_ticker = self.income.groupby('Ticker')
        self.balance_by_ticker = self.balance.groupby('Ticker')
        self.cashflow_by_ticker = self.cashflow.groupby('Ticker')

        # Get most recent data for each ticker
        income_recent = self.income.groupby('Ticker').last()
        balance_recent = self.balance.groupby('Ticker').last()
        cashflow_recent = self.cashflow.groupby('Ticker').last()

        # Merge all statements
        self.data = income_recent.join(balance_recent, how='outer', rsuffix='_bal')
        self.data = self.data.join(cashflow_recent, how='outer', rsuffix='_cf')

        logger.info(f"Merged financial data for {len(self.data)} tickers")

    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Safely divide two series, replacing inf/nan appropriately."""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
        return result.replace([np.inf, -np.inf], np.nan)

    def _calculate_cagr(self, ticker: str, column: str, periods: int = 3) -> float:
        """Calculate Compound Annual Growth Rate over N periods."""
        try:
            if ticker not in self.income_by_ticker.groups:
                return np.nan

            group = self.income.loc[ticker].sort_index()
            if len(group) < periods + 1:
                return np.nan

            values = group[column].dropna()
            if len(values) < periods + 1:
                return np.nan

            ending_value = values.iloc[-1]
            beginning_value = values.iloc[-(periods + 1)]

            if beginning_value <= 0 or ending_value <= 0:
                return np.nan

            cagr = (ending_value / beginning_value) ** (1 / periods) - 1
            return cagr
        except:
            return np.nan

    def _get_current_price(self, ticker: str) -> float:
        """
        Get current market price for a ticker.
        Uses Polygon API (cached to avoid repeated calls).
        """
        if ticker in self.price_cache:
            return self.price_cache[ticker]

        if self.polygon_api:
            try:
                price = self.polygon_api.get_latest_stock_price(ticker)
                self.price_cache[ticker] = price
                return price
            except Exception as e:
                logger.warning(f"Failed to get price for {ticker}: {e}")
                return np.nan
        return np.nan

    def _get_shares_outstanding(self, ticker: str) -> float:
        """Get shares outstanding from financial data."""
        try:
            # Try to get from most recent income statement
            if 'Shares (Basic)' in self.data.columns:
                shares = self.data.loc[ticker, 'Shares (Basic)']
                if pd.notna(shares) and shares > 0:
                    return shares

            # Try diluted shares as fallback
            if 'Shares (Diluted)' in self.data.columns:
                shares = self.data.loc[ticker, 'Shares (Diluted)']
                if pd.notna(shares) and shares > 0:
                    return shares

            return np.nan
        except:
            return np.nan

    def calculate_module_a(self) -> pd.DataFrame:
        """
        Module A: Core Profitability & Management (10 features)
        """
        logger.info("Calculating Module A features (Profitability)...")
        features = pd.DataFrame(index=self.data.index)

        # 1. Operating Margin: Operating Income / Revenue
        if 'Operating Income (Loss)' in self.data.columns and 'Revenue' in self.data.columns:
            features['Operating_Margin'] = self._safe_divide(
                self.data['Operating Income (Loss)'],
                self.data['Revenue']
            )

        # 2. Net Profit Margin: Net Income / Revenue
        if 'Net Income' in self.data.columns and 'Revenue' in self.data.columns:
            features['Net_Profit_Margin'] = self._safe_divide(
                self.data['Net Income'],
                self.data['Revenue']
            )

        # 3. EBITDA Margin - approximate as (Operating Income + D&A) / Revenue
        # Note: SimFin doesn't directly provide EBITDA, we approximate
        if 'Operating Income (Loss)' in self.data.columns and 'Revenue' in self.data.columns:
            # Rough approximation - in practice you'd add back D&A
            features['EBITDA_Margin'] = self._safe_divide(
                self.data['Operating Income (Loss)'],
                self.data['Revenue']
            ) * 1.15  # Rough adjustment

        # 4. Gross Margin: Gross Profit / Revenue
        if 'Gross Profit' in self.data.columns and 'Revenue' in self.data.columns:
            features['Gross_Margin'] = self._safe_divide(
                self.data['Gross Profit'],
                self.data['Revenue']
            )

        # 5. SGA-to-Revenue: Operating overhead
        if 'Selling, General & Admin' in self.data.columns and 'Revenue' in self.data.columns:
            features['SGA_to_Revenue'] = self._safe_divide(
                self.data['Selling, General & Admin'],
                self.data['Revenue']
            )

        # 6. ROIC: Net Income / Invested Capital (approximated)
        # Invested Capital ≈ Total Assets - Current Liabilities
        if all(col in self.data.columns for col in ['Net Income', 'Total Assets', 'Total Current Liabilities']):
            invested_capital = self.data['Total Assets'] - self.data['Total Current Liabilities']
            features['ROIC'] = self._safe_divide(self.data['Net Income'], invested_capital)

        # 7. ROE: Net Income / Total Equity
        if 'Net Income' in self.data.columns and 'Total Equity' in self.data.columns:
            features['ROE'] = self._safe_divide(
                self.data['Net Income'],
                self.data['Total Equity']
            )

        # 8. ROA: Net Income / Total Assets
        if 'Net Income' in self.data.columns and 'Total Assets' in self.data.columns:
            features['ROA'] = self._safe_divide(
                self.data['Net Income'],
                self.data['Total Assets']
            )

        # 9. Asset Turnover: Revenue / Total Assets
        if 'Revenue' in self.data.columns and 'Total Assets' in self.data.columns:
            features['Asset_Turnover'] = self._safe_divide(
                self.data['Revenue'],
                self.data['Total Assets']
            )

        # 10. Research Intensity: R&D / Revenue
        if 'Research & Development' in self.data.columns and 'Revenue' in self.data.columns:
            features['RnD_Intensity'] = self._safe_divide(
                self.data['Research & Development'],
                self.data['Revenue']
            )

        logger.info(f"Module A: {len(features.columns)} features calculated")
        return features

    def calculate_module_b(self) -> pd.DataFrame:
        """
        Module B: Solvency & Capital Structure (10 features)
        """
        logger.info("Calculating Module B features (Solvency)...")
        features = pd.DataFrame(index=self.data.index)

        # 1. Altman Z-Score (for manufacturing companies)
        # Z = 1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*MVE/TL + 1.0*Sales/TA
        # Simplified version without market value
        if all(col in self.data.columns for col in ['Total Assets', 'Total Liabilities']):
            wc = self.data.get('Total Current Assets', 0) - self.data.get('Total Current Liabilities', 0)
            re = self.data.get('Retained Earnings', self.data.get('Total Equity', 0) * 0.5)  # Approximation
            ebit = self.data.get('Operating Income (Loss)', 0)
            sales = self.data.get('Revenue', 0)
            ta = self.data['Total Assets']

            z_score = (
                1.2 * self._safe_divide(wc, ta) +
                1.4 * self._safe_divide(re, ta) +
                3.3 * self._safe_divide(ebit, ta) +
                1.0 * self._safe_divide(sales, ta)
            )
            features['Altman_Z_Score'] = z_score

        # 2. Current Ratio: Current Assets / Current Liabilities
        if 'Total Current Assets' in self.data.columns and 'Total Current Liabilities' in self.data.columns:
            features['Current_Ratio'] = self._safe_divide(
                self.data['Total Current Assets'],
                self.data['Total Current Liabilities']
            )

        # 3. Quick Ratio: (Current Assets - Inventory) / Current Liabilities
        if all(col in self.data.columns for col in ['Total Current Assets', 'Inventories', 'Total Current Liabilities']):
            quick_assets = self.data['Total Current Assets'] - self.data['Inventories'].fillna(0)
            features['Quick_Ratio'] = self._safe_divide(
                quick_assets,
                self.data['Total Current Liabilities']
            )

        # 4. Debt-to-Equity: Total Debt / Total Equity
        if 'Total Debt' in self.data.columns and 'Total Equity' in self.data.columns:
            features['Debt_to_Equity'] = self._safe_divide(
                self.data['Total Debt'],
                self.data['Total Equity']
            )

        # 5. Net Debt/EBITDA
        if 'Total Debt' in self.data.columns:
            cash = self.data.get('Cash, Cash Equivalents & Short Term Investments', 0)
            net_debt = self.data['Total Debt'] - cash
            ebitda_approx = self.data.get('Operating Income (Loss)', 0) * 1.15
            features['Net_Debt_to_EBITDA'] = self._safe_divide(net_debt, ebitda_approx)

        # 6. Interest Coverage: EBIT / Interest Expense
        if 'Operating Income (Loss)' in self.data.columns and 'Interest Expense, Net' in self.data.columns:
            features['Interest_Coverage'] = self._safe_divide(
                self.data['Operating Income (Loss)'],
                self.data['Interest Expense, Net'].abs()
            )

        # 7. Dividend Payout Ratio: Dividends / Net Income
        if 'Dividends Paid' in self.data.columns and 'Net Income' in self.data.columns:
            features['Dividend_Payout_Ratio'] = self._safe_divide(
                self.data['Dividends Paid'].abs(),
                self.data['Net Income']
            )

        # 8. Shareholder Equity Ratio: Total Equity / Total Assets
        if 'Total Equity' in self.data.columns and 'Total Assets' in self.data.columns:
            features['Equity_Ratio'] = self._safe_divide(
                self.data['Total Equity'],
                self.data['Total Assets']
            )

        # 9. Days Sales Outstanding (DSO): (Accounts Receivable / Revenue) * 365
        if 'Accounts & Notes Receivable' in self.data.columns and 'Revenue' in self.data.columns:
            features['DSO'] = self._safe_divide(
                self.data['Accounts & Notes Receivable'],
                self.data['Revenue']
            ) * 365

        # 10. Cash Conversion Cycle (simplified): DSO + DIO - DPO
        # DIO = (Inventory / COGS) * 365
        # DPO = (Accounts Payable / COGS) * 365
        if all(col in self.data.columns for col in ['Inventories', 'Cost of Revenue', 'Accounts Payable']):
            dio = self._safe_divide(self.data['Inventories'], self.data['Cost of Revenue']) * 365
            dpo = self._safe_divide(self.data['Accounts Payable'], self.data['Cost of Revenue']) * 365
            dso = features.get('DSO', 0)
            features['Cash_Conversion_Cycle'] = dso + dio - dpo

        logger.info(f"Module B: {len(features.columns)} features calculated")
        return features

    def calculate_module_c(self) -> pd.DataFrame:
        """
        Module C: Growth & Valuation (10 features)
        Now includes price-based metrics using Polygon API!
        """
        logger.info("Calculating Module C features (Growth & Valuation)...")
        features = pd.DataFrame(index=self.data.index)

        # 1 & 2. Revenue & EPS CAGR (3Y) - requires historical data
        revenue_cagr = []
        for ticker in self.data.index:
            cagr = self._calculate_cagr(ticker, 'Revenue', periods=3)
            revenue_cagr.append(cagr)
        features['Revenue_CAGR_3Y'] = revenue_cagr

        # EPS CAGR - using Net Income as proxy
        eps_cagr = []
        for ticker in self.data.index:
            cagr = self._calculate_cagr(ticker, 'Net Income', periods=3)
            eps_cagr.append(cagr)
        features['EPS_CAGR_3Y'] = eps_cagr

        # 3. Rule of 40: Revenue Growth % + EBITDA Margin %
        if 'Revenue_CAGR_3Y' in features.columns:
            ebitda_margin = self.data.get('Operating Income (Loss)', 0) / self.data.get('Revenue', 1) * 1.15
            features['Rule_of_40'] = features['Revenue_CAGR_3Y'] * 100 + ebitda_margin * 100

        # 10. Free Cash Flow (calculate first, used below)
        if 'Net Cash from Operating Activities' in self.data.columns:
            if 'Net Change in Property, Plant & Equipment' in self.data.columns:
                capex = self.data['Net Change in Property, Plant & Equipment'].abs()
            else:
                capex = 0
            features['Free_Cash_Flow'] = self.data['Net Cash from Operating Activities'] - capex

        # 4-9. Price-based metrics - NOW CALCULATED with Polygon API!
        if self.polygon_api:
            logger.info("Fetching current prices for price-based metrics...")

            price_metrics = []
            for ticker in self.data.index:
                metrics = {
                    'Price_to_Sales': np.nan,
                    'Price_to_Book': np.nan,
                    'PE_Ratio': np.nan,
                    'PEG_Ratio': np.nan,
                    'EV_to_EBITDA': np.nan,
                    'FCF_Yield': np.nan
                }

                try:
                    # Get current price
                    price = self._get_current_price(ticker)
                    shares = self._get_shares_outstanding(ticker)

                    if pd.notna(price) and pd.notna(shares) and price > 0 and shares > 0:
                        market_cap = price * shares

                        # Price-to-Sales
                        revenue = self.data.loc[ticker, 'Revenue']
                        if pd.notna(revenue) and revenue > 0:
                            metrics['Price_to_Sales'] = market_cap / revenue

                        # Price-to-Book
                        book_value = self.data.loc[ticker, 'Total Equity']
                        if pd.notna(book_value) and book_value > 0:
                            metrics['Price_to_Book'] = market_cap / book_value

                        # P/E Ratio
                        net_income = self.data.loc[ticker, 'Net Income']
                        if pd.notna(net_income) and net_income > 0:
                            metrics['PE_Ratio'] = market_cap / net_income

                            # PEG Ratio = P/E / Growth Rate
                            growth = features.loc[ticker, 'EPS_CAGR_3Y']
                            if pd.notna(growth) and growth > 0:
                                metrics['PEG_Ratio'] = metrics['PE_Ratio'] / (growth * 100)

                        # EV/EBITDA
                        ebitda_approx = self.data.loc[ticker].get('Operating Income (Loss)', 0) * 1.15
                        debt = self.data.loc[ticker].get('Total Debt', 0)
                        cash = self.data.loc[ticker].get('Cash, Cash Equivalents & Short Term Investments', 0)
                        enterprise_value = market_cap + debt - cash

                        if pd.notna(ebitda_approx) and ebitda_approx > 0:
                            metrics['EV_to_EBITDA'] = enterprise_value / ebitda_approx

                        # FCF Yield
                        if 'Free_Cash_Flow' in features.columns:
                            fcf = features.loc[ticker, 'Free_Cash_Flow']
                            if pd.notna(fcf) and market_cap > 0:
                                metrics['FCF_Yield'] = fcf / market_cap

                except Exception as e:
                    logger.debug(f"Error calculating price metrics for {ticker}: {e}")

                price_metrics.append(metrics)

            # Add to features DataFrame
            price_df = pd.DataFrame(price_metrics, index=self.data.index)
            features = pd.concat([features, price_df], axis=1)

            logger.info(f"Module C: {len(features.columns)} features calculated (WITH price data!)")
        else:
            # No Polygon API - set to NaN
            features['Price_to_Sales'] = np.nan
            features['Price_to_Book'] = np.nan
            features['PE_Ratio'] = np.nan
            features['PEG_Ratio'] = np.nan
            features['EV_to_EBITDA'] = np.nan
            features['FCF_Yield'] = np.nan
            logger.warning("Module C: Price-based metrics skipped (Polygon API not available)")

        return features

    def calculate_module_d(self) -> pd.DataFrame:
        """
        Module D: "Smart Money" & Options Greeks (12 features)
        Requires Polygon options chain data.
        """
        logger.info("Calculating Module D features (Options & Smart Money)...")
        features = pd.DataFrame(index=self.data.index)

        if not self.polygon_api:
            logger.warning("Module D skipped - Polygon API not available")
            # Return empty features with NaN
            for col in ['GEX', 'Spot_Gamma_Pct', 'Put_Call_OI_Ratio', 'Put_Call_Volume_Ratio',
                       'IV_Rank_252D', 'Option_Skew_25D', 'Call_Wall_Distance', 'Put_Wall_Distance']:
                features[col] = np.nan
            return features

        logger.info("Fetching options data for all tickers (this may take a while)...")

        for ticker in self.data.index:
            ticker_metrics = {
                'GEX': np.nan,
                'Spot_Gamma_Pct': np.nan,
                'Put_Call_OI_Ratio': np.nan,
                'Put_Call_Volume_Ratio': np.nan,
                'IV_Rank_252D': np.nan,
                'Option_Skew_25D': np.nan,
                'Call_Wall_Distance': np.nan,
                'Put_Wall_Distance': np.nan
            }

            try:
                # Fetch options chain (30-day timeframe for good liquidity)
                options_df = self.polygon_api.fetch_options_chain(ticker, timeframe_days=30)

                if options_df.empty:
                    logger.debug(f"No options data for {ticker}")
                    features = pd.concat([features, pd.DataFrame([ticker_metrics], index=[ticker])])
                    continue

                # Get current stock price
                current_price = self._get_current_price(ticker)
                if pd.isna(current_price) or current_price <= 0:
                    features = pd.concat([features, pd.DataFrame([ticker_metrics], index=[ticker])])
                    continue

                # Separate calls and puts
                calls = options_df[options_df['details.contract_type'] == 'call']
                puts = options_df[options_df['details.contract_type'] == 'put']

                # 1 & 2. Aggregate GEX and Spot Gamma %
                if 'greeks.gamma' in options_df.columns and 'day.open_interest' in options_df.columns:
                    # GEX = sum(gamma * open_interest * 100 * strike^2)
                    # This is dealer gamma exposure
                    options_df['gamma_exposure'] = (
                        options_df['greeks.gamma'] *
                        options_df['day.open_interest'] *
                        100 *
                        options_df['details.strike_price'] ** 2
                    )
                    total_gex = options_df['gamma_exposure'].sum()
                    ticker_metrics['GEX'] = total_gex

                    # Spot Gamma % = GEX / Market Cap
                    shares = self._get_shares_outstanding(ticker)
                    if pd.notna(shares) and shares > 0:
                        market_cap = current_price * shares
                        ticker_metrics['Spot_Gamma_Pct'] = abs(total_gex) / market_cap

                # 3 & 4. Put/Call Ratios
                if 'day.open_interest' in options_df.columns:
                    put_oi = puts['day.open_interest'].sum()
                    call_oi = calls['day.open_interest'].sum()
                    if call_oi > 0:
                        ticker_metrics['Put_Call_OI_Ratio'] = put_oi / call_oi

                if 'day.volume' in options_df.columns:
                    put_vol = puts['day.volume'].sum()
                    call_vol = calls['day.volume'].sum()
                    if call_vol > 0:
                        ticker_metrics['Put_Call_Volume_Ratio'] = put_vol / call_vol

                # 5. IV Rank (252D) - approximated
                # Would need historical IV data; we'll use current IV percentile as proxy
                if 'implied_volatility' in options_df.columns:
                    iv_values = options_df['implied_volatility'].dropna()
                    if len(iv_values) > 0:
                        current_iv = iv_values.median()
                        # Rough approximation: normalize to 0-100 scale
                        ticker_metrics['IV_Rank_252D'] = min(current_iv * 100, 100)

                # 6. Option Skew (25D): Put IV / Call IV at 25-delta equivalent
                # Simplified: compare OTM put IV to OTM call IV
                if 'implied_volatility' in options_df.columns:
                    # OTM puts (strike < spot)
                    otm_puts = puts[puts['details.strike_price'] < current_price * 0.95]
                    # OTM calls (strike > spot)
                    otm_calls = calls[calls['details.strike_price'] > current_price * 1.05]

                    if not otm_puts.empty and not otm_calls.empty:
                        put_iv = otm_puts['implied_volatility'].median()
                        call_iv = otm_calls['implied_volatility'].median()
                        if pd.notna(call_iv) and call_iv > 0:
                            ticker_metrics['Option_Skew_25D'] = put_iv / call_iv

                # 7 & 8. Call/Put Walls (strikes with max gamma)
                if 'greeks.gamma' in options_df.columns and 'day.open_interest' in options_df.columns:
                    # Call wall: strike with max call gamma * OI
                    calls['gamma_oi'] = calls['greeks.gamma'] * calls['day.open_interest']
                    if not calls.empty:
                        call_wall_strike = calls.loc[calls['gamma_oi'].idxmax(), 'details.strike_price']
                        ticker_metrics['Call_Wall_Distance'] = (call_wall_strike - current_price) / current_price

                    # Put wall: strike with max put gamma * OI
                    puts['gamma_oi'] = puts['greeks.gamma'] * puts['day.open_interest']
                    if not puts.empty:
                        put_wall_strike = puts.loc[puts['gamma_oi'].idxmax(), 'details.strike_price']
                        ticker_metrics['Put_Wall_Distance'] = (current_price - put_wall_strike) / current_price

                logger.debug(f"Calculated options metrics for {ticker}")

            except Exception as e:
                logger.debug(f"Error calculating options metrics for {ticker}: {e}")

            # Add ticker metrics to features
            features = pd.concat([features, pd.DataFrame([ticker_metrics], index=[ticker])])

        logger.info(f"Module D: {len(features.columns)} features calculated")
        return features

    def calculate_module_f(self, metadata: pd.DataFrame = None) -> pd.DataFrame:
        """
        Module F: Syntactic & Categorical (using entity embeddings, NOT one-hot!)
        """
        logger.info("Calculating Module F features (Categorical)...")
        features = pd.DataFrame(index=self.data.index)

        # 1. Operating Leverage: % Change EBIT / % Change Sales
        # Requires historical data - simplified for now
        features['Operating_Leverage'] = np.nan

        # 2. Financial Leverage: % Change EPS / % Change EBIT
        features['Financial_Leverage'] = np.nan

        # 3. Burn Rate Runway: Cash / Net Loss (for growth firms)
        if 'Cash, Cash Equivalents & Short Term Investments' in self.data.columns and 'Net Income' in self.data.columns:
            net_loss = -self.data['Net Income'].where(self.data['Net Income'] < 0, 0)
            features['Burn_Rate_Runway'] = self._safe_divide(
                self.data['Cash, Cash Equivalents & Short Term Investments'],
                net_loss.replace(0, np.nan)
            )

        # 4. Sector Embeddings (3D) - CRITICAL: Use entity embeddings, NOT one-hot!
        if metadata is not None:
            if metadata.index.name == 'Ticker':
                meta_aligned = metadata.reindex(self.data.index)
            else:
                meta_aligned = metadata.set_index('Ticker').reindex(self.data.index)

            # Find industry column
            industry_col = None
            for col in ['IndustryId', 'industryid', 'Sector', 'sector']:
                if col in meta_aligned.columns:
                    industry_col = col
                    break

            if industry_col:
                embedder = SectorEmbedder(embedding_dim=3)
                sector_embeddings = embedder.fit_transform(meta_aligned[industry_col])
                features = pd.concat([features, sector_embeddings], axis=1)
                logger.info(f"Added 3D sector embeddings (NOT one-hot!)")
            else:
                logger.warning("No industry column found for sector embeddings")

        logger.info(f"Module F: {len(features.columns)} features calculated")
        return features

    def calculate_interaction_terms(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Module F: Add interaction terms (combinations of important ratios)
        """
        logger.info("Calculating interaction terms...")
        interactions = pd.DataFrame(index=features_df.index)

        # Example interactions (add 10-15 most important)
        if 'ROE' in features_df.columns and 'Debt_to_Equity' in features_df.columns:
            interactions['ROE_x_Leverage'] = features_df['ROE'] * features_df['Debt_to_Equity']

        if 'ROIC' in features_df.columns and 'Revenue_CAGR_3Y' in features_df.columns:
            interactions['ROIC_x_Growth'] = features_df['ROIC'] * features_df['Revenue_CAGR_3Y']

        if 'Current_Ratio' in features_df.columns and 'Quick_Ratio' in features_df.columns:
            interactions['Liquidity_Spread'] = features_df['Current_Ratio'] - features_df['Quick_Ratio']

        if 'Operating_Margin' in features_df.columns and 'Asset_Turnover' in features_df.columns:
            interactions['DuPont_Component'] = features_df['Operating_Margin'] * features_df['Asset_Turnover']

        if 'Free_Cash_Flow' in features_df.columns and 'Net Income' in self.data.columns:
            interactions['FCF_Quality'] = self._safe_divide(
                features_df['Free_Cash_Flow'],
                self.data['Net Income']
            )

        logger.info(f"Added {len(interactions.columns)} interaction terms")
        return interactions

    def engineer_all(self, metadata: pd.DataFrame = None) -> pd.DataFrame:
        """
        Run all feature engineering modules and return the final feature matrix.
        """
        logger.info("="*70)
        logger.info("ENGINEERING 'MASTER 80' FEATURE SET")
        logger.info("="*70)

        # Calculate all modules
        df_a = self.calculate_module_a()
        df_b = self.calculate_module_b()
        df_c = self.calculate_module_c()
        df_d = self.calculate_module_d()
        df_f = self.calculate_module_f(metadata)

        # Combine all features
        combined = pd.concat([df_a, df_b, df_c, df_d, df_f], axis=1)

        # Add interaction terms
        interactions = self.calculate_interaction_terms(combined)
        combined = pd.concat([combined, interactions], axis=1)

        logger.info(f"\nFeature counts by module:")
        logger.info(f"  Module A (Profitability):    {len(df_a.columns)} features")
        logger.info(f"  Module B (Solvency):         {len(df_b.columns)} features")
        logger.info(f"  Module C (Growth/Valuation): {len(df_c.columns)} features")
        logger.info(f"  Module D (Options/Greeks):   {len(df_d.columns)} features")
        logger.info(f"  Module F (Categorical):      {len(df_f.columns)} features")
        logger.info(f"  Interaction Terms:           {len(interactions.columns)} features")
        logger.info(f"\nTotal BEFORE cleaning: {len(combined.columns)} features")

        # Clean, fill, and normalize
        combined = self.validator.clean_ratios(combined)
        combined = self.validator.fill_missing(combined)
        combined = self.validator.normalize(combined)

        # Remove columns with all NaN
        combined = combined.dropna(axis=1, how='all')

        logger.info(f"Total AFTER cleaning:  {len(combined.columns)} features")
        logger.info(f"Final shape: {combined.shape[0]} tickers × {combined.shape[1]} features")
        logger.info("="*70)

        return combined
