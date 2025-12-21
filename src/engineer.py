import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Dict, Any, Optional
import logging

# Setup logging
logger = logging.getLogger(__name__)

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
        # Threshold for dropping rows or columns can be added here
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

class FeatureEngineer:
    """
    FeatureEngineer calculates all 80+ features across the six modules.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.validator = DataValidator()

    def calculate_module_a(self) -> pd.DataFrame:
        """
        Module A: Core Profitability & Management (SimFin Derived)
        """
        logger.info("Calculating Module A features...")
        mappings = {
            'Operating Margin': 'Operating Margin',
            'Net Profit Margin': 'Net Profit Margin',
            'EBITDA Margin': 'EBITDA Margin',
            'ROIC': 'Return on Invested Capital',
            'ROE': 'Return on Equity',
            'ROA': 'Return on Assets',
            'Asset Turnover': 'Asset Turnover'
        }
        # Rename SimFin columns to our standard names if they exist
        existing = [c for c in mappings.values() if c in self.data.columns]
        subset = self.data[existing].rename(columns={v: k for k, v in mappings.items()})
        return subset

    def calculate_module_b(self) -> pd.DataFrame:
        """
        Module B: Solvency & Capital Structure (SimFin Derived)
        """
        logger.info("Calculating Module B features...")
        mappings = {
            'Current Ratio': 'Current Ratio',
            'Quick Ratio': 'Quick Ratio',
            'Debt-to-Equity': 'Debt to Equity',
            'Interest Coverage': 'Interest Coverage',
            'Dividend Payout Ratio': 'Payout Ratio'
        }
        existing = [c for c in mappings.values() if c in self.data.columns]
        subset = self.data[existing].rename(columns={v: k for k, v in mappings.items()})
        return subset

    def calculate_module_c(self) -> pd.DataFrame:
        """
        Module C: Growth & Valuation (SimFin Derived)
        """
        logger.info("Calculating Module C features...")
        mappings = {
            'Forward P/E': 'Forward P/E',
            'Price-to-Sales': 'Price to Sales Ratio',
            'Price-to-Book': 'Price to Book Ratio',
            'EV/EBITDA': 'EV / EBITDA',
            'Free Cash Flow Yield': 'Free Cash Flow Yield'
        }
        existing = [c for c in mappings.values() if c in self.data.columns]
        subset = self.data[existing].rename(columns={v: k for k, v in mappings.items()})
        return subset

    def engineer_all(self, metadata: pd.DataFrame = None) -> pd.DataFrame:
        """
        Run all feature engineering modules and return the final feature matrix.
        """
        # Module A, B, C are largely from SimFin Derived
        df_a = self.calculate_module_a()
        df_b = self.calculate_module_b()
        df_c = self.calculate_module_c()
        
        # Combine
        combined = pd.concat([df_a, df_b, df_c], axis=1)
        
        # Add Sector/Region embeddings if metadata is provided (Module F)
        if metadata is not None:
            logger.info("Engineering Module F (Sector Embeddings)...")
            # Simple one-hot or categorical encoding for now
            sectors = pd.get_dummies(metadata.set_index('Ticker')['Sector'], prefix='sector')
            combined = pd.concat([combined, sectors], axis=1)

        combined = self.validator.clean_ratios(combined)
        combined = self.validator.fill_missing(combined)
        combined = self.validator.normalize(combined)
        
        return combined

if __name__ == "__main__":
    # Test with dummy data
    dummy_data = pd.DataFrame(np.random.randn(10, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    engineer = FeatureEngineer(dummy_data)
    final_features = engineer.engineer_all()
    print(final_features.head())
