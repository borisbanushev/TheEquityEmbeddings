#!/usr/bin/env python
"""
Interactive CLI tool to query similar/dissimilar companies using the trained embeddings.

Usage:
    python query_ticker.py AAPL
    python query_ticker.py TSLA --top 10
"""

import sys
import argparse
import pandas as pd
import numpy as np
from src.search import EquitySearch
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_search_engine():
    """Load embeddings and metadata to create search engine."""
    try:
        # Load embeddings
        embeddings_df = pd.read_csv('data/processed/embeddings.csv', index_col=0)
        embeddings = embeddings_df.values

        # Load metadata (company info)
        # Try to load from the simfin cache
        import simfin as sf
        import os
        from dotenv import load_dotenv

        load_dotenv()
        sf.set_api_key(os.getenv('SIMFIN_API_KEY'))
        sf.set_data_dir('data/simfin_cache')

        companies = sf.load_companies(market='us')

        # Handle index - Ticker is the index in SimFin data
        # Reset index to make Ticker a column
        companies = companies.reset_index()

        # Lowercase ALL column names first
        companies.columns = [c.lower() for c in companies.columns]

        # Now check for ticker column (should be lowercase now)
        if 'ticker' not in companies.columns:
            # If still not found, try index column
            if 'index' in companies.columns:
                companies = companies.rename(columns={'index': 'ticker'})
            else:
                raise ValueError(f"Cannot find ticker column. Available: {list(companies.columns)}")

        # Align metadata with embeddings - keep only companies we have embeddings for
        metadata = companies[companies['ticker'].isin(embeddings_df.index)].copy()
        metadata = metadata.set_index('ticker').reindex(embeddings_df.index)

        # Reset index without keeping the old index name
        metadata = metadata.reset_index()

        # Ensure the ticker column name is lowercase after reset
        if metadata.columns[0] != 'ticker':
            metadata.columns = ['ticker'] + list(metadata.columns[1:])

        logger.info(f"Loaded {len(embeddings)} embeddings")

        # Verify ticker column exists
        if 'ticker' not in metadata.columns:
            raise ValueError(f"Metadata columns after processing: {list(metadata.columns)}")

        return EquitySearch(embeddings, metadata)

    except FileNotFoundError:
        logger.error("Embeddings not found. Please run 'python main.py' first to generate embeddings.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)


def format_company_list(df, show_distance=True):
    """Format company list for display."""
    if df.empty:
        return "No results found"

    # Select columns to display
    display_cols = []
    if 'ticker' in df.columns:
        display_cols.append('ticker')
    if 'name' in df.columns:
        display_cols.append('name')
    elif 'company name' in df.columns:
        display_cols.append('company name')
    if 'industryid' in df.columns:
        display_cols.append('industryid')
    if show_distance and 'distance' in df.columns:
        display_cols.append('distance')

    result = df[display_cols].copy()

    # Format distance if present
    if 'distance' in result.columns:
        result['distance'] = result['distance'].apply(lambda x: f"{x:.4f}")

    return result.to_string(index=False)


def query_ticker(ticker, top_k=5, show_least_similar=True):
    """Query and display similar/dissimilar companies."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Analyzing: {ticker.upper()}")
    logger.info(f"{'='*70}\n")

    search_engine = load_search_engine()

    try:
        # Get comprehensive analysis
        analysis = search_engine.analyze_ticker(ticker.upper(), k=top_k)

        # Display query company info
        query_info = analysis['query_info']
        logger.info("COMPANY INFORMATION:")
        logger.info(f"  Ticker: {query_info.get('ticker', 'N/A')}")
        logger.info(f"  Name:   {query_info.get('name', query_info.get('company name', 'N/A'))}")
        logger.info(f"  Industry ID: {query_info.get('industryid', 'N/A')}")
        logger.info("")

        # Display most similar companies
        logger.info(f"\n🔗 TOP {top_k} MOST SIMILAR COMPANIES:")
        logger.info("─" * 70)
        most_similar = analysis['most_similar']
        logger.info(format_company_list(most_similar, show_distance=True))

        if show_least_similar:
            # Display least similar companies
            logger.info(f"\n⚡ TOP {top_k} LEAST SIMILAR COMPANIES:")
            logger.info("─" * 70)
            least_similar = analysis['least_similar']
            logger.info(format_company_list(least_similar, show_distance=True))

        logger.info(f"\n{'='*70}\n")

    except ValueError as e:
        logger.error(f"Error: {e}")
        logger.info("\nAvailable tickers:")
        available = search_engine.metadata['ticker'].dropna().unique()
        logger.info(f"Total: {len(available)} companies")
        logger.info(f"Examples: {', '.join(available[:10].tolist())}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Query similar and dissimilar companies using equity embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query_ticker.py AAPL
  python query_ticker.py TSLA --top 10
  python query_ticker.py MSFT --top 3 --no-dissimilar

The tool uses the 16-dimensional embeddings generated by the autoencoder
to find companies that are structurally similar (or dissimilar) based on
their fundamental financial metrics.
        """
    )

    parser.add_argument('ticker', type=str, help='Stock ticker symbol to query')
    parser.add_argument('--top', '-k', type=int, default=5,
                        help='Number of similar/dissimilar companies to show (default: 5)')
    parser.add_argument('--no-dissimilar', action='store_true',
                        help='Only show most similar companies, not least similar')

    args = parser.parse_args()

    query_ticker(args.ticker, top_k=args.top, show_least_similar=not args.no_dissimilar)


if __name__ == "__main__":
    main()
