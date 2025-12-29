import os
import torch
import pandas as pd
import numpy as np
from src.harvester import DataHarvester
from src.engineer import FeatureEngineer
from src.model import MultiModalAutoencoder
from src.trainer import ModelTrainer
from src.search import EquitySearch
from scripts.visualize import generate_3d_market_galaxy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting EquityEmbeddings Pipeline...")
    
    # 1. Data Harvesting
    harvester = DataHarvester()
    
    # Fetch S&P 500 tickers
    ticker_list = harvester.fetch_sp500_tickers()
    if not ticker_list:
        logger.warning("Could not fetch S&P 500 tickers. Using default list.")
        ticker_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'JNJ', 'V', 'JPM', 'GS', 'MS', 'BAC', 'WFC']
    
    logger.info(f"Using universe of {len(ticker_list)} tickers.")
    financial_data, company_info = harvester.fetch_simfin_data(tickers=ticker_list)

    # --- CHECKPOINT: DATA INTEGRITY ---
    if financial_data is None or company_info is None or company_info.empty:
        logger.critical("CHECKPOINT FAILED: Real financial data could not be downloaded from SimFin.")
        logger.critical("Possible reasons: Invalid SIMFIN_API_KEY, network issues, or SimFin server error.")
        logger.critical("The pipeline will stop now to prevent processing dummy data.")
        import sys
        sys.exit(1)

    # Check that we have data
    num_tickers = len(financial_data['income'].index.get_level_values('Ticker').unique())
    logger.info(f"Fetched REAL data for {num_tickers} tickers.")

    # 2. Feature Engineering with real data
    engineer = FeatureEngineer(financial_data)
    # We pass company_info as metadata for sector embeddings
    processed_features = engineer.engineer_all(metadata=company_info)

    # Save processed features to CSV
    os.makedirs('data/processed', exist_ok=True)
    features_path = 'data/processed/processed_features.csv'
    processed_features.to_csv(features_path)
    logger.info(f"Saved processed features to {features_path}")

    # Prepare metadata for visualization/search
    metadata = company_info.copy()
    # Ticker is the index, so reset it to make it a column
    if metadata.index.name == 'Ticker':
        metadata = metadata.reset_index()
    metadata.columns = [c.lower() for c in metadata.columns] # Normalize for visualization script
    if 'name' not in metadata.columns and 'company name' in metadata.columns:
        metadata['name'] = metadata['company name']

    input_dim = processed_features.shape[1]
    
    # 3. Model Training
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    logger.info(f"Using device: {device}")
    model = MultiModalAutoencoder(input_dim=input_dim, latent_dim=16)
    trainer = ModelTrainer(model, device=device)
    
    train_tensor = torch.tensor(processed_features.values, dtype=torch.float32)
    logger.info("Training Autoencoder...")
    trainer.fit(train_tensor, epochs=20, batch_size=8) # Small batch for small dataset

    # 4. Extraction of Embeddings
    model.eval()
    with torch.no_grad():
        _, latent = model(train_tensor.to(device))
        embeddings = latent.cpu().numpy()

    # Save embeddings to CSV
    embeddings_df = pd.DataFrame(
        embeddings,
        index=processed_features.index,
        columns=[f'dim_{i}' for i in range(embeddings.shape[1])]
    )
    embeddings_path = 'data/processed/embeddings.csv'
    embeddings_df.to_csv(embeddings_path)
    logger.info(f"Saved embeddings to {embeddings_path}")

    # 5. Search Indexing
    logger.info("Building Search Index...")
    search_engine = EquitySearch(embeddings, metadata)
    
    # Use a real ticker for the example query if possible
    query_ticker = ticker_list[0] if ticker_list[0] in metadata['ticker'].values else metadata['ticker'].iloc[0]
    logger.info(f"Finding companies similar to {query_ticker}...")
    similar = search_engine.query_similar_tickers(query_ticker, k=3)

    # Display available columns
    available_cols = [c for c in ['ticker', 'name', 'sector', 'industryid'] if c in similar.columns]
    if available_cols:
        logger.info(f"Similar companies:\n{similar[available_cols]}")
    else:
        logger.info(f"Similar companies:\n{similar}")

    # 6. Visualization
    logger.info("Generating Visualization...")
    generate_3d_market_galaxy(embeddings, metadata, output_path='docs/market_galaxy.html')

    logger.info("Pipeline Complete!")

if __name__ == "__main__":
    main()
