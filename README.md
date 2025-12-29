# EquityEmbeddings

A deep learning system that transforms complex financial data into compact, semantically-rich embeddings for intelligent equity analysis and comparison.

## Problem Statement

Financial markets generate vast amounts of heterogeneous data across multiple dimensions: fundamentals, valuations, growth metrics, options market behavior, and categorical attributes. Traditional approaches to equity analysis face critical challenges:

- **Data Dimensionality**: Analyzing 80+ financial metrics simultaneously is intractable for human analysts
- **Nonlinear Relationships**: Traditional clustering methods fail to capture complex interdependencies between financial ratios
- **Computational Efficiency**: Real-time similarity search across thousands of securities requires efficient data structures
- **Signal Extraction**: Identifying truly comparable companies requires distilling fundamental characteristics from noisy market data

## Business Implications

EquityEmbeddings addresses these challenges through learned compression of financial data, enabling:

### Investment Applications
- **Peer Discovery**: Identify true operational peers beyond traditional sector classifications
- **Pairs Trading**: Discover statistically similar equities for market-neutral strategies
- **Portfolio Construction**: Build diversified portfolios by maximizing distance in embedding space
- **Risk Assessment**: Detect concentration risk by identifying clustered portfolio positions

### Research & Analytics
- **Quantitative Screening**: Efficiently filter investment universes by semantic similarity
- **Anomaly Detection**: Identify outliers deviating from their peer group embeddings
- **Factor Discovery**: Extract latent factors from the learned embedding space
- **Market Structure Analysis**: Visualize and understand market topology in low-dimensional space

### Operational Efficiency
- **Automated Monitoring**: Continuous tracking of competitive positioning in embedding space
- **Scalable Analysis**: Sub-linear search complexity enables real-time queries across large universes
- **Dimensionality Reduction**: Compress 80+ features into 16-dimensional representations with minimal information loss

## Technical Approach

### Data Sources

The system integrates multiple institutional-grade data sources to construct comprehensive financial profiles:

#### 1. SimFin API - Fundamental Data
- **Income Statements**: Revenue, operating income, net income, R&D expenditure
- **Balance Sheets**: Assets, liabilities, equity, working capital components
- **Cash Flow Statements**: Operating cash flow, capital expenditures, free cash flow
- **Coverage**: Annual statements for S&P 500 constituents
- **Data Quality**: Standardized, as-reported financial statements

#### 2. Polygon.io API - Market Data
- **Real-time Pricing**: Current stock prices for market-cap calculations
- **Options Chains**: Complete options data including strikes, expiries, open interest
- **Greeks**: Delta, gamma, vega, theta for options positioning analysis
- **Implied Volatility**: Option-implied volatility surfaces and skew metrics

#### 3. Wikipedia - Index Constituents
- **S&P 500 List**: Current index composition for universe definition

### Feature Engineering: The "Master 80" Framework

Features are organized into six specialized modules representing distinct aspects of corporate financial health:

#### Module A: Core Profitability & Management (10 features)
Financial efficiency and operational effectiveness metrics:
- Operating Margin, Net Profit Margin, EBITDA Margin
- Gross Margin, SG&A-to-Revenue ratio
- Return on Invested Capital (ROIC), Return on Equity (ROE), Return on Assets (ROA)
- Asset Turnover, R&D Intensity

#### Module B: Solvency & Capital Structure (10 features)
Balance sheet health and financial stability indicators:
- Altman Z-Score, Current Ratio, Quick Ratio
- Debt-to-Equity, Net Debt-to-EBITDA, Interest Coverage
- Dividend Payout Ratio, Shareholder Equity Ratio
- Days Sales Outstanding (DSO), Cash Conversion Cycle

#### Module C: Growth & Valuation (10 features)
Forward-looking metrics and market pricing:
- Revenue CAGR (3-year), EPS CAGR (3-year), Rule of 40
- Price-to-Sales, Price-to-Book, P/E Ratio
- PEG Ratio, EV/EBITDA, FCF Yield
- Free Cash Flow

#### Module D: Options Market & Smart Money (8 features)
Derivative market positioning and institutional flow:
- Gamma Exposure (GEX), Spot Gamma %
- Put/Call Open Interest Ratio, Put/Call Volume Ratio
- IV Rank (252-day), 25-Delta Option Skew
- Call Wall Distance, Put Wall Distance

#### Module F: Categorical & Interaction Terms (6+ features)
Non-linear combinations and sector embeddings:
- Operating Leverage, Financial Leverage, Burn Rate Runway
- **Sector Entity Embeddings** (3D learned representations, NOT one-hot encoding)
- Interaction Terms: ROE × Leverage, ROIC × Growth, Liquidity Spread, DuPont Components, FCF Quality

### Data Processing Pipeline

```
1. Data Harvesting
   ├── Fetch S&P 500 constituents
   ├── Download financial statements (SimFin)
   └── Retrieve market data (Polygon)

2. Feature Engineering
   ├── Calculate 80+ financial metrics
   ├── Generate sector embeddings
   ├── Create interaction terms
   ├── Handle missing values (median imputation)
   └── Normalize features (StandardScaler)

3. Autoencoder Training
   ├── Input: 80+ dimensional feature vectors
   ├── Compression: 80+ → 16 dimensions
   └── Reconstruction loss minimization

4. Embedding Generation
   ├── Extract latent representations
   ├── Save embeddings to disk
   └── Build FAISS index

5. Applications
   ├── Similarity search (k-NN queries)
   ├── 3D visualization (market galaxy)
   └── Dissimilarity analysis
```

## Architecture

### Symmetric Deep Autoencoder

The system employs a **MultiModalAutoencoder** - a symmetric neural network that learns to compress high-dimensional financial features into a compact latent space while preserving critical information.

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL AUTOENCODER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT LAYER                                                     │
│  ┌──────────────────────────────────────┐                       │
│  │  Financial Features (80+ dimensions) │                       │
│  │  ・ Profitability ratios              │                       │
│  │  ・ Solvency metrics                  │                       │
│  │  ・ Growth indicators                 │                       │
│  │  ・ Options market data               │                       │
│  │  ・ Sector embeddings                 │                       │
│  └──────────────────────────────────────┘                       │
│                    │                                             │
│                    ▼                                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      ENCODER                            │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  Layer 1: Linear(input_dim → 128)                       │    │
│  │           BatchNorm1d(128)                              │    │
│  │           GELU activation                               │    │
│  │           Dropout(0.2)                                  │    │
│  │                                                         │    │
│  │  Layer 2: Linear(128 → 64)                              │    │
│  │           BatchNorm1d(64)                               │    │
│  │           GELU activation                               │    │
│  │                                                         │    │
│  │  Layer 3: Linear(64 → 16)  [LATENT BOTTLENECK]         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                    │                                             │
│                    ▼                                             │
│  ┌──────────────────────────────────────┐                       │
│  │   LATENT SPACE (16 dimensions)      │ ◄── EMBEDDINGS         │
│  │                                      │     EXTRACTED          │
│  │  Compressed semantic representation  │     FROM HERE          │
│  │  of financial characteristics       │                        │
│  └──────────────────────────────────────┘                       │
│                    │                                             │
│                    ▼                                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      DECODER                            │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  Layer 1: Linear(16 → 64)                               │    │
│  │           BatchNorm1d(64)                               │    │
│  │           GELU activation                               │    │
│  │                                                         │    │
│  │  Layer 2: Linear(64 → 128)                              │    │
│  │           BatchNorm1d(128)                              │    │
│  │           GELU activation                               │    │
│  │                                                         │    │
│  │  Layer 3: Linear(128 → output_dim)                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                    │                                             │
│                    ▼                                             │
│  ┌──────────────────────────────────────┐                       │
│  │  RECONSTRUCTED FEATURES              │                       │
│  │  (80+ dimensions)                    │                       │
│  └──────────────────────────────────────┘                       │
│                                                                  │
│  TRAINING OBJECTIVE:                                             │
│  Minimize MSE Loss = ||Input - Reconstructed||²                 │
│                                                                  │
│  OPTIMIZER: Adam (lr=1e-3)                                       │
│  BATCH SIZE: 8-32 (adaptive based on dataset size)              │
│  EPOCHS: 20-100 (with early stopping)                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Architectural Decisions

**Symmetric Design**: Encoder and decoder are perfect mirrors, ensuring balanced information flow and preventing gradient pathologies.

**GELU Activations**: Gaussian Error Linear Units provide smoother gradients than ReLU, critical for financial data with heavy-tailed distributions.

**Batch Normalization**: Stabilizes training on small datasets (S&P 500 = ~500 samples) and accelerates convergence.

**Dropout Regularization**: 20% dropout in encoder prevents overfitting to specific financial regimes or market conditions.

**16-Dimensional Latent Space**: Empirically optimized to balance:
- Information preservation (reconstruction quality)
- Computational efficiency (FAISS search performance)
- Visualization capability (reducible to 3D via PCA/t-SNE)

**Linear Bottleneck**: Unlike variational autoencoders, we use a deterministic latent space for consistent, reproducible embeddings.

## Implementation Details

### Search Infrastructure

The system uses **FAISS** (Facebook AI Similarity Search) for efficient nearest-neighbor queries:

- **Index Type**: `IndexFlatL2` (exact L2 distance)
- **Query Complexity**: O(n·d) for brute force, optimizable to O(log n) with IVF/HNSW indices
- **Use Cases**:
  - `query_similar_tickers()`: Find k nearest neighbors
  - `query_dissimilar_tickers()`: Find k furthest points (inverse search)
  - `analyze_ticker()`: Comprehensive peer analysis

### Visualization

3D market visualization using dimensionality reduction:
- PCA or t-SNE projection from 16D → 3D
- Interactive HTML visualization with Plotly
- Color-coded by sector, sized by market cap

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/EquityEmbeddings.git
cd EquityEmbeddings

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys:
# SIMFIN_API_KEY=your_simfin_key
# POLYGON_API_KEY=your_polygon_key
```

## Usage

### Full Pipeline Execution

```python
python main.py
```

This runs the complete workflow:
1. Fetches S&P 500 constituents
2. Downloads financial data
3. Engineers 80+ features
4. Trains autoencoder
5. Generates embeddings
6. Creates search index
7. Produces 3D visualization

### Query Similar Companies

```python
from src.search import EquitySearch
import pandas as pd
import numpy as np

# Load pre-computed embeddings
embeddings = pd.read_csv('data/processed/embeddings.csv', index_col=0).values
metadata = pd.read_csv('data/processed/metadata.csv')

# Initialize search engine
search = EquitySearch(embeddings, metadata)

# Find companies similar to Apple
similar = search.query_similar_tickers('AAPL', k=5)
print(similar[['ticker', 'name', 'sector']])

# Find dissimilar companies (diversification candidates)
dissimilar = search.query_dissimilar_tickers('AAPL', k=5)
print(dissimilar[['ticker', 'name', 'sector', 'distance']])
```

### Custom Analysis

```python
# Analyze a specific ticker comprehensively
analysis = search.analyze_ticker('TSLA', k=5)
print("Query Company:", analysis['query_info'])
print("\nMost Similar:", analysis['most_similar'])
print("\nLeast Similar:", analysis['least_similar'])
```

## Output Files

```
data/
├── processed/
│   ├── processed_features.csv    # 80+ engineered features
│   ├── embeddings.csv             # 16D latent representations
│   └── metadata.csv               # Company names, sectors, etc.
├── simfin_cache/                  # SimFin data cache
docs/
└── market_galaxy.html             # 3D interactive visualization
```

## System Requirements

- **Python**: 3.8+
- **RAM**: 4GB minimum (scales with universe size)
- **GPU**: Optional (MPS/CUDA support for faster training)
- **Dependencies**:
  - PyTorch 2.0+
  - FAISS-CPU or FAISS-GPU
  - SimFin, Polygon.io API access
  - pandas, numpy, scikit-learn

## Performance Metrics

- **Compression Ratio**: 80+ → 16 dimensions (5x compression)
- **Training Time**: ~2-5 minutes for S&P 500 (CPU)
- **Search Latency**: <10ms for k-NN queries on 500 tickers
- **Reconstruction Loss**: Typically <0.05 MSE after convergence

## Future Enhancements

- **Temporal Embeddings**: Time-series modeling for trend detection
- **Sector-Specific Models**: Fine-tuned autoencoders per industry
- **Alternative Architectures**: VAE, β-VAE for uncertainty quantification
- **Multi-Asset Support**: Extend to fixed income, commodities, crypto
- **Real-time Updates**: Incremental learning for daily embedding updates

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Financial data provided by SimFin and Polygon.io
- FAISS library by Facebook Research
- Inspired by modern representation learning in NLP (Word2Vec, BERT) applied to financial domain

## Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

**Disclaimer**: This tool is for research and educational purposes only. Not financial advice. Past performance does not guarantee future results.
