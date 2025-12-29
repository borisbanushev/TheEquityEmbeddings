# EquityEmbeddings - Usage Guide

## Overview

EquityEmbeddings transforms complex financial data into a **16-dimensional vector space** where similar companies are close together and dissimilar companies are far apart. This enables powerful similarity search and analysis.

## How It Works

### 1. Data Pipeline
```
Raw Financial Data → Feature Engineering → Autoencoder → 16D Embeddings
    (80 columns)          (72 features)        (compress)    (essence)
```

### 2. What Are Embeddings?

Think of embeddings as a "fingerprint" for each company:
- **16 numbers** that capture the company's financial structure, health, and characteristics
- Companies with similar business models and financials have similar embeddings
- Distance in embedding space = dissimilarity in financial structure

### 3. 3D Visualization vs Full Embeddings

**The 3D chart uses dimensionality reduction:**
- Takes all 16 dimensions → reduces to 3 for visualization
- Uses t-SNE or UMAP to preserve relative distances
- **You lose information** (like taking a shadow of a 3D object)

**Similarity calculations use all 16 dimensions:**
- No information loss
- More accurate comparisons
- This is what `query_ticker.py` uses

---

## Usage Methods

### Method 1: Command Line Interface (Recommended)

After running the pipeline, use the CLI tool:

```bash
# Basic query - show top 5 most/least similar
python query_ticker.py AAPL

# Show top 10 companies
python query_ticker.py TSLA --top 10

# Only show similar, not dissimilar
python query_ticker.py MSFT --no-dissimilar

# Get help
python query_ticker.py --help
```

**Example Output:**
```
======================================================================
Analyzing: AAPL
======================================================================

COMPANY INFORMATION:
  Ticker: AAPL
  Name:   APPLE INC
  Industry ID: 103001

🔗 TOP 5 MOST SIMILAR COMPANIES:
──────────────────────────────────────────────────────────────────────
ticker  name                           distance
MSFT    MICROSOFT CORP                 2.1532
GOOGL   ALPHABET INC CLASS A           2.4891
META    META PLATFORMS INC             2.6743
NVDA    NVIDIA CORP                    2.9012
ADBE    ADOBE INC                      3.1234

⚡ TOP 5 LEAST SIMILAR COMPANIES:
──────────────────────────────────────────────────────────────────────
ticker  name                           distance
XOM     EXXON MOBIL CORP              18.4532
CVX     CHEVRON CORP                  17.9821
...
```

### Method 2: Python Script

```python
import pandas as pd
import numpy as np
from src.search import EquitySearch

# Load embeddings and metadata
embeddings_df = pd.read_csv('data/processed/embeddings.csv', index_col=0)
embeddings = embeddings_df.values

# Load metadata (simplified - see query_ticker.py for full version)
import simfin as sf
companies = sf.load_companies(market='us')
# ... align with embeddings ...

# Create search engine
search = EquitySearch(embeddings, metadata)

# Query
analysis = search.analyze_ticker('AAPL', k=5)
print(analysis['most_similar'])
print(analysis['least_similar'])
```

### Method 3: Jupyter Notebook (Interactive)

Open the provided notebook:
```bash
jupyter notebook notebooks/explore_embeddings.ipynb
```

The notebook includes:
- Interactive similarity queries
- PCA/t-SNE visualizations
- Company comparisons
- Clustering analysis
- Feature importance analysis

---

## Common Use Cases

### 1. Find Competitors
"Who are the companies most similar to AAPL?"
```bash
python query_ticker.py AAPL
```
**Use Case:** Competitive analysis, sector mapping

### 2. Portfolio Diversification
"Which companies are LEAST similar to my holdings?"
```bash
python query_ticker.py AAPL --top 20
# Look at the "least similar" section
```
**Use Case:** Reduce correlation, hedge positions

### 3. Peer Group Analysis
Find companies in a specific distance range (notebook):
```python
find_companies_in_range('JPM', min_dist=2.0, max_dist=4.0)
```
**Use Case:** Relative valuation, peer benchmarking

### 4. Sector Discovery
Cluster companies to find natural groupings:
```python
# See clustering section in notebook
```
**Use Case:** Discover hidden sector relationships beyond GICS

### 5. Anomaly Detection
Companies far from everyone:
```python
# Calculate average distance to all others
avg_distances = []
for ticker in embeddings_df.index:
    emb = embeddings_df.loc[ticker].values
    distances = np.linalg.norm(embeddings_df.values - emb, axis=1)
    avg_distances.append(distances.mean())

# Find outliers
outliers = embeddings_df.index[np.argsort(avg_distances)[-10:]]
```
**Use Case:** Special situations, unique business models

### 6. Pair Trading
Find highly similar companies (low distance):
```bash
python query_ticker.py AAPL --top 3
# Look for top similar with distance < 3.0
```
**Use Case:** Statistical arbitrage, mean reversion

---

## Understanding Distances

**Distance Ranges** (typical values):

| Distance | Interpretation | Example Pairs |
|----------|---------------|---------------|
| 0.0 - 2.0 | Very similar | MSFT ↔ GOOGL (tech giants) |
| 2.0 - 5.0 | Moderately similar | AAPL ↔ NVDA (tech, different sectors) |
| 5.0 - 10.0 | Somewhat similar | AAPL ↔ JPM (large cap, different industries) |
| 10.0+ | Very different | AAPL ↔ XOM (tech vs oil) |

**What Drives Similarity?**
- Profitability metrics (margins, ROE, ROA)
- Capital structure (debt ratios, liquidity)
- Growth characteristics (cash flow patterns)
- Sector embeddings (industry classification)

---

## Output Files

After running `python main.py`:

### 1. `data/processed/processed_features.csv`
- **72 features** × **435 companies**
- Engineered financial ratios + sector encodings
- Normalized and cleaned
- Use for: Understanding what drives the embeddings

### 2. `data/processed/embeddings.csv`
- **16 dimensions** × **435 companies**
- Compressed representation
- Columns: `dim_0` through `dim_15`
- Use for: All similarity calculations

### 3. `docs/market_galaxy.html`
- Interactive 3D visualization (t-SNE projection)
- Open in browser
- Hover to see company names
- Use for: Quick visual exploration

---

## Advanced Usage

### Custom Distance Metrics

By default, we use **Euclidean distance** (L2 norm). You can experiment with others:

```python
from scipy.spatial.distance import cosine, cityblock

# Cosine similarity (angle-based)
similarity = 1 - cosine(emb1, emb2)

# Manhattan distance (L1 norm)
distance = cityblock(emb1, emb2)
```

### Weighted Search

Give more importance to certain embedding dimensions:

```python
# Assume dim_0 and dim_1 are most important
weights = np.ones(16)
weights[0] = 2.0  # Double weight on dim_0
weights[1] = 2.0

weighted_distance = np.sqrt(((emb1 - emb2)**2 * weights).sum())
```

### Time Series Analysis

Track how embeddings change over time by:
1. Running pipeline on different date ranges
2. Comparing embeddings for same companies
3. Detecting structural shifts

---

## Troubleshooting

### "Ticker not found"
- Check spelling (must be uppercase)
- Check ticker exists in S&P 500
- Check it has complete financial data

### "Embeddings not found"
- Run `python main.py` first to generate embeddings
- Check `data/processed/` directory exists

### Poor similarity results
- May need more features (add technical indicators, options data)
- Try increasing latent dimensions (16 → 32)
- Check feature quality (too many NaNs?)

---

## Next Steps

1. **Add more features** from Polygon API (Module D, E)
2. **Experiment with architecture** (deeper encoder, different latent_dim)
3. **Time-based analysis** (how do embeddings evolve quarterly?)
4. **Integration** (use embeddings in portfolio optimizer, risk model)

---

## Citation

If you use this in research or production:
```
EquityEmbeddings: Multi-Modal Autoencoder for Financial Similarity
Derived from fundamental, technical, and options data
Architecture: 72 → 128 → 64 → 16 → 64 → 128 → 72
```
