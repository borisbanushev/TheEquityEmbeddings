PRD: EquityEmbeddings — High-Dimensional Equity Embeddings
Project: EquityEmbeddings — Quantitative Multi-Modal Representation Learning


1. Product Vision
To build a non-linear vector space representing the global equity market. By capturing ~80 orthogonal features across four data modalities, EquityEmbeddings allows us to calculate the "Data-driven stock similarity" between companies based on structural health rather than just sector labels.

2. Technical Architecture: Multi-Modal Autoencoder
The system employs a Symmetric Deep Autoencoder with specialized input branches to handle the different statistical distributions of fundamental ratios, options Greeks, and technical momentum.

2.1 Layer Specifications

Input Layer (D≈85): Normalized concatenated feature vector.

Encoder Block: * Dense(128) → BatchNorm → Dropout(0.2) → GELU

Dense(64) → BatchNorm → GELU

Latent Bottleneck (The Embedding): 16 Units (Linear).

Decoder Block: Symmetric to Encoder (64 → 128 → Output).

Output Layer: Reconstructed N features (Loss: MSE).

3. The "Master 80" Feature Dictionary
Module A: Core Profitability & Management (SimFin)

Operating Margin: EBIT / Revenue.

Net Profit Margin: Net Income / Revenue.

EBITDA Margin: Proxy for cash-flow efficiency.

Gross Margin Stability: 3Y Std Dev of Gross Margin.

SGA-to-Revenue: Measure of operational overhead/bloat.

ROIC: Return on Invested Capital (Quality of management).

ROE: Return on Equity.

ROA: Return on Assets (Asset intensity).

Asset Turnover: Revenue / Total Assets.

Research Intensity: R&D Expense / Revenue.

Module B: Solvency & Capital Structure (SimFin)

Altman Z-Score: Weighted bankruptcy risk metric.

Current Ratio: Current Assets / Current Liabilities.

Quick Ratio: (Current Assets - Inventory) / Liabilities.

Debt-to-Equity: Total Debt / Shareholder Equity.

Net Debt/EBITDA: Debt relative to earning power.

Interest Coverage: EBIT / Interest Expense.

Dividend Payout Ratio: Dividends / Net Income.

Shareholder Equity Ratio: Equity / Total Assets.

Days Sales Outstanding (DSO): Efficiency of collections.

Cash Conversion Cycle: Days to turn inventory into cash.

Module C: Growth & Valuation (SimFin/Polygon)

Revenue CAGR (3Y): Multi-year sales velocity.

EPS CAGR (3Y): Earnings growth stability.

Rule of 40: Revenue Growth % + EBITDA Margin %.

Forward P/E: Current Price / Est. EPS.

PEG Ratio: (P/E) / Growth Rate.

Price-to-Sales: Revenue-based valuation.

Price-to-Book: Asset-based valuation.

EV/EBITDA: Enterprise value relative to cash flow.

Free Cash Flow Yield: FCF / Market Cap.

Earnings Surprise %: Mean surprise over last 4 quarters.

Module D: "Smart Money" & Options (Polygon Massive)

Calculated daily using full options chain snapshots. 31. Aggregate GEX (Gamma Exposure): Net dealer gamma positioning. 32. Spot Gamma %: Total Gamma relative to Market Cap. 33. Vanna Exposure: Sensitivity of delta to Volatility changes. 34. Charm Exposure: Sensitivity of delta to Time decay. 35. Option Skew (25D): 25-delta Put IV / 25-delta Call IV. 36. IV Rank (252D): Percentile of current IV over 1 year. 37. IV-to-Realized Vol Spread: "Volatility Risk Premium" (VRP). 38. Put/Call OI Ratio: Open Interest sentiment. 39. Put/Call Volume Ratio: Daily trading sentiment. 40. Call Wall Distance: Distance to the strike with max Call Gamma. 41. Put Wall Distance: Distance to the strike with max Put Gamma. 42. Zero-Gamma Level Distance: Distance to where dealer hedging flips.

Module E: Technical & Market Dynamics (Polygon)

Beta (v S&P 500): Market correlation.

Idiosyncratic Volatility: Residual vol unexplained by Beta.

RSI (14): Relative Strength Index.

Distance to 200-day MA: Trend health indicator.

52-Week High Distance: Nearness to breakout.

Average True Range (ATR) %: Price "noise" level.

Volume Z-Score: Abnormal volume relative to 30-day mean.

Short Interest %: Percentage of float held short.

Module F: Syntactic & Categorical (Engineered)

Operating Leverage: % Change in EBIT / % Change in Sales.

Financial Leverage: % Change in EPS / % Change in EBIT.

Burn Rate Runway: Cash / (Net Loss) [For growth firms].

Sector Embedding (1-3): Vectorized GICS sector representation.

Region Embedding (1-2): Latent geo-risk (NorthAm, EMEA, etc). + 25 additional "interaction terms" (e.g., ROIC/Beta, FCF/Debt).

4. Software Development Standards
4.1 Production Stack

Language: Python

Framework: PyTorch (for Autoencoder), Scikit-Learn (for Scaling).

Database: Parquet files for local storage; PostgreSQL/TimescaleDB for production.

Secrets: .env using python-dotenv.

4.2 Code Best Practices

Class Encapsulation: All logic resides in DataHarvester, FeatureEngineer, and ModelTrainer.

Typing & Docs: Mandated typing module for all functions and Google-style docstrings.

Data Quality Checks: Implement a DataValidator class to catch NaNs or "infinite" ratios (e.g., P/E during loss-making quarters) before training.

Environment Safety: ```bash

.env template
POLYGON_API_KEY=your_key_here SIMFIN_API_KEY=your_key_here DB_CONNECTION_STRING=...


5. Deployment & Outcome
Output: A faiss index allowing for vector search.

Visualization: A 3D interactive "Market Galaxy" using UMAP for dimensionality reduction.