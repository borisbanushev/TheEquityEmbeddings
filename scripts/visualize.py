import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

# Try to import umap, fallback to t-SNE if it fails (e.g. segfaults or missing)
# try:
#     import umap
#     HAS_UMAP = True
# except ImportError:
#     HAS_UMAP = False
#     logger.warning("UMAP not found, will use t-SNE as fallback.")
# except Exception as e:
#     HAS_UMAP = False
#     logger.warning(f"Error importing UMAP: {e}. Falling back to t-SNE.")

HAS_UMAP = False

def generate_3d_market_galaxy(embeddings: np.ndarray, metadata: pd.DataFrame, output_path: str = 'market_galaxy.html'):
    """
    Generate a 3D visualization of the market embeddings.
    Uses UMAP if available, otherwise t-SNE or PCA.
    """
    logger.info("Generating 3D Market Galaxy...")
    
    # Check dimensions. If we only have 16D, we need reduction.
    n_samples = embeddings.shape[0]
    n_components = 3
    
    if HAS_UMAP:
        try:
            logger.info("Using UMAP for dimensionality reduction...")
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            projections = reducer.fit_transform(embeddings)
        except Exception as e:
            logger.warning(f"UMAP failed during fit: {e}. Using t-SNE instead.")
            reducer = TSNE(n_components=n_components, random_state=42)
            projections = reducer.fit_transform(embeddings)
    else:
        logger.info("Using t-SNE for dimensionality reduction...")
        # t-SNE can be slow for very large datasets, but for ~1000-5000 it's fine
        reducer = TSNE(n_components=n_components, random_state=42, init='pca', learning_rate='auto')
        projections = reducer.fit_transform(embeddings)

    df_plot = pd.DataFrame(projections, columns=['x', 'y', 'z'])
    df_plot = pd.concat([df_plot, metadata.reset_index(drop=True)], axis=1)

    fig = px.scatter_3d(
        df_plot, x='x', y='y', z='z',
        color='sector' if 'sector' in df_plot.columns else None,
        hover_data=['ticker', 'name'] if 'ticker' in df_plot.columns and 'name' in df_plot.columns else None,
        title='EquityEmbeddings: 3D Market Galaxy (Latent Space)',
        template='plotly_dark'
    )

    fig.update_traces(marker=dict(size=4, opacity=0.8, line=dict(width=0)))
    fig.write_html(output_path)
    logger.info(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    # Test with dummy data
    num_samples = 200
    dummy_embeddings = np.random.randn(num_samples, 16)
    dummy_metadata = pd.DataFrame({
        'ticker': [f'T{i}' for i in range(num_samples)],
        'name': [f'Company {i}' for i in range(num_samples)],
        'sector': np.random.choice(['Tech', 'Finance', 'Energy', 'Healthcare'], num_samples)
    })
    generate_3d_market_galaxy(dummy_embeddings, dummy_metadata)
