#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import plotly.express as px

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from modules import *
input_data = InputData("ANSPRACHE_MARKETING_IMPUTED", max_twin_num = 10, kernel_size = 1) #Options: ANSPRACHE, ANSPRACHE_MARKETING_IMPUTED
df_grid_results = pd.read_csv("results/grid_results.csv")
#%%
#Idee im Median über alle Items ist das einfache Bootstrapping besser, aber für einzelne Items/Itemcluster können abweichende Parametereinstellungen besser sein
#wir können max perplexity nehmen, dadurch ist das Ergebnis "deterministisch"? weil globales optimum
#ergibt es mehr sinn nur gute oder gute und schlechte parameter zu nehmen
#muss ich hier noch die daten skalieren?
#run tsne pre and post tsne to ensure there are no tsne artifacts
#they recommend DBSCAN for clustering bc tsne does create non-spherical clusters

#gruppieren per item, order by wasserstein, select the top 5/10 combinations, cluster on similarity of the top 5/10 combinations to result in maybe 4 clusters?
# list(input_data.TestData.keys())[:5]

#df_grid_results.query("TEST_ITEM_COMMUNICATIONKEY == 804245827 & WINDOW_SIZE != 0").sort_values("WASSERSTEIN")

# %%
def dimensionality_reduction(vectors, method="tsne", n_components=2, perplexity=0.5, random_state=161):
    """
    Apply dimensionality reduction using either t-SNE or PCA.
    
    Parameters:
    - vectors: np.array, the high-dimensional data to reduce.
    - method: str, "tsne" or "pca" (default: "tsne").
    - n_components: int, number of dimensions to reduce to (default: 2).
    - perplexity: int, only used for t-SNE (default: 30).
    - random_state: int, ensures reproducibility.

    Returns:
    - DataFrame with reduced components.
    """
    if method == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        X_reduced = reducer.fit_transform(vectors)
        return pd.DataFrame(X_reduced, columns=[f"Dim_{i+1}" for i in range(n_components)])
    
    elif method == "pca":
        reducer = PCA(n_components=n_components)
        X_reduced = reducer.fit_transform(vectors)

         # Print PCA information
        print("Explained Variance:")
        print(reducer.explained_variance_)
        
        print("\nExplained Variance Ratio:")
        print(reducer.explained_variance_ratio_)
        
        print("\nComponent Loadings:")
        loadings = pd.DataFrame(reducer.components_, columns=[f"Feature_{i+1}" for i in range(vectors.shape[1])])
        print(loadings)
        
        return pd.DataFrame(X_reduced, columns=[f"Dim_{i+1}" for i in range(n_components)])
    
    else:
        raise ValueError("Method must be 'tsne' or 'pca'.")

def plot_elbow_silhouette(vectors, max_k=10):
    """
    Automatically determine the best number of clusters (k) for K-Means using the Elbow Method and Silhouette Score.

    Parameters:
    - vectors: np.array, the data for clustering.
    - max_k: int, maximum number of clusters to evaluate (default: 10).

    Returns:
    - best_k: int, the optimal number of clusters.
    """
    distortions = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=161, n_init=10)
        labels = kmeans.fit_predict(vectors)
        
        # Compute inertia (sum of squared distances)
        distortions.append(kmeans.inertia_)

        # Compute silhouette score (higher is better)
        silhouette_avg = silhouette_score(vectors, labels)
        silhouette_scores.append(silhouette_avg)

    # Plot Elbow Method (Inertia)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ax[0].plot(k_range, distortions, marker='o', linestyle='-')
    ax[0].set_xlabel("Number of Clusters (k)")
    ax[0].set_ylabel("Inertia (Distortion)")
    ax[0].set_title("Elbow Method")

    # Plot Silhouette Score
    ax[1].plot(k_range, silhouette_scores, marker='o', linestyle='-')
    ax[1].set_xlabel("Number of Clusters (k)")
    ax[1].set_ylabel("Silhouette Score")
    ax[1].set_title("Silhouette Score Analysis")

    plt.show()

    # Select best k based on silhouette score
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters based on silhouette score: {best_k}")

    return best_k

# Now update the main function to automatically print the elbow plot
def main_reduction_and_clustering(df_grid_results, input_data, reduction_method="tsne"):
    # Normalize selected columns
    columns_to_scale = ["BLOCK_SIZE", "WINDOW_SIZE"]
    scaler = StandardScaler()
    df_scaled = df_grid_results.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])

    # Prepare results DataFrame
    results = pd.DataFrame()
    for key in input_data.TestData.keys():
        subset = df_scaled.query(f"TEST_ITEM_COMMUNICATIONKEY == {key} & WINDOW_SIZE != 0").sort_values("WASSERSTEIN")

        # Select top/bottom 5 values
        subset_head = subset.head(25)
        subset_tail = subset.tail(25)

        block_size_values = list(subset_head["BLOCK_SIZE"]) + list(subset_tail["BLOCK_SIZE"])
        window_size_values = list(subset_head["WINDOW_SIZE"]) + list(subset_tail["WINDOW_SIZE"])
        vector = block_size_values + window_size_values

        results = pd.concat(
            [results, pd.DataFrame({"TEST_ITEM_COMMUNICATIONKEY": [key], "vec": [vector]})],
            ignore_index=True
        )

    vectors = np.array(results["vec"].tolist())

    # Apply dimensionality reduction dynamically
    reduced_results = dimensionality_reduction(vectors, method=reduction_method)

    # Determine optimal k and plot elbow method
    best_k = plot_elbow_silhouette(reduced_results, max_k=10)

    # Apply K-Means clustering with optimal k
    kmeans = KMeans(n_clusters=best_k, random_state=161, n_init=10)
    reduced_results["Cluster"] = kmeans.fit_predict(reduced_results)

    # Attach TEST_ITEM_COMMUNICATIONKEY back
    reduced_results["TEST_ITEM_COMMUNICATIONKEY"] = results["TEST_ITEM_COMMUNICATIONKEY"]

    # Plot results
    fig = px.scatter(
        reduced_results,
        x="Dim_1",
        y="Dim_2",
        color=reduced_results["Cluster"].astype(str),
        hover_data={
            "Dim_1": True,
            "Dim_2": True,
            "TEST_ITEM_COMMUNICATIONKEY": True
        },
        title=f"{reduction_method.upper()} Visualization with Optimal K={best_k}",
        labels={"Dim_1": f"{reduction_method.upper()} Component 1", "Dim_2": f"{reduction_method.upper()} Component 2"},
        template="plotly"
    )

    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.show()

# %%
main_reduction_and_clustering(df_grid_results, input_data, reduction_method="tsne")
# %%
