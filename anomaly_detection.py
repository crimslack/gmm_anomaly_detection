import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.manifold import TSNE

# Read the dataset
df = pd.read_csv("steel_plates_faults_original_dataset.csv")

# Define fault columns
fault_columns = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]

# Create a new column: 'Hata' (1 = faulty, 0 = non-faulty)
df["Hata"] = df[fault_columns].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

# Check the number of faulty products
print(df["Hata"].value_counts())

# Drop non-feature columns to define model input (X)
X = df.drop(columns=["id", "Hata"] + fault_columns)

# Standardize the numerical features (important for GMM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the GMM model (initially 2 clusters)
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X_scaled)

# Predict cluster assignments for each sample
clusters = gmm.predict(X_scaled)

# Add cluster labels to the dataframe
df["Cluster"] = clusters

# Show cluster distribution
print(df["Cluster"].value_counts())

# Cross-tabulation of cluster labels and true fault labels
crosstab = pd.crosstab(df["Cluster"], df["Hata"])
print(crosstab)

# Compute average feature values by cluster
cluster_means = df.groupby("Cluster").mean(numeric_only=True).T

# Sort features by difference between cluster averages
cluster_means["difference"] = abs(cluster_means[0] - cluster_means[1])
cluster_means_sorted = cluster_means.sort_values("difference", ascending=False)

# Show top 10 most discriminative features
print(cluster_means_sorted.head(10))

# BIC / AIC evaluation for best number of clusters
bic_scores = []
aic_scores = []
n_components_range = range(1, 10)

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))

# Plot BIC/AIC scores
plt.figure(figsize=(10, 5))
plt.plot(n_components_range, bic_scores, label='BIC', marker='o')
plt.plot(n_components_range, aic_scores, label='AIC', marker='s')
plt.xlabel('Number of Clusters')
plt.ylabel('Score (Lower = Better)')
plt.title('GMM BIC / AIC Evaluation')
plt.legend()
plt.grid(True)
plt.show()

# ROC Curve (anomaly score evaluation)
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X_scaled)

# Use log-likelihood as anomaly score (lower = more anomalous)
scores = -gmm.score_samples(X_scaled)

# True labels (1 = faulty)
y_true = df["Hata"]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, scores)
roc_auc = roc_auc_score(y_true, scores)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='navy')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - GMM Anomaly Detection')
plt.legend()
plt.grid(True)
plt.show()

# Train GMM model again with 6 clusters based on BIC results
gmm_6 = GaussianMixture(n_components=6, random_state=42)
gmm_6.fit(X_scaled)

# Predict new clusters
df["Cluster_6"] = gmm_6.predict(X_scaled)

# Show distribution of the new clusters
print(df["Cluster_6"].value_counts())

# Analyze feature differences across 6 clusters
cluster6_means = df.groupby("Cluster_6").mean(numeric_only=True).T
cluster6_means["max_diff"] = cluster6_means.max(axis=1) - cluster6_means.min(axis=1)
print(cluster6_means.sort_values("max_diff", ascending=False).head(10))

# t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df["Cluster_6"], cmap='tab10', s=12)
plt.title("t-SNE Visualization of GMM (6 Clusters)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()
