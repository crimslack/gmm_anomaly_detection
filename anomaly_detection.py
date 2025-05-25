import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.manifold import TSNE


# Veriyi oku
df = pd.read_csv("steel_plates_faults_original_dataset.csv")

# Hata sütunlarını tanımla
fault_columns = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]

# Yeni bir sütun oluştur: 'Hata' (1 = hatalı, 0 = hatasız)
df["Hata"] = df[fault_columns].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

# Kontrol et: kaç tane hatalı ürün var?
print(df["Hata"].value_counts())

# Hedef dışındaki tüm sayısal sütunları al (model için X)
X = df.drop(columns=["id", "Hata"] + fault_columns)

# Sayısal veriyi ölçekle (GMM için önemli)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Modeli tanımla (örneğin 2 grup)
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X_scaled)

# Her gözlemin ait olduğu grubu tahmin et
clusters = gmm.predict(X_scaled)

# Sonuçları dataframe'e ekle
df["Cluster"] = clusters

# Grupların dağılımını göster
print(df["Cluster"].value_counts())

# Cluster ve Hata bilgilerini karşılaştır
crosstab = pd.crosstab(df["Cluster"], df["Hata"])
print(crosstab)

# Küme bazında ortalama değerleri hesapla
cluster_means = df.groupby("Cluster").mean(numeric_only=True).T

# Büyük fark olanları sırala (farkın mutlak değeri)
cluster_means["difference"] = abs(cluster_means[0] - cluster_means[1])
cluster_means_sorted = cluster_means.sort_values("difference", ascending=False)

# En çok fark gösteren ilk 10 özellik
print(cluster_means_sorted.head(10))

#BIC/AIC ile en uygun küme sayısını bulma 
bic_scores = []
aic_scores = []
n_components_range = range(1, 10)  # 1'den 9 kümeye kadar dene

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))

# Grafik çiz
plt.figure(figsize=(10, 5))
plt.plot(n_components_range, bic_scores, label='BIC', marker='o')
plt.plot(n_components_range, aic_scores, label='AIC', marker='s')
plt.xlabel('Cluster Sayısı')
plt.ylabel('Skor (Daha düşük = Daha iyi)')
plt.title('GMM BIC / AIC Değerlendirmesi')
plt.legend()
plt.grid(True)
plt.show()


#ROC Curve 
# GMM her örnek için "anomalilik olasılığı" da verir: predict_proba
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X_scaled)

# log-likelihood'tan anomaly skoru üret
scores = -gmm.score_samples(X_scaled)  # düşük olasılık = anomali

# Gerçek hata etiketleri (1 = hatalı)
y_true = df["Hata"]

# ROC hesapla
fpr, tpr, thresholds = roc_curve(y_true, scores)
roc_auc = roc_auc_score(y_true, scores)

# ROC Curve grafiği
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='navy')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - GMM Anomaly Detection')
plt.legend()
plt.grid(True)
plt.show()

#BIC sonucu 6 kümeli olarak yeniden modeli eğitiyoruz
# 6 kümeli GMM modeli kur
gmm_6 = GaussianMixture(n_components=6, random_state=42)
gmm_6.fit(X_scaled)

# Her örneğe ait cluster numarasını tahmin et
df["Cluster_6"] = gmm_6.predict(X_scaled)

# Kümelerin dağılımına bak
print(df["Cluster_6"].value_counts())

cluster6_means = df.groupby("Cluster_6").mean(numeric_only=True).T
cluster6_means["max_diff"] = cluster6_means.max(axis=1) - cluster6_means.min(axis=1)
print(cluster6_means.sort_values("max_diff", ascending=False).head(10))

#T-SNE ile görselleştirme
# t-SNE modeli
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

# Görselleştirme
plt.figure(figsize=(10, 7))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df["Cluster_6"], cmap='tab10', s=12)
plt.title("t-SNE ile GMM (6 Cluster) Görselleştirme")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()



