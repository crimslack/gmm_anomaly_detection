# gmm_anomaly_detection
Production fault segmentation using Gaussian Mixture Models and unsupervised anomaly detection.

# Anomaly Detection in Manufacturing Using Gaussian Mixture Model (GMM)

This project aims to segment and detect anomalies in faulty steel plate production data using the Gaussian Mixture Model (GMM) as an unsupervised learning method.

---

## 📁 Dataset

- **Steel Plates Faults Dataset** (Kaggle)
- 1,941 faulty product records
- 35 physical and geometric features per sample

---

## 🔧 Methods Used

- Gaussian Mixture Model (GMM)
- BIC / AIC analysis for optimal cluster selection
- ROC Curve evaluation
- t-SNE for 2D cluster visualization

---

## 📊 Project Flow

1. The dataset was loaded and a combined "Fault" column was created from all error type columns.
2. The GMM model was initially trained with 2 clusters.
3. BIC and AIC values were calculated to determine the optimal number of clusters.
4. Based on the analysis, **6 clusters** were selected and the model was retrained.
5. ROC Curve analysis was attempted but could not produce a result due to the dataset containing only faulty samples (labels = 1).
6. Finally, t-SNE was applied for 2D visualization of the clusters.

---

## 🗂️ Project Structure

├── data/
│ └── steel_plates_faults_original_dataset.csv
├── visuals/
│ ├── bic_aic_plot.png
│ ├── tsne.png
│ └── roc_curve.png
├── gmm_tsne_plot.py
├── requirements.txt
├── .gitignore
└── README.md

## 🧪 Installation & Execution

```bash
pip install -r requirements.txt
python gmm_tsne_plot.py

📌 Notes
ROC Curve could not be evaluated due to single-class labels (all samples labeled as faulty).

Based on BIC / AIC analysis, the GMM model was retrained using 6 clusters.

t-SNE visualization showed clear separation between clusters.

<3 crimslack
