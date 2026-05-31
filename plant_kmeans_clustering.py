import matplotlib

matplotlib.use('Agg')  # 非交互式后端

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_plant_features(feat_path):
    batch_features = np.load(feat_path, allow_pickle=True).item()
    feature_list = []
    img_names = []
    for img_name, feat_data in batch_features.items():
        flat_feat = feat_data['flat_feature'].reshape(-1)
        feature_list.append(flat_feat)
        img_names.append(img_name)
    features = np.array(feature_list)
    print(f"✅ 特征加载完成：样本数={features.shape[0]}, 维度={features.shape[1]}")
    return features, img_names


def preprocess_features(features):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print(f"✅ 特征标准化完成")
    return features_scaled, scaler


def run_kmeans(features, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(features)
    print(f"✅ K-Means聚类完成：聚类数={n_clusters}, 各聚类样本数={np.bincount(cluster_labels)}")
    return kmeans, cluster_labels


def visualize_clusters(features, cluster_labels, img_names):
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    print(f"✅ PCA降维完成，解释方差比={pca.explained_variance_ratio_.sum():.2f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_labels, cmap='tab10', s=80)
    plt.title('Plant K-Means Clustering')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.grid(alpha=0.3)
    plt.savefig('plant_kmeans_clustering.png', dpi=200)
    plt.close()
    print(f"✅ 聚类图已保存：plant_kmeans_clustering.png")


if __name__ == "__main__":
    feat_path = r"C:\Users\1\yolov5\batch_plant_features.npy"
    n_clusters = 4

    features, img_names = load_plant_features(feat_path)
    features_scaled, scaler = preprocess_features(features)
    kmeans, cluster_labels = run_kmeans(features_scaled, n_clusters)
    visualize_clusters(features_scaled, cluster_labels, img_names)

    print("\n📌 聚类结果：")
    for img, label in zip(img_names, cluster_labels):
        print(f"   {img} → 标签{label}")