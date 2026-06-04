import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def perform_segmentation(processed_dir="data/processed"):
    profile_path = os.path.join(processed_dir, "customer_profiles.csv")
    
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Processed customer profiles not found at {profile_path}. Run processing first.")
        
    print("Loading customer profiles...")
    df = pd.read_csv(profile_path)
    
    # Exclude customers with 0 transactions (e.g. if any registered but never bought)
    active_mask = df["total_transactions"] > 0
    df_active = df[active_mask].copy()
    df_inactive = df[~active_mask].copy()
    
    # ----------------------------------------------------
    # 1. RFM Analysis
    # ----------------------------------------------------
    print("Running RFM analysis...")
    
    # Recency: lower is better (more recent) -> Q5 is most recent
    # Frequency: higher is better -> Q5 is highest frequency
    # Monetary: higher is better -> Q5 is highest spend
    
    # We use qcut to divide into quintiles (1 to 5).
    # Since recency is better when small, we label 5 as the smallest values, 1 as the largest.
    df_active["R_Score"] = pd.qcut(df_active["recency_days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    
    # Frequency and Monetary could have duplicate bin edges if values are highly uniform,
    # so we'll use rank-based qcut to avoid errors
    df_active["F_Score"] = pd.qcut(df_active["total_transactions"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df_active["M_Score"] = pd.qcut(df_active["total_spend"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    # RFM Segment formulation
    df_active["RFM_Score"] = df_active["R_Score"].astype(str) + df_active["F_Score"].astype(str) + df_active["M_Score"].astype(str)
    
    # Map RFM to marketing segments
    # Custom segmentation rules
    def get_rfm_segment(row):
        r = row["R_Score"]
        f = row["F_Score"]
        m = row["M_Score"]
        
        # Champions: bought recently, buy frequently and spend the most
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        # Loyal Customers: buy regularly, responsive to promotions
        elif r >= 3 and f >= 3 and m >= 3:
            return "Loyalists"
        # Potential Loyalists: recent customers with average frequency and spend
        elif r >= 4 and f >= 2 and m >= 2:
            return "Potential Loyalists"
        # New Customers: bought recently, low frequency/spend
        elif r >= 4 and (f == 1 or m == 1):
            return "New Customers"
        # Can't Lose Them: spent a lot and bought often, but haven't bought in a long time
        elif r <= 2 and f >= 4 and m >= 4:
            return "Can't Lose Them"
        # At Risk: bought often but long time ago; needs activation
        elif r <= 2 and f >= 2 and m >= 2:
            return "At Risk"
        # Hibernating: last purchase was long ago, low frequency & low spend
        else:
            return "Hibernating"
            
    df_active["rfm_segment"] = df_active.apply(get_rfm_segment, axis=1)
    
    # ----------------------------------------------------
    # 2. K-Means Clustering
    # ----------------------------------------------------
    print("Running K-Means clustering...")
    
    # Clustering variables: Recency (log-scale to reduce skew), Frequency (log-scale), Spend (log-scale)
    # Add a small epsilon to avoid log(0)
    eps = 0.001
    cluster_features = ["recency_days", "total_transactions", "total_spend"]
    
    X = df_active[cluster_features].copy()
    for col in cluster_features:
        X[col] = np.log1p(X[col] + eps)
        
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit K-Means with 4 clusters
    # 4 clusters gives a clear, simple segmentation (High Spenders, Active Spenders, Dormant Spenders, Lost)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_active["kmeans_cluster"] = kmeans.fit_predict(X_scaled)
    
    # Describe clusters by sorting by median values
    cluster_stats = df_active.groupby("kmeans_cluster")[cluster_features].median()
    
    # We assign meaningful names to K-Means clusters based on their median profiles
    # Cluster characteristics:
    # 1. High spend, high frequency, low recency -> "VIP / Champions"
    # 2. Moderate/high spend, moderate frequency, low recency -> "Active Spenders"
    # 3. Moderate spend, low frequency, high recency -> "At-Risk / Slipping"
    # 4. Low spend, low frequency, high recency -> "Dormant / Lost"
    
    cluster_mapping = {}
    
    # Sort cluster indices by total_spend median to systematically assign profiles
    sorted_clusters = cluster_stats.sort_values(by="total_spend", ascending=False).index.tolist()
    
    # 0 = highest spend, 3 = lowest spend (usually)
    cluster_names = ["VIP Spenders", "Active Customers", "Slipping Customers", "Lost Customers"]
    for i, c_idx in enumerate(sorted_clusters):
        cluster_mapping[c_idx] = cluster_names[i]
        
    df_active["cluster_name"] = df_active["kmeans_cluster"].map(cluster_mapping)
    
    # For inactive customers, fill default values
    if len(df_inactive) > 0:
        df_inactive["R_Score"] = 0
        df_inactive["F_Score"] = 0
        df_inactive["M_Score"] = 0
        df_inactive["RFM_Score"] = "000"
        df_inactive["rfm_segment"] = "Inactive"
        df_inactive["kmeans_cluster"] = -1
        df_inactive["cluster_name"] = "Never Purchased"
        
        df_final = pd.concat([df_active, df_inactive]).sort_index()
    else:
        df_final = df_active
        
    # Save segmented profiles
    output_path = os.path.join(processed_dir, "segmented_customers.csv")
    df_final.to_csv(output_path, index=False)
    print(f"Customer segmentation complete! Saved to {output_path}")
    
    # Print cluster size distribution
    print("\nK-Means Cluster Distribution:")
    print(df_final["cluster_name"].value_counts())
    print("\nRFM Segment Distribution:")
    print(df_final["rfm_segment"].value_counts())

if __name__ == "__main__":
    perform_segmentation()
