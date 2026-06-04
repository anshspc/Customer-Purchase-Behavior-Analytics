import os
import pandas as pd
import numpy as np

def clean_and_process_data(raw_dir="data/raw", processed_dir="data/processed"):
    os.makedirs(processed_dir, exist_ok=True)
    
    cust_path = os.path.join(raw_dir, "customers.csv")
    tx_path = os.path.join(raw_dir, "transactions.csv")
    
    if not os.path.exists(cust_path) or not os.path.exists(tx_path):
        raise FileNotFoundError("Raw datasets not found. Please run data_generation.py first.")
        
    print("Loading raw datasets...")
    customers = pd.read_csv(cust_path)
    transactions = pd.read_csv(tx_path)
    
    # ----------------------------------------------------
    # 1. Clean Datasets
    # ----------------------------------------------------
    print("Cleaning datasets...")
    # Drop duplicates if any
    customers = customers.drop_duplicates(subset=["customer_id"])
    transactions = transactions.drop_duplicates(subset=["transaction_id"])
    
    # Convert dates
    customers["join_date"] = pd.to_datetime(customers["join_date"])
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])
    
    # Fill missing values (if any exist - good practice)
    customers["satisfaction_score"] = customers["satisfaction_score"].fillna(3).astype(int)
    customers["annual_income"] = customers["annual_income"].fillna(customers["annual_income"].median()).astype(int)
    customers["age"] = customers["age"].fillna(customers["age"].median()).astype(int)
    
    # ----------------------------------------------------
    # 2. Transaction Feature Engineering
    # ----------------------------------------------------
    print("Engineering transaction-level features...")
    # Calculate net revenue per transaction
    transactions["revenue"] = (
        transactions["price"] * 
        transactions["quantity"] * 
        (1 - transactions["discount_applied"])
    )
    
    # Date parts
    transactions["year"] = transactions["transaction_date"].dt.year
    transactions["month"] = transactions["transaction_date"].dt.month
    transactions["quarter"] = transactions["transaction_date"].dt.quarter
    transactions["day_of_week"] = transactions["transaction_date"].dt.day_name()
    transactions["year_month"] = transactions["transaction_date"].dt.to_period("M").astype(str)
    
    # Add season
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"
            
    transactions["season"] = transactions["month"].apply(get_season)
    
    # Save cleaned transactions
    clean_tx_path = os.path.join(processed_dir, "cleaned_transactions.csv")
    transactions.to_csv(clean_tx_path, index=False)
    print(f"Cleaned transactions saved to {clean_tx_path}")
    
    # ----------------------------------------------------
    # 3. Customer-Level Aggregation
    # ----------------------------------------------------
    print("Aggregating customer behavior profiles...")
    
    # Base metrics from transactions
    tx_grouped = transactions.groupby("customer_id").agg(
        total_spend=("revenue", "sum"),
        total_transactions=("transaction_id", "count"),
        total_items_purchased=("quantity", "sum"),
        avg_discount_received=("discount_applied", "mean"),
        last_purchase_date=("transaction_date", "max"),
        first_purchase_date=("transaction_date", "min")
    ).reset_index()
    
    tx_grouped["avg_order_value"] = tx_grouped["total_spend"] / tx_grouped["total_transactions"]
    
    # Determine the preferred product category for each customer (the category they spend the most on)
    cat_spend = transactions.groupby(["customer_id", "product_category"])["revenue"].sum().reset_index()
    preferred_cat = cat_spend.sort_values("revenue", ascending=False).drop_duplicates("customer_id")
    preferred_cat = preferred_cat.rename(columns={"product_category": "preferred_category"}).drop(columns=["revenue"])
    
    # Merge aggregations back to customer demographics
    customer_profile = pd.merge(customers, tx_grouped, on="customer_id", how="left")
    customer_profile = pd.merge(customer_profile, preferred_cat, on="customer_id", how="left")
    
    # For customers who haven't made any purchases, fill with defaults
    customer_profile["total_spend"] = customer_profile["total_spend"].fillna(0)
    customer_profile["total_transactions"] = customer_profile["total_transactions"].fillna(0)
    customer_profile["total_items_purchased"] = customer_profile["total_items_purchased"].fillna(0)
    customer_profile["avg_discount_received"] = customer_profile["avg_discount_received"].fillna(0)
    customer_profile["avg_order_value"] = customer_profile["avg_order_value"].fillna(0)
    customer_profile["preferred_category"] = customer_profile["preferred_category"].fillna("None")
    
    # Calculate Recency (days since last purchase relative to the max transaction date in the entire dataset)
    max_tx_date = transactions["transaction_date"].max()
    
    def calculate_recency(date):
        if pd.isna(date):
            return 365 * 3 # default value for no purchases
        return (max_tx_date - pd.to_datetime(date)).days
        
    customer_profile["recency_days"] = customer_profile["last_purchase_date"].apply(calculate_recency)
    
    # Calculate Tenure (days registered)
    customer_profile["tenure_days"] = customer_profile["join_date"].apply(lambda d: (max_tx_date - d).days)
    
    # Handle negative tenures if join date is somehow after max transaction
    customer_profile["tenure_days"] = customer_profile["tenure_days"].clip(lower=1)
    
    # Frequency: transactions per 100 days of tenure (normalized)
    customer_profile["purchase_frequency_normalized"] = (
        (customer_profile["total_transactions"] / customer_profile["tenure_days"]) * 100
    )
    
    # Label potential churn: Recency > 180 days (6 months) is high risk
    customer_profile["churn_status"] = np.where(customer_profile["recency_days"] > 180, "Churned", "Active")
    
    # Save profiles
    profile_path = os.path.join(processed_dir, "customer_profiles.csv")
    customer_profile.to_csv(profile_path, index=False)
    print(f"Customer profiles saved to {profile_path}")
    print(f"Processing complete! Profiles shape: {customer_profile.shape}")

if __name__ == "__main__":
    # If executed stand-alone, we expect to find data/raw folder
    clean_and_process_data()
