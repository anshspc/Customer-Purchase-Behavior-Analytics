import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_data(num_customers=1000, num_transactions=15000, seed=42):
    np.random.seed(seed)
    
    # ----------------------------------------------------
    # 1. Generate Customers Data
    # ----------------------------------------------------
    print("Generating customer profiles...")
    customer_ids = [f"CUST{str(i).zfill(4)}" for i in range(1, num_customers + 1)]
    
    # Age distribution: normal distribution centered around 40, clipped between 18 and 80
    ages = np.clip(np.random.normal(40, 12, num_customers).astype(int), 18, 80)
    
    # Income distribution: normal centered around 65,000, standard dev 25,000, clipped at min 20,000
    incomes = np.clip(np.random.normal(65000, 25000, num_customers).astype(int), 20000, 220000)
    
    # Genders with specified probabilities
    genders = np.random.choice(["Female", "Male", "Non-binary"], size=num_customers, p=[0.51, 0.45, 0.04])
    
    # Locations
    locations = np.random.choice(
        ["California", "New York", "Texas", "Florida", "Illinois", "Washington", "Colorado", "Ohio"], 
        size=num_customers, 
        p=[0.25, 0.18, 0.15, 0.12, 0.08, 0.08, 0.07, 0.07]
    )
    
    # Join dates over the past 3 years (up to 2026-05-01)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2026, 5, 1)
    days_range = (end_date - start_date).days
    
    join_dates = [start_date + timedelta(days=int(np.random.randint(0, days_range))) for _ in range(num_customers)]
    
    # Satisfaction score (1-5 scale), with higher income slightly more critical, but generally normal-ish
    satisfaction_scores = np.random.choice([1, 2, 3, 4, 5], size=num_customers, p=[0.05, 0.10, 0.20, 0.40, 0.25])
    
    customers_df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "gender": genders,
        "annual_income": incomes,
        "location": locations,
        "join_date": [d.strftime("%Y-%m-%d") for d in join_dates],
        "satisfaction_score": satisfaction_scores
    })
    
    # ----------------------------------------------------
    # 2. Generate Transactions Data
    # ----------------------------------------------------
    print("Generating customer transaction histories...")
    
    # Define products and category information
    products_db = {
        "Electronics": {
            "items": ["Wireless Headphones", "Smartwatch", "Bluetooth Speaker", "Tablet", "Mechanical Keyboard"],
            "prices": [99.99, 199.99, 59.99, 349.99, 89.99],
            "weights": [0.3, 0.2, 0.2, 0.1, 0.2]
        },
        "Apparel": {
            "items": ["Running Shoes", "Hoodie", "Denim Jacket", "T-Shirt", "Sunglasses"],
            "prices": [79.99, 45.00, 65.00, 22.50, 15.00],
            "weights": [0.2, 0.25, 0.15, 0.25, 0.15]
        },
        "Home & Kitchen": {
            "items": ["Coffee Maker", "Air Fryer", "Blender", "Cookware Set", "Water Flask"],
            "prices": [89.99, 119.99, 49.99, 149.99, 29.99],
            "weights": [0.2, 0.2, 0.2, 0.1, 0.3]
        },
        "Books": {
            "items": ["Sci-Fi Novel", "Biography", "Self-Help Book", "Cookbook", "Mystery Thriller"],
            "prices": [14.99, 19.99, 16.50, 24.99, 12.99],
            "weights": [0.25, 0.2, 0.2, 0.15, 0.2]
        },
        "Beauty & Care": {
            "items": ["Skincare Serum", "Moisturizer", "Perfume", "Hair Dryer", "Makeup Palette"],
            "prices": [35.00, 25.00, 75.00, 45.00, 40.00],
            "weights": [0.3, 0.25, 0.15, 0.1, 0.2]
        }
    }
    
    categories = list(products_db.keys())
    category_weights = [0.25, 0.30, 0.18, 0.12, 0.15] # Apparel & Electronics are most popular
    
    tx_data = []
    
    # We will simulate transactions per customer to model behavior
    # Some customers are high-value (frequent, high-spending), others churned, others are moderate.
    
    # Assign customer "types" to make segmentation more interesting
    # 0: Regular/Moderate, 1: High-Value/Loyal, 2: One-time/Churned, 3: Bargain Hunter (high discount influence)
    cust_types = np.random.choice([0, 1, 2, 3], size=num_customers, p=[0.55, 0.15, 0.20, 0.10])
    
    txn_id_counter = 1
    
    for idx, row in customers_df.iterrows():
        c_id = row["customer_id"]
        c_join = join_dates[idx]
        c_type = cust_types[idx]
        c_satisfaction = row["satisfaction_score"]
        
        # Determine number of transactions based on customer profile
        if c_type == 1: # High-Value Loyal
            num_txs = np.random.randint(15, 30)
            purchase_span_days = (end_date - c_join).days
        elif c_type == 2: # One-time / Churned
            num_txs = np.random.randint(1, 4)
            # Most of these transactions happen right after join date
            purchase_span_days = min(60, (end_date - c_join).days)
        elif c_type == 3: # Bargain Hunter
            num_txs = np.random.randint(5, 12)
            purchase_span_days = (end_date - c_join).days
        else: # Regular
            num_txs = np.random.randint(4, 15)
            purchase_span_days = (end_date - c_join).days
            
        if purchase_span_days <= 0:
            purchase_span_days = 1
            
        # Satisfaction affects transaction frequency (lower satisfaction -> fewer repeat visits)
        if c_satisfaction <= 2 and c_type != 1:
            num_txs = max(1, int(num_txs * 0.5))
            purchase_span_days = min(90, purchase_span_days)
            
        # Generate transaction dates
        tx_dates = []
        for _ in range(num_txs):
            random_offset = np.random.randint(0, purchase_span_days)
            t_date = c_join + timedelta(days=random_offset)
            # Clip transaction date to not exceed the final dataset date
            if t_date > end_date:
                t_date = end_date
            tx_dates.append(t_date)
            
        tx_dates.sort()
        
        for t_date in tx_dates:
            # Select category based on customer type
            if c_type == 1: # High value prefers electronics and beauty
                cat = np.random.choice(categories, p=[0.35, 0.20, 0.15, 0.10, 0.20])
            elif c_type == 3: # Bargain hunter prefers clothing and home items
                cat = np.random.choice(categories, p=[0.15, 0.40, 0.20, 0.15, 0.10])
            else:
                cat = np.random.choice(categories, p=category_weights)
                
            db = products_db[cat]
            prod_idx = np.random.choice(len(db["items"]), p=db["weights"])
            prod_name = db["items"][prod_idx]
            base_price = db["prices"][prod_idx]
            
            # Quantity distribution (mostly 1, occasionally up to 3)
            qty = np.random.choice([1, 2, 3, 4], p=[0.70, 0.20, 0.07, 0.03])
            
            # Discounts: Bargain hunters get more discounts.
            # Also simulate seasonal discounts (Nov/Dec, and Jul summer sale)
            month = t_date.month
            is_holiday_season = month in [11, 12]
            is_summer_sale = month == 7
            
            if c_type == 3: # Bargain hunter always hunts for items with discounts
                discount = np.random.choice([0.1, 0.15, 0.2, 0.3], p=[0.2, 0.3, 0.3, 0.2])
            elif is_holiday_season:
                discount = np.random.choice([0.0, 0.1, 0.15, 0.2, 0.25], p=[0.3, 0.3, 0.2, 0.1, 0.1])
            elif is_summer_sale:
                discount = np.random.choice([0.0, 0.05, 0.1, 0.15, 0.2], p=[0.4, 0.3, 0.15, 0.1, 0.05])
            else:
                discount = np.random.choice([0.0, 0.05, 0.1], p=[0.8, 0.15, 0.05])
                
            # Payment method
            pay_method = np.random.choice(
                ["Credit Card", "PayPal", "Debit Card", "Apple Pay", "Google Pay"],
                p=[0.45, 0.25, 0.15, 0.10, 0.05]
            )
            
            tx_data.append({
                "transaction_id": f"TXN{str(txn_id_counter).zfill(6)}",
                "customer_id": c_id,
                "transaction_date": t_date.strftime("%Y-%m-%d %H:%M:%S"),
                "product_category": cat,
                "product_name": prod_name,
                "quantity": qty,
                "price": base_price,
                "discount_applied": discount,
                "payment_method": pay_method
            })
            txn_id_counter += 1

    transactions_df = pd.DataFrame(tx_data)
    
    # Sort transactions by date
    transactions_df["transaction_date"] = pd.to_datetime(transactions_df["transaction_date"])
    transactions_df = transactions_df.sort_values(by="transaction_date").reset_index(drop=True)
    transactions_df["transaction_date"] = transactions_df["transaction_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Save datasets
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    cust_path = os.path.join(raw_dir, "customers.csv")
    tx_path = os.path.join(raw_dir, "transactions.csv")
    
    customers_df.to_csv(cust_path, index=False)
    transactions_df.to_csv(tx_path, index=False)
    
    print(f"Dataset generated successfully!")
    print(f"Generated {len(customers_df)} customer profiles -> saved to {cust_path}")
    print(f"Generated {len(transactions_df)} transactions -> saved to {tx_path}")
    
    # Print basic summary
    total_rev = (transactions_df["price"] * transactions_df["quantity"] * (1 - transactions_df["discount_applied"])).sum()
    print(f"Total simulated gross revenue: ${total_rev:,.2f}")

if __name__ == "__main__":
    generate_data()
