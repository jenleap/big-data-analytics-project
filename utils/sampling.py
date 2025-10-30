import pandas as pd
import numpy as np
from categorize import categorize_by_distribution


def order_sampling(orders_df, sample_size):
    # Randomly sample order_ids
    order_ids = orders_df['order_id'].unique()
    sample_size = int(len(order_ids) * sample_size) 
    sampled_order_ids = pd.Series(order_ids).sample(sample_size, random_state=42)

    # Filter to those orders
    orders_sample = orders_df[orders_df['order_id'].isin(sampled_order_ids)]
    return orders_sample

def order_sampling_by_num(sample_size):
    orders_df = pd.read_csv('../data/intermediary/order-products-enhanced.csv')
    # Randomly sample order_ids
    order_ids = orders_df['order_id'].unique() 
    sampled_order_ids = pd.Series(order_ids).sample(sample_size, random_state=42)

    # Filter to those orders
    orders_sample = orders_df[orders_df['order_id'].isin(sampled_order_ids)]
    return orders_sample


def stratified_product_sampling_fixed_size(
    products_df,
    target_sample_size=5000,
    min_orders=50,
    random_state=42
):
    """
    Sample an exact number of products for probability/lift calculations,
    stratified by quartile/IQR/outlier buckets.
    
    products_df: DataFrame with ['product_id', 'count']
    target_sample_size: total number of products to sample
    min_orders: minimum number of orders a product must have
    random_state: for reproducibility
    
    Returns: List of sampled product_ids
    """
    
    # Step 1: Filter products by minimum orders
    products_df = products_df[products_df['count'] >= min_orders].copy()
    
    # Step 2: Categorize into buckets
    products_df['bucket'] = categorize_by_distribution(products_df['count'])
    
    # Step 3: Determine proportion of products in each bucket
    bucket_counts = products_df['bucket'].value_counts(normalize=True).to_dict()
    
    sampled_products = []
    
    # Step 4: Sample proportionally from each bucket
    for bucket, proportion in bucket_counts.items():
        bucket_products = products_df[products_df['bucket'] == bucket]['product_id']
        n_sample = max(1, int(round(proportion * target_sample_size)))
        n_sample = min(n_sample, len(bucket_products))  # cannot sample more than available
        sampled_products.extend(bucket_products.sample(n=n_sample, random_state=random_state).tolist())
    
    # Step 5: Adjust in case rounding caused fewer or more than target_sample_size
    if len(sampled_products) < target_sample_size:
        remaining = target_sample_size - len(sampled_products)
        remaining_products = products_df[~products_df['product_id'].isin(sampled_products)]['product_id']
        if len(remaining_products) > 0:
            sampled_products.extend(
                remaining_products.sample(n=min(remaining, len(remaining_products)), random_state=random_state).tolist()
            )
    elif len(sampled_products) > target_sample_size:
        # Randomly trim to target size
        sampled_products = np.random.RandomState(random_state).choice(sampled_products, target_sample_size, replace=False).tolist()
    
    print(f"Sampled {len(sampled_products)} products from {len(products_df)} eligible products.")
    return sampled_products
