import pandas as pd
import numpy as np


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

def order_sampling_with_products(sample_size, products_df):
    """
    Sample orders ensuring they contain products from a given products_df.

    sample_size: number of unique orders to sample
    products_df: DataFrame containing at least ['product_id'] to filter orders

    Returns: DataFrame of sampled orders
    """
    orders_df = pd.read_csv('../data/cleaned/order-products-full.csv')
    
    # Step 1: Filter orders to only those containing products in products_df
    eligible_orders_df = orders_df[orders_df['product_id'].isin(products_df['product_id'])]
    
    if eligible_orders_df.empty:
        print("No orders contain the specified products.")
        return pd.DataFrame()  # return empty DF if no eligible orders
    
    # Step 2: Sample unique order_ids
    order_ids = eligible_orders_df['order_id'].unique()
    sample_size = min(sample_size, len(order_ids))  # adjust if fewer orders available
    sampled_order_ids = pd.Series(order_ids).sample(sample_size, random_state=42)
    
    # Step 3: Filter orders to sampled order_ids
    orders_sample = eligible_orders_df[eligible_orders_df['order_id'].isin(sampled_order_ids)]
    
    print(f"Sampled {len(sampled_order_ids)} orders covering {orders_sample['product_id'].nunique()} products.")
    return orders_sample


def average_product_sampling(sample_size):
    products_df = pd.read_csv('../data/cleaned/product-info.csv')

    # Products that fall in the interquartile range
    average_products_df = products_df[products_df['orders_per_product_cat'] > 2].copy()
    product_ids = average_products_df['product_id'].unique() 

    sampled_product_ids = pd.Series(product_ids).sample(sample_size, random_state=42)

    # Filter to those products
    products_sample = average_products_df[average_products_df['product_id'].isin(sampled_product_ids)]
    return products_sample

def product_sampling(sample_size):
    products_df = pd.read_csv('../data/cleaned/product-info.csv')

    # Products that fall in the interquartile range
    average_products_df = products_df[products_df['orders_per_product'] > 50].copy()
    product_ids = average_products_df['product_id'].unique() 

    sampled_product_ids = pd.Series(product_ids).sample(sample_size, random_state=42)

    # Filter to those products
    products_sample = average_products_df[average_products_df['product_id'].isin(sampled_product_ids)]
    return products_sample

def product_sampling_fixed_size(
    products_df,
    target_sample_size=5000,
    min_orders=50,
    random_state=42
):
    """
    Sample an exact number of products for probability/lift calculations,
    without stratifying by distribution.
    
    products_df: DataFrame with ['product_id', 'count']
    target_sample_size: total number of products to sample
    min_orders: minimum number of orders a product must have
    random_state: for reproducibility
    
    Returns: List of sampled product_ids
    """
    
    # Step 1: Filter products by minimum orders
    eligible_products = products_df[products_df['count'] >= min_orders].copy()
    
    if len(eligible_products) == 0:
        raise ValueError("No products meet the minimum order requirement.")
    
    # Step 2: If there are fewer eligible products than the target sample size, take all
    if len(eligible_products) <= target_sample_size:
        sampled_products = eligible_products['product_id'].tolist()
    else:
        # Step 3: Randomly sample target number of products
        sampled_products = (
            eligible_products['product_id']
            .sample(n=target_sample_size, random_state=random_state)
            .tolist()
        )
    
    print(f"Sampled {len(sampled_products)} products from {len(eligible_products)} eligible products.")
    return sampled_products


def prepare_temporal_sampled_data(
    sampled_products,
    orders_df,
    min_user_orders=3,
    test_orders=1,
    expand_cooccurrence=True
):
    """
    Prepare temporally split train/test orders for complement calculations based on sampled products

    Parameters
    ----------
    sampled_products : list
        List of sampled product_ids.

    orders_df : pd.DataFrame
        Must include ['user_id', 'order_id', 'order_number', 'product_id'].

    min_user_orders : int
        Minimum number of orders a user must have to be included.

    test_orders : int
        Number of most recent orders per user to use as test.

    expand_cooccurrence : bool
        If True, retain all products co-occurring in orders containing sampled products
        (preserves realistic complement relationships).

    Returns
    -------

    train_orders : pd.DataFrame
        Temporally split training orders.

    test_orders : pd.DataFrame
        Temporally split test orders.
    """

    # --- Step 2: Filter orders for sampled products ---
    orders_subset = orders_df[orders_df['product_id'].isin(sampled_products)].copy()

    # --- Step 3: Keep users with enough orders ---
    user_order_counts = orders_subset.groupby('user_id')['order_id'].nunique()
    eligible_users = user_order_counts[user_order_counts >= min_user_orders].index
    orders_subset = orders_subset[orders_subset['user_id'].isin(eligible_users)].copy()

    # --- Step 4 (optional): Expand orders to include all co-occurring products ---
    if expand_cooccurrence:
        relevant_order_ids = orders_subset['order_id'].unique()
        orders_subset = orders_df[orders_df['order_id'].isin(relevant_order_ids)].copy()

    # --- Step 5: Temporal split per user ---
    orders_subset = orders_subset.sort_values(['user_id', 'order_number'])
    orders_subset['rank'] = orders_subset.groupby('user_id')['order_number'].rank(method='first')
    max_order = orders_subset.groupby('user_id')['order_number'].transform('max')
    orders_subset['is_test'] = orders_subset['rank'] > (max_order - test_orders)

    train_orders = orders_subset[~orders_subset['is_test']].copy()
    test_orders = orders_subset[orders_subset['is_test']].copy()

    print(f"Sampled {len(sampled_products)} products.")
    print(f"Users retained: {orders_subset['user_id'].nunique()}")
    print(f"Orders retained: {orders_subset['order_id'].nunique()}")
    print(f"Train orders: {train_orders['order_id'].nunique()}, Test orders: {test_orders['order_id'].nunique()}")

    return train_orders, test_orders


def prepare_temporal_sampled_data_full(
    products_df,
    orders_df,
    target_sample_size=5000,
    min_orders=50,
    min_user_orders=3,
    test_orders=1,
    random_state=42,
    expand_cooccurrence=True
):
    """
    Sample products and prepare temporally split train/test orders for complement calculations.

    Parameters
    ----------
    products_df : pd.DataFrame
        Must include ['product_id', 'count', 'department_id', 'aisle_id'].

    orders_df : pd.DataFrame
        Must include ['user_id', 'order_id', 'order_number', 'product_id'].

    target_sample_size : int
        Number of products to sample.

    min_orders : int
        Minimum number of orders a product must appear in to be eligible.

    min_user_orders : int
        Minimum number of orders a user must have to be included.

    test_orders : int
        Number of most recent orders per user to use as test.

    random_state : int
        Random seed for reproducibility.

    expand_cooccurrence : bool
        If True, retain all products co-occurring in orders containing sampled products
        (preserves realistic complement relationships).

    Returns
    -------
    sampled_products : list
        List of sampled product_ids.

    train_orders : pd.DataFrame
        Temporally split training orders.

    test_orders : pd.DataFrame
        Temporally split test orders.
    """
    # --- Step 1: Stratified product sampling (your existing method) ---
    sampled_products = stratified_product_sampling_fixed_size(
        products_df,
        target_sample_size=target_sample_size,
        min_orders=min_orders,
        random_state=random_state
    )

    # --- Step 2: Filter orders for sampled products ---
    orders_subset = orders_df[orders_df['product_id'].isin(sampled_products)].copy()

    # --- Step 3: Keep users with enough orders ---
    user_order_counts = orders_subset.groupby('user_id')['order_id'].nunique()
    eligible_users = user_order_counts[user_order_counts >= min_user_orders].index
    orders_subset = orders_subset[orders_subset['user_id'].isin(eligible_users)].copy()

    # --- Step 4 (optional): Expand orders to include all co-occurring products ---
    if expand_cooccurrence:
        relevant_order_ids = orders_subset['order_id'].unique()
        orders_subset = orders_df[orders_df['order_id'].isin(relevant_order_ids)].copy()

    # --- Step 5: Temporal split per user ---
    orders_subset = orders_subset.sort_values(['user_id', 'order_number'])
    orders_subset['rank'] = orders_subset.groupby('user_id')['order_number'].rank(method='first')
    max_order = orders_subset.groupby('user_id')['order_number'].transform('max')
    orders_subset['is_test'] = orders_subset['rank'] > (max_order - test_orders)

    train_orders = orders_subset[~orders_subset['is_test']].copy()
    test_orders = orders_subset[orders_subset['is_test']].copy()

    print(f"Sampled {len(sampled_products)} products.")
    print(f"Users retained: {orders_subset['user_id'].nunique()}")
    print(f"Orders retained: {orders_subset['order_id'].nunique()}")
    print(f"Train orders: {train_orders['order_id'].nunique()}, Test orders: {test_orders['order_id'].nunique()}")

    return sampled_products, train_orders, test_orders
