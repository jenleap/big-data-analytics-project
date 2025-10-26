import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

def compute_pairwise_probabilities_chunked(orders_df, output_csv=None, batch_size=1000, min_pij=0.0):
    """
    Memory-safe pairwise probability computation using batched sparse multiplication.
    """

    order_codes, order_uniques = pd.factorize(orders_df['order_id'])
    product_codes, product_uniques = pd.factorize(orders_df['product_id'])
    num_orders = len(order_uniques)
    num_products = len(product_uniques)

    data = np.ones(len(orders_df), dtype=np.int8)
    basket_sparse = csr_matrix((data, (order_codes, product_codes)),
                               shape=(num_orders, num_products))

    p_i_array = np.array(basket_sparse.sum(axis=0)).flatten() / num_orders
    p_i_dict = dict(zip(range(num_products), p_i_array))

    if output_csv:
        with open(output_csv, "w") as f:
            f.write("product_i,product_j,P_i,P_j,P_ij\n")

    for start in tqdm(range(0, num_products, batch_size)):
        end = min(start + batch_size, num_products)

        # compute only this batch of rows × all columns
        co_block = (basket_sparse[:, start:end].T @ basket_sparse).tocoo()

        co_df = pd.DataFrame({
            'product_i_idx': co_block.row + start,
            'product_j_idx': co_block.col,
            'P_ij': co_block.data / num_orders
        })

        # remove self-pairs and low co-occurrence
        co_df = co_df[co_df['product_i_idx'] != co_df['product_j_idx']]
        if min_pij > 0:
            co_df = co_df[co_df['P_ij'] >= min_pij]

        co_df['product_i'] = co_df['product_i_idx'].map(lambda x: product_uniques[x])
        co_df['product_j'] = co_df['product_j_idx'].map(lambda x: product_uniques[x])
        co_df['P_i'] = co_df['product_i_idx'].map(p_i_dict)
        co_df['P_j'] = co_df['product_j_idx'].map(p_i_dict)

        co_df = co_df[['product_i', 'product_j', 'P_i', 'P_j', 'P_ij']]

        # Append to file to avoid large memory usage
        if output_csv:
            co_df.to_csv(output_csv, mode="a", index=False, header=False)

    print(f"✅ Completed computation. Saved to {output_csv if output_csv else 'DataFrame'}")

    if not output_csv:
        return co_df  # return last chunk if you need it
    


