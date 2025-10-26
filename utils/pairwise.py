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
    
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def compute_pairwise_probabilities_old(
    orders_df: pd.DataFrame,
    output_csv: str = None,
    batch_size: int = 1000,
    min_pij: float = 0.0
) -> pd.DataFrame:
    """
    Compute P_i, P_j, P_ij for all co-occurring product pairs in a memory-efficient way.
    Includes both directions: (i,j) and (j,i).
    
    Parameters
    ----------
    orders_df : pd.DataFrame
        Must have columns ['order_id', 'product_id']
    output_csv : str, optional
        If provided, saves results to CSV incrementally
    batch_size : int
        Number of products to process per batch
    min_pij : float
        Minimum co-occurrence probability to keep pair
    
    Returns
    -------
    co_df : pd.DataFrame
        Columns: product_i, product_j, P_i, P_j, P_ij
    """

    # Factorize orders and products
    order_codes, order_uniques = pd.factorize(orders_df['order_id'])
    product_codes, product_uniques = pd.factorize(orders_df['product_id'])
    num_orders = len(order_uniques)
    num_products = len(product_uniques)

    # Build sparse order × product matrix
    data = np.ones(len(orders_df), dtype=np.int8)
    basket_sparse = csr_matrix((data, (order_codes, product_codes)),
                               shape=(num_orders, num_products))

    # Compute marginal probabilities
    p_i_array = np.array(basket_sparse.sum(axis=0)).flatten() / num_orders

    # Prepare incremental storage
    all_results = []

    # Process in batches
    for start_idx in range(0, num_products, batch_size):
        end_idx = min(start_idx + batch_size, num_products)
        batch_results = []

        for i in range(start_idx, end_idx):
            # Sparse row for product i
            row_sparse = basket_sparse[:, i].T @ basket_sparse
            row = row_sparse.toarray().flatten() / num_orders
            row[i] = 0.0  # skip self

            # Filter non-zero / min_pij
            nz_idx = np.where(row >= min_pij)[0]

            for j in nz_idx:
                batch_results.append({
                    'product_i': product_uniques[i],
                    'product_j': product_uniques[j],
                    'P_i': p_i_array[i],
                    'P_j': p_i_array[j],
                    'P_ij': row[j]
                })
                # Add reverse direction
                batch_results.append({
                    'product_i': product_uniques[j],
                    'product_j': product_uniques[i],
                    'P_i': p_i_array[j],
                    'P_j': p_i_array[i],
                    'P_ij': row[j]
                })

        # Convert batch to DataFrame
        batch_df = pd.DataFrame(batch_results)

        if output_csv:
            # Append to CSV
            if start_idx == 0:
                batch_df.to_csv(output_csv, index=False, mode='w')
            else:
                batch_df.to_csv(output_csv, index=False, mode='a', header=False)
        else:
            all_results.append(batch_df)

    if not output_csv:
        co_df = pd.concat(all_results, ignore_index=True)
        return co_df
    else:
        return pd.DataFrame()  # CSV mode only

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def compute_pairwise_probabilities(
    orders_df: pd.DataFrame,
    output_csv: str = None,
    batch_size: int = 1000
) -> pd.DataFrame:
    """
    Compute P_i, P_j, P_ij for all co-occurring product pairs (both directions),
    using a memory-efficient sparse matrix approach with batching.
    """
    
    # Factorize order and product IDs
    order_codes, order_uniques = pd.factorize(orders_df['order_id'])
    product_codes, product_uniques = pd.factorize(orders_df['product_id'])
    num_orders = len(order_uniques)
    num_products = len(product_uniques)
    
    # Build sparse order x product matrix
    data = np.ones(len(orders_df), dtype=np.int8)
    basket_sparse = csr_matrix((data, (order_codes, product_codes)),
                               shape=(num_orders, num_products))
    
    # Compute P_i
    p_i_array = np.array(basket_sparse.sum(axis=0)).flatten() / num_orders
    
    # Prepare results
    results = []
    
    # Batch processing to save memory
    for start_idx in range(0, num_products, batch_size):
        end_idx = min(start_idx + batch_size, num_products)
        batch_rows = basket_sparse[:, start_idx:end_idx].T @ basket_sparse  # batch x products
        batch_rows = batch_rows.astype(np.float32) / num_orders
        
        for local_i, global_i in enumerate(range(start_idx, end_idx)):
            row = batch_rows[local_i].toarray().flatten() if hasattr(batch_rows[local_i], 'toarray') else batch_rows[local_i]
            row[global_i] = 0.0  # skip self
            nz_idx = np.where(row > 0)[0]
            
            for j in nz_idx:
                results.append({
                    'product_i': product_uniques[global_i],
                    'product_j': product_uniques[j],
                    'P_i': p_i_array[global_i],
                    'P_j': p_i_array[j],
                    'P_ij': row[j]
                })
                # symmetric entry
                results.append({
                    'product_i': product_uniques[j],
                    'product_j': product_uniques[global_i],
                    'P_i': p_i_array[j],
                    'P_j': p_i_array[global_i],
                    'P_ij': row[j]
                })
    
    co_df = pd.DataFrame(results)
    
    if output_csv:
        co_df.to_csv(output_csv, index=False)
    
    return co_df




