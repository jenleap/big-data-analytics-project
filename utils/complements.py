import time
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

# optional: Louvain community detection
try:
    import community as community_louvain
except Exception:
    community_louvain = None

# -------------------------
# 1) Compute lift (from pairwise probabilities)
# pairwise_df must contain: ['product_i','product_j','P_ij','P_i','P_j']
# sampled_products: list-like of product ids to compute lift for (speeds up work)
# -------------------------
def compute_lift(sampled_products, pairwise_df, total_orders, 
                 min_pij=1e-6, min_pi=1e-4, min_pj=1e-4, min_co_count=5, eps=1e-12):
    """
    Compute lift and b_complementarity for sampled products with robust filtering.
    
    Parameters:
    -----------
    sampled_products : list
        List of product_ids to compute complements for.
    pairwise_df : pd.DataFrame
        DataFrame with columns ['product_i','product_j','P_i','P_j','P_ij']
    total_orders : int
        Total number of orders in the dataset (for co-occurrence count)
    min_pij : float
        Minimum P_ij to include in calculations
    min_pi : float
        Minimum P_i to include in calculations
    min_pj : float
        Minimum P_j to include in calculations
    min_co_count : int
        Minimum number of co-occurrences to include
    eps : float
        Small value to avoid division by zero in b_complementarity
    
    Returns:
    --------
    pd.DataFrame
        Columns: ['product_i','product_j','P_ij','co_count','lift','b_complementarity']
    """
    rows = []
    g_i = pairwise_df.groupby("product_i")
    g_j = pairwise_df.groupby("product_j")

    for pid in tqdm(sampled_products, desc="Computing lift"):
        df_i = g_i.get_group(pid) if pid in g_i.groups else pd.DataFrame()
        df_j = g_j.get_group(pid) if pid in g_j.groups else pd.DataFrame()

        # Reverse df_j so joins align as (product_i, product_j)
        df_j_rev = df_j.rename(columns={
            'product_i': 'product_j',
            'product_j': 'product_i',
            'P_i': 'P_j',
            'P_j': 'P_i'
        })

        combined = pd.concat([df_i, df_j_rev], ignore_index=True, sort=False)
        if combined.empty:
            continue

        # Filter by min P_ij
        combined = combined[combined['P_ij'] >= min_pij]

        # Compute co-occurrence count
        combined['co_count'] = (combined['P_ij'] * total_orders).astype(int)

        # Filter by minimum co-occurrence
        combined = combined[combined['co_count'] >= min_co_count]

        # Filter by minimum product probabilities
        combined = combined[(combined['P_i'] >= min_pi) & (combined['P_j'] >= min_pj)]
        if combined.empty:
            continue

        # Compute lift safely
        combined['lift'] = combined.apply(
            lambda r: (r['P_ij'] / (r['P_i'] * r['P_j'])) if (r['P_i'] * r['P_j']) > 0 else 0,
            axis=1
        )

        # Compute b_complementarity with small epsilon to avoid -1 or division by zero
        combined['b_complementarity'] = combined.apply(
            lambda r: ((r['P_ij']/r['P_j'] - r['P_i']) / (r['P_ij']/r['P_j'] + r['P_i'] + eps))
            if r['P_j'] > 0 else 0,
            axis=1
        )

        rows.append(combined[['product_i','product_j','P_ij','co_count','lift','b_complementarity']])

    if not rows:
        return pd.DataFrame(columns=['product_i','product_j','P_ij','co_count','lift','b_complementarity'])
    return pd.concat(rows, ignore_index=True)

def compute_hybrid_score(pairwise_df, focus_products=None, top_n=None):
    """
    Compute hybrid score = normalized(lift) * normalized(b_complementarity)
    Optionally return top_n complements per product.
    """
    from sklearn.preprocessing import MinMaxScaler

    df = pairwise_df.copy()
    
    # filter focus products
    if focus_products is not None:
        df = df[df['product_i'].isin(focus_products)]

    # normalize lift and b_complementarity
    scaler = MinMaxScaler()
    df[['lift_norm','bcomp_norm']] = scaler.fit_transform(df[['lift','b_complementarity']])
    
    # compute hybrid
    df['hybrid_score'] = df['lift_norm'] * df['bcomp_norm']
    
    # get top_n per product
    if top_n is not None:
        df['rank'] = df.groupby('product_i')['hybrid_score'].rank(method='first', ascending=False)
        df = df[df['rank'] <= top_n].copy()
        df = df.sort_values(['product_i','rank']).reset_index(drop=True)
    
    # rename columns for consistency
    df = df.rename(columns={
        'product_i': 'product_id',
        'product_j': 'complement_id'
    })

    return df[['product_id','complement_id','lift','b_complementarity','hybrid_score']]

def compute_complement_impact_index(complements_df, pairwise_df, r=0.3):
    """
    Computes impact percentage and impact index for complement pairs.

    Inputs:
        complements_df columns:
            product_id (i), complement_id (j)
        pairwise_df columns:
            product_i, product_j, P_i, P_j, P_ij

    Returns:
        DataFrame with:
            product_i, product_j, P_i, P_j, P_ij,
            impact_pct_j,
            impact_index_j
    """

    # Merge pairwise probabilities only for known complements
    df = complements_df.merge(
        pairwise_df,
        left_on=['product_id', 'complement_id'],
        right_on=['product_i', 'product_j'],
        how='left'
    )

    # Probability that j depends on i
    df['P_i_given_j'] = df['P_ij'] / df['P_j']

    # Impact percentage (0–100% of j's demand at risk)
    df['impact_pct_j'] = 100 * df['P_i_given_j']

    # Impact index (0–1 scaled)
    df['impact_index_j'] = r * df['P_i_given_j']

    return df.sort_values(
        ['product_i', 'impact_index_j'],
        ascending=[True, False]
    )

def build_complement_network(complements_df, weight_col="lift"):
    """
    Build weighted directed graph where i -> j means j is a complement of i.
    """
    G = nx.DiGraph()

    for _, r in complements_df.iterrows():
        i = r['product_id']
        j = r['complement_id']
        w = r.get(weight_col, 1.0)
        G.add_edge(i, j, weight=w)

    return G

def compute_network_enhanced_impact(
    complements_df,
    pairwise_df,
    weight_col="lift",
    r=0.3, 
    lambda_pr=0.5,
    lambda_deg=0.3,
    lambda_bc=0.2
):
    # Step 1: Merge pairwise probabilities
    """  df = complements_df.merge(
        pairwise_df,
        left_on=['product_id', 'complement_id'],
        right_on=['product_i', 'product_j'],
        how='left'
    ) """

    df = complements_df.copy()

    df['P_i_given_j'] = df['P_ij'] / df['P_j']
    df['impact_pct_j'] = 100 * df['P_i_given_j']

    # Step 2: Build network
    G = build_complement_network(complements_df, weight_col)

    # Step 3: Network scores
    pr = nx.pagerank(G, weight='weight')
    deg = dict(G.in_degree(weight='weight'))
    bc = nx.betweenness_centrality(G, weight='weight', normalized=True)

    df['pagerank_j'] = df['product_j'].map(pr)
    df['indegree_j'] = df['product_j'].map(deg)
    df['betweenness_j'] = df['product_j'].map(bc)

    # Step 4: Enhanced impact
    df['impact_index_j'] = (
        r * df['P_i_given_j'] *
        (1 +
         lambda_pr * df['pagerank_j'] +
         lambda_deg * df['indegree_j'] +
         lambda_bc * df['betweenness_j'])
    )

    return df.sort_values(
        ['product_i', 'impact_index_j'],
        ascending=[True, False]
    )


# -------------------------
# 3) Validation (temporal, correlations, delisting fast, optional exact)
# -------------------------
def validate_complements_pipeline(complements_df, lift_df, orders_test_df,
                                  top_n=5, alpha=None, track_runtime=False,
                                  delist_recompute=False, max_exact_delist=5):
    """
    complements_df: output from compute_complements_and_cii (must contain P_ij and CII)
    lift_df: the raw pairwise lift dataframe (product_i, product_j, P_ij, lift)
    orders_test_df: order-level test data with ['order_id','product_id']
    delist_recompute: if True, do an exact (slow) recompute of CII after removing a product.
                      Defaults to False (fast local-impact method).
    max_exact_delist: limit number of exact recomputes (safety).
    """
    results = {}
    start = time.time() if track_runtime else None

    # safe checks
    if complements_df.empty:
        raise ValueError("complements_df is empty. Run compute_complements_and_cii first.")
    if 'P_ij' not in complements_df.columns:
        # try merging P_ij from lift_df if possible (lift_df may have product_i/product_j)
        if set(['product_i','product_j','P_ij']).issubset(lift_df.columns):
            complements_df = complements_df.merge(
                lift_df[['product_i','product_j','P_ij']],
                left_on=['product_id','complement_id'],
                right_on=['product_i','product_j'],
                how='left'
            ).drop(columns=[c for c in ['product_i','product_j'] if c in complements_df.columns or c in lift_df.columns], errors='ignore')
        else:
            complements_df['P_ij'] = np.nan

    # -------------------------
    # 1. Temporal validation (Precision@N, Recall@N, Hit Rate, Coverage)
    # -------------------------
    comp_top = complements_df.groupby('product_id').head(top_n).reset_index(drop=True)
    test_orders = orders_test_df.groupby('order_id')['product_id'].apply(set)

    precision_vals, recall_vals, hit_vals = [], [], []
    coverage_count = 0
    for pid, group in comp_top.groupby('product_id'):
        predicted = set(group['complement_id'].tolist())
        true_coocc = set()
        # find orders where pid appears
        for prods in test_orders:
            if pid in prods:
                true_coocc |= (prods - {pid})
        if not true_coocc:
            continue
        coverage_count += 1
        inter = predicted & true_coocc
        precision_vals.append(len(inter) / len(predicted) if predicted else np.nan)
        recall_vals.append(len(inter) / len(true_coocc) if true_coocc else np.nan)
        hit_vals.append(1.0 if inter else 0.0)

    results['temporal'] = {
        'precision@N': float(np.nanmean(precision_vals)) if precision_vals else np.nan,
        'recall@N': float(np.nanmean(recall_vals)) if recall_vals else np.nan,
        'hit_rate': float(np.mean(hit_vals)) if hit_vals else np.nan,
        'coverage_fraction': float(coverage_count / complements_df['product_id'].nunique() if complements_df['product_id'].nunique() else np.nan)
    }

    # -------------------------
    # 2. CII correlations (construct validity)
    # -------------------------
    total_lift = complements_df.groupby('product_id')['lift'].sum().rename('total_lift')
    neighbor_count = complements_df.groupby('product_id')['complement_id'].nunique().rename('neighbor_count')
    metrics = complements_df[['product_id','CII']].drop_duplicates(subset=['product_id']).set_index('product_id')
    metrics = metrics.join(total_lift).join(neighbor_count)
    results['correlations'] = {
        'CII_vs_total_lift': float(metrics['CII'].corr(metrics['total_lift'])) if ('total_lift' in metrics.columns and not metrics['total_lift'].isna().all()) else np.nan,
        'CII_vs_neighbors': float(metrics['CII'].corr(metrics['neighbor_count'])) if ('neighbor_count' in metrics.columns and not metrics['neighbor_count'].isna().all()) else np.nan
    }

    # -------------------------
    # 3. Delisting simulation
    #    - fast (local-impact): edges removed, neighbors affected, sum neighbor CII before
    #    - optional exact recompute (slow): recompute CII for remaining network (limited by max_exact_delist)
    # -------------------------
    def simulate_removal_fast(pids):
        rows = []
        # precompute neighbor CII lookup (CII per product_id)
        cii_map = complements_df.groupby('product_id')['CII'].first().to_dict()
        # Use complement edges to find affected edges quickly
        for pid in pids:
            mask = (complements_df['product_id'] == pid) | (complements_df['complement_id'] == pid)
            direct_edges = complements_df.loc[mask]
            edges_removed = len(direct_edges)
            # neighbors are nodes connected to pid
            neighbors = set(direct_edges['product_id'].tolist() + direct_edges['complement_id'].tolist())
            neighbors.discard(pid)
            sum_neighbor_cii_before = sum(cii_map.get(n, 0.0) for n in neighbors)
            avg_neighbor_cii_before = (sum_neighbor_cii_before / len(neighbors)) if neighbors else 0.0
            rows.append({
                'removed_product': int(pid),
                'edges_removed': int(edges_removed),
                'num_neighbors_affected': int(len(neighbors)),
                'sum_neighbor_CII_before': float(sum_neighbor_cii_before),
                'avg_neighbor_CII_before': float(avg_neighbor_cii_before)
            })
        return pd.DataFrame(rows)

    def simulate_removal_exact(pids, limit=max_exact_delist):
        # exact: remove pid from lift_df (all pairs containing pid) and recompute complements + CII
        rows = []
        for idx, pid in enumerate(pids):
            if idx >= limit:
                break
            remaining_pairs = lift_df[(lift_df['product_i'] != pid) & (lift_df['product_j'] != pid)].copy()
            # recompute complements + cii (expensive!)
            try:
                updated_complements = compute_complements_and_cii(remaining_pairs)
                # get mean CII across remaining products (or NaN if none)
                mean_CII = float(updated_complements['CII'].mean()) if not updated_complements.empty else np.nan
                rows.append({
                    'removed_product': int(pid),
                    'avg_CII_remaining_exact': mean_CII,
                    'remaining_num_products': int(updated_complements['product_id'].nunique()) if not updated_complements.empty else 0
                })
            except Exception as e:
                rows.append({'removed_product': int(pid), 'error': str(e)})
        return pd.DataFrame(rows)

    top_products = complements_df[['product_id','CII']].drop_duplicates().nlargest(top_n,'CII')['product_id'].tolist()
    bottom_products = complements_df[['product_id','CII']].drop_duplicates().nsmallest(top_n,'CII')['product_id'].tolist()

    results['delisting_fast'] = {
        'top': simulate_removal_fast(top_products),
        'bottom': simulate_removal_fast(bottom_products)
    }

    if delist_recompute:
        # do a limited number of exact recomputes (expensive). Default limit = max_exact_delist
        results['delisting_exact'] = {
            'top': simulate_removal_exact(top_products, limit=max_exact_delist),
            'bottom': simulate_removal_exact(bottom_products, limit=max_exact_delist)
        }

    # -------------------------
    # 4. Optional category / aisle alignment
    # -------------------------
    if {'aisle_product','aisle_complement'}.issubset(complements_df.columns):
        aisle_match = complements_df['aisle_product'] == complements_df['aisle_complement']
        results['aisle_alignment_ratio'] = float(aisle_match.mean())

    # -------------------------
    # 5. Runtime
    # -------------------------
    if track_runtime:
        results['runtime_seconds'] = time.time() - start

    return results


# -------------------------
# USAGE example (multiple samples, safe defaults for Instacart)
# -------------------------
def run_samples_example(pairwise_train_df, pairwise_test_df, orders_test_df,
                        num_samples=3, sample_fraction=0.02, top_n=5, alpha=None):
    """
    Example pipeline: sample products, compute lift, compute complements+CII, validate.
    sample_fraction default 2% for large catalogs (50k -> ~1k products).
    """
    all_products = pd.concat([pairwise_train_df['product_i'], pairwise_train_df['product_j']]).unique()
    n_products = len(all_products)
    sample_size = max(50, int(n_products * sample_fraction))  # guard minimum

    results_summary = []
    for i in range(num_samples):
        print(f"\n--- Sample {i+1}/{num_samples} (size={sample_size}) ---")
        sampled = np.random.choice(all_products, size=sample_size, replace=False)

        t0 = time.time()
        lift_df = compute_lift(sampled, pairwise_train_df)   # returns product_i, product_j, P_ij, lift
        print(f"compute_lift: {time.time()-t0:.1f}s, edges: {len(lift_df)}")

        t1 = time.time()
        complements_df = compute_complements_and_cii(lift_df, alpha=alpha)
        print(f"compute_complements_and_cii: {time.time()-t1:.1f}s, complements: {len(complements_df)}")

        # ensure P_ij is present (it is included by function, but double-check)
        if 'P_ij' not in complements_df.columns and 'P_ij' in lift_df.columns:
            complements_df = complements_df.merge(
                lift_df[['product_i','product_j','P_ij']],
                left_on=['product_id','complement_id'],
                right_on=['product_i','product_j'],
                how='left'
            ).drop(columns=[c for c in ['product_i','product_j'] if c in complements_df.columns or c in lift_df.columns], errors='ignore')

        # Validate (fast delisting by default)
        val = validate_complements_pipeline(
            complements_df=complements_df,
            lift_df=lift_df,
            orders_test_df=orders_test_df,
            top_n=top_n,
            alpha=alpha,
            track_runtime=True,
            delist_recompute=False  # default: fast only
        )
        results_summary.append(val)
        print(f"Validation (sample {i+1}) summary:", val.get('temporal', {}))

    return results_summary
