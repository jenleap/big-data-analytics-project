import time
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from pairwise import compute_pairwise_probabilities_sample


def get_min_pij(min_co_occurrences=5):
    orders_full_df = pd.read_csv('../dataset/order_products__prior.csv')
    num_orders = orders_full_df['order_id'].nunique()
    min_pij = min_co_occurrences / num_orders
    return num_orders, min_pij

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

    #for pid in tqdm(sampled_products, desc="Computing lift"):
    for pid in sampled_products:
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
            lambda r: ((r['P_ij'] / r['P_i'] - r['P_j']) /
                    (r['P_ij'] / r['P_i'] + r['P_j'] + eps))
            if r['P_i'] > 0 else 0,
            axis=1
        )


        rows.append(combined[['product_i','product_j','P_ij','co_count','lift','b_complementarity']])

    if not rows:
        return pd.DataFrame(columns=['product_i','product_j','P_ij','co_count','lift','b_complementarity'])
    return pd.concat(rows, ignore_index=True)


def compute_hybrid_score(pairwise_df, focus_products, top_n=None):
    """
    Compute hybrid score = normalized(lift) * normalized(b_complementarity)
    Optionally return top_n complements per product.
    """
    from sklearn.preprocessing import MinMaxScaler

    df = pairwise_df.copy()

    df = df.drop_duplicates(subset=['product_i', 'product_j'])

    # filter focus products
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


def build_complement_network(complements_df, weight_col="hybrid_score"):
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
    weight_col="hybrid_score",
    r=0.3, 
    lambda_pr=0.5,
    lambda_deg=0.3,
    lambda_bc=0.2
):
    
    complements_df = complements_df.drop_duplicates(subset=['product_id', 'complement_id'])
    pairwise_df = pairwise_df.drop_duplicates(subset=['product_i', 'product_j'])
    # Step 1: Merge pairwise probabilities
    df = complements_df.merge(
        pairwise_df,
        left_on=['product_id', 'complement_id'],
        right_on=['product_i', 'product_j'],
        how='left'
    ) 

    df['P_j_given_i'] = df['P_ij'] / df['P_i']
    df['impact_pct'] = 100 * df['P_j_given_i']

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
    df['impact_index'] = (
        r * df['P_j_given_i'] *
        (1 +
         lambda_pr * df['pagerank_j'] +
         lambda_deg * df['indegree_j'] +
         lambda_bc * df['betweenness_j'])
    )

    # Normalize
    df['impact_index'] = df['impact_index'] / df['impact_index'].max()

    return df.sort_values(
        ['product_i', 'impact_index'],
        ascending=[True, False]
    )


def compute_total_impact(
    pairwise_impact_df,
    penetration_df,
    product_i_col="product_id",
    product_j_col="complement_id",
    impact_col="impact_index",
    penetration_col="order_penetration_pct",   # if percent (0-100). function will detect and convert
    top_k=5,
    min_penetration=1e-6,
    min_impact=0.0,
):
    """
    Compute total impact per product i using penetration of complements (j).

    Inputs:
    - pairwise_impact_df: DataFrame with columns [product_i, product_j, impact_index, ...]
      (impact_col should already be computed, either pairwise or network-enhanced)
    - penetration_df: DataFrame or Series with product j penetration info.
      If DataFrame, must have columns [product_id, penetration_col].
      If Series, index=product_id, values=penetration (pct or fraction).
    - top_k: how many top complements to include (only used for topk_weighted)
    - min_penetration: floor to avoid dividing by zero / tiny weights
    - min_impact: filter small pairwise impacts

    Returns:
    - totals_df: DataFrame with columns ['product_i', 'total_impact', 'total_weight', ...]
    """

    # normalize penetration input to series indexed by product id
    if isinstance(penetration_df, pd.DataFrame):
        if 'product_id' in penetration_df.columns:
            pen = penetration_df.set_index('product_id')[penetration_col].astype(float)
        else:
            # assume first column is product id
            pen = penetration_df.set_index(penetration_df.columns[0])[penetration_col].astype(float)
    elif isinstance(penetration_df, pd.Series):
        pen = penetration_df.astype(float)
    else:
        raise ValueError("penetration_df must be DataFrame or Series with product id index/column")

    # If penetration looks like percent (max > 1), convert to fraction
    if pen.max() > 1.0:
        pen = pen / 100.0

    # Merge pairwise impacts with penetration of j
    merged = pairwise_impact_df[[product_i_col, product_j_col, impact_col]].copy()
    merged = merged.merge(
        pen.rename("penetration"),
        left_on=product_j_col,
        right_index=True,
        how="left"
    )

    # Fill missing penetration with tiny value
    merged['penetration'] = merged['penetration'].fillna(0.0).clip(lower=min_penetration)

    # Filter tiny impacts if requested
    merged = merged[merged[impact_col] >= min_impact].copy()

    # Compute contribution per pair (i->j)
    merged['contribution'] = merged[impact_col] * merged['penetration']

    merged = merged.sort_values([product_i_col, impact_col], ascending=[True, False])
    # rank within product_i
    merged['rank_within_i'] = merged.groupby(product_i_col)[impact_col].rank(method="first", ascending=False)
    merged = merged[merged['rank_within_i'] <= top_k].copy()

    # Aggregate per product_i
    agg = merged.groupby(product_i_col).agg(
        total_impact = ('contribution', 'sum'),
        total_weight = ('penetration', 'sum'),
        n_complements = (product_j_col, 'nunique'),
        mean_pair_impact = (impact_col, 'mean'),
    ).reset_index()

    maxv = agg['total_impact'].max()
    if maxv > 0:
        agg['total_impact_norm'] = agg['total_impact'] / maxv
    else:
        agg['total_impact_norm'] = 0.0

    return agg

def simulate_removal_exact(pids, orders_df, focus_products, top_n=10, limit=5):
    """
    Exact recompute of network-enhanced impact after removing a product.

    Parameters:
    - pids: list of product_ids to remove
    - pairwise_df: original pairwise probabilities dataframe
    - focus_products: list of products to compute hybrid score for
    - top_n: number of top complements to include in hybrid score
    - limit: max number of products to remove

    Returns:
    - DataFrame with product removed and updated metrics
    """
    rows = []

    for idx, pid in enumerate(pids):
        if idx >= limit:
            break

        filtered_orders = orders_df[orders_df['product_id'] != pid]

        pairwise_df = compute_pairwise_probabilities_sample(filtered_orders, focus_products, batch_size=1000)

        num_orders = filtered_orders['order_id'].nunique()
        min_pij = 5 / num_orders

        try:
            # Re-run the complements pipeline on remaining products
            lift_remaining = compute_lift(focus_products, pairwise_df, min_pij=min_pij, total_orders=num_orders)

            if lift_remaining.empty:
                rows.append({
                    'removed_product': int(pid),
                    'avg_total_impact': 0.0,
                    'remaining_num_products': 0,
                    'sum_neighbor_CII_before': 0.0,
                    'avg_neighbor_CII_before': 0.0
                })
                continue

            complements_remaining = compute_hybrid_score(lift_remaining, focus_products, top_n=top_n)
            network_remaining = compute_network_enhanced_impact(complements_remaining, pairwise_df)

            if 'impact_index' not in network_remaining.columns:
                network_remaining['impact_index'] = network_remaining['total_lift'] \
                    if 'total_lift' in network_remaining.columns else 0.0

            if 'P_ij' not in network_remaining.columns and 'P_ij' in lift_remaining.columns:
                network_remaining = network_remaining.merge(
                    lift_remaining[['product_i','product_j','P_ij']],
                    left_on=['product_id','complement_id'],
                    right_on=['product_i','product_j'],
                    how='left'
                ).drop(columns=['product_i','product_j'], errors='ignore')

            # Handle empty network_remaining
            if network_remaining.empty:
                avg_impact = 0.0
                num_products = 0
                cii_map = {}
            else:
                avg_impact = float(network_remaining['impact_index'].mean())
                num_products = network_remaining['product_id'].nunique()
                cii_map = network_remaining.groupby('product_id')['impact_index'].first().to_dict()

            # Compute neighbors only connected to removed product
            mask = (network_remaining['product_id'] == pid) | (network_remaining['complement_id'] == pid)
            direct_edges = network_remaining.loc[mask]
            neighbors = set(direct_edges['product_id'].tolist() + direct_edges['complement_id'].tolist())
            neighbors.discard(pid)

            sum_neighbor_cii_before = sum(cii_map.get(n, 0.0) for n in neighbors)
            avg_neighbor_cii_before = (sum_neighbor_cii_before / len(neighbors)) if neighbors else 0.0

            rows.append({
                'removed_product': int(pid),
                'avg_total_impact': avg_impact,
                'remaining_num_products': num_products,
                'sum_neighbor_CII_before': sum_neighbor_cii_before,
                'avg_neighbor_CII_before': avg_neighbor_cii_before
            })

        except Exception as e:
            rows.append({'removed_product': int(pid), 'avg_total_impact': 0.0, 
                         'remaining_num_products': 0, 'sum_neighbor_CII_before': 0.0,
                         'avg_neighbor_CII_before': 0.0, 'error': str(e)})

    return pd.DataFrame(rows)


# Delisting simulation
#    - fast (local-impact): edges removed, neighbors affected, sum neighbor CII before
def simulate_removal_fast(pids, complements_df):
    rows = []
    # precompute neighbor CII lookup (CII per product_id)
    cii_map = complements_df.groupby('product_id')['impact_index'].first().to_dict()
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

# Validation (temporal, correlations, delisting fast, optional exact)
def validate_complements_pipeline(complements_df, lift_df, orders_df, orders_test_df, pairwise_df, focus_products,
                                  top_n=5, max_exact_delist=5):
    """
    complements_df: output from compute_complements_and_cii (must contain P_ij and CII)
    lift_df: the raw pairwise lift dataframe (product_i, product_j, P_ij, lift)
    orders_test_df: order-level test data with ['order_id','product_id']
    delist_recompute: if True, do an exact (slow) recompute of CII after removing a product.
                      Defaults to False (fast local-impact method).
    max_exact_delist: limit number of exact recomputes (safety).
    """
    results = {}
    start = time.time()

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


    # Temporal validation (Precision@N, Recall@N, Hit Rate, Coverage)
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

    # CII correlations (construct validity)
    metrics = complements_df[['product_id','impact_index']].drop_duplicates(subset=['product_id']).set_index('product_id')
    metrics = metrics.join(complements_df.groupby('product_id')['pagerank_j'].first().rename('pagerank'))

    results['correlations'] = {
        'CII_vs_total_lift': float(metrics['impact_index'].corr(metrics['total_lift'])) 
                            if 'total_lift' in metrics.columns and not metrics['total_lift'].isna().all() else np.nan,
        'CII_vs_pagerank': float(metrics['impact_index'].corr(metrics['pagerank'])) 
                        if 'pagerank' in metrics.columns and not metrics['pagerank'].isna().all() else np.nan
    }

    # --- Delisting simulations ---
    top_products = complements_df[['product_id','impact_index']].drop_duplicates().nlargest(top_n,'impact_index')['product_id'].tolist()
    bottom_products = complements_df[['product_id','impact_index']].drop_duplicates().nsmallest(top_n,'impact_index')['product_id'].tolist()

    delist_bottom = simulate_removal_exact(bottom_products,orders_df, focus_products, limit=max_exact_delist)
    delist_top = simulate_removal_exact(top_products, orders_df, focus_products, limit=max_exact_delist)

    # Assign risk categories
    def assign_risk(avg_impact, avg_neighbor_cii):
        if avg_impact > 0.05 or avg_neighbor_cii > 0.05:
            return 'HIGH'
        elif avg_impact > 0.02 or avg_neighbor_cii > 0.02:
            return 'MEDIUM'
        else:
            return 'LOW'

    for df in [delist_top, delist_bottom]:
        df['risk_category'] = df.apply(lambda x: assign_risk(x['avg_total_impact'], x['avg_neighbor_CII_before']), axis=1)

    # Combined for easy export/inspection
    combined_delist = pd.concat([delist_top, delist_bottom], ignore_index=True)

    # weighted combination: impact + avg_neighbor_CII_before
    combined_delist['effectiveness_score'] = (
        0.7 * combined_delist['avg_total_impact'].fillna(0) +
        0.3 * combined_delist['avg_neighbor_CII_before']
    )

    # Optional: categorize
    def effectiveness_category(score):
        if score > 0.05:
            return 'HIGH'
        elif score > 0.02:
            return 'MEDIUM'
        else:
            return 'LOW'

    combined_delist['effectiveness_category'] = combined_delist['effectiveness_score'].apply(effectiveness_category)

    results['delisting'] = {
        'top': delist_top,
        'bottom': delist_bottom,
        'combined': combined_delist
    }

    results['runtime_seconds'] = time.time() - start

    return results


def run_samples_validation(pairwise_train_df, pairwise_test_df, orders_df, focus_products, orders_test_df, num_orders,min_pij,
                        num_samples=3, sample_fraction=0.02, top_n=5):
    """
    Example pipeline: sample products, compute lift, compute complements+CII, validate.
    sample_fraction default 2% for large catalogs (50k -> ~1k products).
    """
    all_products = pd.concat([pairwise_train_df['product_i'], pairwise_train_df['product_j']]).unique()
    n_products = len(all_products)
    sample_size = max(50, int(n_products * sample_fraction))  # guard minimum

    results_summary = []
    for i in range(int(num_samples)):
        print(f"\n--- Sample {i+1}/{num_samples} (size={sample_size}) ---")
        sampled = np.random.choice(focus_products, size=sample_size, replace=False)

        t_comp = time.time()
        lift_df = compute_lift(sampled, pairwise_train_df, min_pij=min_pij, total_orders=num_orders)   
        complements_df = compute_hybrid_score(lift_df, focus_products, top_n=10)
        network_df = compute_network_enhanced_impact(complements_df, pairwise_train_df)
        complement_runtime = time.time() - t_comp
        print(f"Complement pipeline time: {complement_runtime:.1f}s, rows={len(network_df)}")

        # ensure P_ij is present (it is included by function, but double-check)
        if 'P_ij' not in network_df.columns and 'P_ij' in lift_df.columns:
            network_df = network_df.merge(
                lift_df[['product_i','product_j','P_ij']],
                left_on=['product_id','complement_id'],
                right_on=['product_i','product_j'],
                how='left'
            ).drop(columns=[c for c in ['product_i','product_j'] if c in network_df.columns or c in lift_df.columns], errors='ignore')

        # Validate (fast delisting by default)
        val = validate_complements_pipeline(
            complements_df=network_df,
            lift_df=lift_df,
            orders_df=orders_df,
            orders_test_df=orders_test_df,
            pairwise_df=pairwise_train_df,
            focus_products=sampled,
            top_n=top_n
        )

        val['complement_runtime'] = complement_runtime

        results_summary.append(val)
        print(f"Validation (sample {i+1}) summary:", val.get('temporal', {}))

    # Aggregate temporal metrics
    temporal_df = pd.DataFrame([r['temporal'] for r in results_summary])
    temporal_stats = temporal_df.agg(['mean','std'])
    print("\nTemporal metrics across samples (mean ± std):\n", temporal_stats)

    # Aggregate delisting metrics
    delist_combined = pd.concat([r['delisting']['combined'] for r in results_summary], ignore_index=True)
    delist_summary = delist_combined.groupby('removed_product').agg(
        avg_total_impact=('avg_total_impact','mean'),
        std_total_impact=('avg_total_impact','std'),
        avg_neighbor_CII_before=('avg_neighbor_CII_before','mean'),
        std_neighbor_CII_before=('avg_neighbor_CII_before','std')
    ).reset_index()

    delist_summary['effectiveness_score'] = (
        0.7 * delist_summary['avg_total_impact'].fillna(0) +
        0.3 * delist_summary['avg_neighbor_CII_before']
    )

    print("\n===============================================")
    print("        COMPLEMENTS PIPELINE – SUMMARY         ")
    print("===============================================")

    # ---- Timing summary ----
    comp_times = [r['complement_runtime'] for r in results_summary]

    print("\n--- Runtime Performance ---")
    print(f"Avg complements pipeline time:    {np.mean(comp_times):.2f}s  (std={np.std(comp_times):.2f})")

    # ---- Temporal metrics ----
    print("\n--- Temporal Metrics (Mean ± Std) ---")
    for metric in temporal_stats.columns:
        mean_val = temporal_stats.loc['mean', metric]
        std_val  = temporal_stats.loc['std', metric]
        print(f"{metric:20s}: {mean_val:.4f} ± {std_val:.4f}")

    # ---- Delisting / effectiveness summary ----
    print("\n--- Complement Impact Summary (Top 10 Risky Products) ---")
    top10 = delist_summary.sort_values('effectiveness_score', ascending=False).head(10)
    print(top10[['removed_product','avg_total_impact','avg_neighbor_CII_before','effectiveness_score']])

    # ---- Stability summary ----
    print("\n--- Stability Across Samples ---")
    print(f"Temporal metric stability (avg coefficient of variation): "
        f"{(temporal_stats.loc['std'] / temporal_stats.loc['mean']).mean():.3f}")

    print("===============================================\n")

    return {
        'per_sample': results_summary,
        'temporal_stats': temporal_stats,
        'delist_summary': delist_summary
    }


def validate_network_total_impact(network_df, total_impact_df=None,
                                  top_n=10, check_network=True, check_total=True,
                                  check_consistency=True):
    """
    Validation for network-enhanced CII and total impact with optional consistency check.

    Parameters
    ----------
    network_df : pd.DataFrame
        Output from compute_network_enhanced_impact or compute_complement_impact_index,
        must contain columns ['product_id', 'complement_id', 'CII_network'] or similar.
    total_impact_df : pd.DataFrame or None
        Output from compute_total_impact. Must contain ['product_id', 'total_impact'] if provided.
    top_n : int
        Number of top products to check for sanity ranking.
    check_network : bool
        Whether to validate network-enhanced CII values.
    check_total : bool
        Whether to validate total impact values.
    check_consistency : bool
        Whether to check alignment between top products by CII and total impact.
    
    Returns
    -------
    dict
        Dictionary containing validation results and summary stats.
    """
    results = {}

    # -------------------------
    # 1. Network CII checks
    # -------------------------
    if check_network:
        if 'CII_network' not in network_df.columns:
            raise ValueError("network_df must contain 'CII_network' column for network validation.")

        # Aggregate CII per product
        cii_vals = network_df.groupby('product_id')['CII_network'].mean()
        results['network_CII'] = {
            'mean_CII': float(cii_vals.mean()),
            'min_CII': float(cii_vals.min()),
            'max_CII': float(cii_vals.max()),
            'top_products': cii_vals.nlargest(top_n).to_dict()
        }

        # Check for negatives or NaNs
        n_negative = (cii_vals < 0).sum()
        n_missing = cii_vals.isna().sum()
        results['network_CII']['negatives_count'] = int(n_negative)
        results['network_CII']['missing_count'] = int(n_missing)

    # -------------------------
    # 2. Total impact checks
    # -------------------------
    if check_total and total_impact_df is not None:
        if 'total_impact' not in total_impact_df.columns:
            raise ValueError("total_impact_df must contain 'total_impact' column for total impact validation.")

        ti_vals = total_impact_df['total_impact']
        results['total_impact'] = {
            'mean_total_impact': float(ti_vals.mean()),
            'min_total_impact': float(ti_vals.min()),
            'max_total_impact': float(ti_vals.max()),
            'top_products': total_impact_df.nlargest(top_n, 'total_impact')[['product_id','total_impact']].set_index('product_id')['total_impact'].to_dict()
        }

        # Check for negatives or NaNs
        n_negative = (ti_vals < 0).sum()
        n_missing = ti_vals.isna().sum()
        results['total_impact']['negatives_count'] = int(n_negative)
        results['total_impact']['missing_count'] = int(n_missing)

    # -------------------------
    # 3. Consistency check between network CII and total impact
    # -------------------------
    if check_consistency and check_network and check_total and total_impact_df is not None:
        # Get top product IDs
        top_cii_ids = set(cii_vals.nlargest(top_n).index)
        top_impact_ids = set(total_impact_df.nlargest(top_n, 'total_impact')['product_id'])
        # Overlap fraction
        overlap = top_cii_ids & top_impact_ids
        results['consistency'] = {
            'top_n': top_n,
            'overlap_count': len(overlap),
            'overlap_fraction': len(overlap)/top_n,
            'missing_from_CII_top': list(top_impact_ids - top_cii_ids),
            'missing_from_total_impact_top': list(top_cii_ids - top_impact_ids)
        }

    return results
