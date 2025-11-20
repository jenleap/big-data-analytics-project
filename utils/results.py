from IPython.display import display
import pandas as pd
import numpy as np

def show_all_complements_tables(df):
    """
    Loop through each product_id and print:
      Product Name
      Table of: sub_name, transferability_pct
    """

    # Ensure sorted by product for consistent output
    grouped = df.groupby("product_id")

    for product_id, group in grouped:
        
        product_name = group["product_name"].iloc[0]

        print(f"\n==============================")
        print(f"Product: {product_name}  (ID: {product_id})")
        print("==============================")

        # Build display table
        table = (
            group[["comp_name", "hybrid_score", "aisle"]]
            .sort_values("hybrid_score", ascending=False)
            .reset_index(drop=True)
        )

        display(table)


def show_all_substitute_tables(df):
    """
    Loop through each product_id and print:
      Product Name
      Table of: sub_name, transferability_pct
    """

    # Ensure sorted by product for consistent output
    grouped = df.groupby("product_id")

    for product_id, group in grouped:
        
        product_name = group["product_name"].iloc[0]

        print(f"\n==============================")
        print(f"Product: {product_name}  (ID: {product_id})")
        print("==============================")

        # Build display table
        table = (
            group[["sub_name", "transferability_pct", "aisle"]]
            .sort_values("transferability_pct", ascending=False)
            .reset_index(drop=True)
        )

        display(table)

def simulate_delisting_extended(
    complements_df,
    focus_product_id,
    alpha=None,
    baseline_sales=None,
    transfer_map=None,
    compute_CII_fn=None
):
    """
    Simulate delisting a focus product and compute network + sales impacts.

    Parameters
    ----------
    complements_df : pd.DataFrame
        Must contain columns ['product_id', 'complement_id', 'lift', 'cluster'].
        This is the full complement network *before* delisting.
    focus_product_id : int or str
        Product to simulate removal of.
    alpha : dict (optional)
        Weight coefficients for CII computation (same as compute_complement_impact_index).
    baseline_sales : pd.Series or dict (optional)
        Indexable by product_id returning baseline sales (units or revenue) over your chosen period.
        Example: pd.Series(data=[100.0, 20.0], index=[prodA, prodB])
    transfer_map : pd.DataFrame (optional)
        Optional mapping of expected substitution fractions when focus product is removed.
        Required cols: ['focus_product_id','substitute_id','transfer_pct'].
        transfer_pct are fractions summing <= 1 (rest is 'lost' demand).
    compute_CII_fn : callable (optional)
        Function to compute CII, signature: compute_CII_fn(df, alpha) -> pd.DataFrame with
        columns ['product_id','CII','cohesion'].
        If None, function assumes a global compute_complement_impact_index is available.
    Returns
    -------
    dict with keys:
        'updated_CII' : pd.DataFrame (new CII for remaining products)
        'old_CII' : pd.DataFrame (CII before removal)
        'delta_CII' : pd.DataFrame (merge old/new with pct change)
        'removed_edges' : int
        'cluster_impact' : pd.DataFrame (avg_CII, avg_cohesion per cluster)
        'recovered_sales_by_sub' : pd.DataFrame (substitute_id, recovered_amount)
        'total_recovered_sales' : float
        'estimated_lost_complement_sales' : float
        'baseline_focus_sales' : float or None
        'net_change' : float  # negative means net loss to retailer
    """
    # --- sanity / defaults ---
    if compute_CII_fn is None:
        compute_CII_fn = compute_complement_impact_index  # assumes this exists

    # --- 0. Precompute old CII on full graph ---
    old_CII = compute_CII_fn(complements_df, alpha=alpha).copy()
    old_CII = old_CII[['product_id', 'CII', 'cohesion']]

    # --- 1. Remove focus product from complements_df ---
    remaining_df = complements_df[
        (complements_df['product_id'] != focus_product_id) &
        (complements_df['complement_id'] != focus_product_id)
    ].copy()
    removed_edges = len(complements_df) - len(remaining_df)

    # --- 2. Recompute CII for remaining products ---
    updated_CII = compute_CII_fn(remaining_df, alpha=alpha).copy()
    updated_CII = updated_CII[['product_id', 'CII', 'cohesion']]

    # --- 3. Delta CII (old vs new) ---
    # Some products might have disappeared if they were only connected to the focus product.
    delta = pd.merge(
        old_CII,
        updated_CII,
        on='product_id',
        how='left',
        suffixes=('_old', '_new')
    ).fillna({'CII_new': 0.0, 'cohesion_new': 0.0})
    # percent change: (new - old) / old, handle old==0
    def pct_change(new, old):
        if old == 0:
            return np.nan if new == 0 else np.inf
        return (new - old) / old
    delta['pct_CII_change'] = delta.apply(lambda r: pct_change(r['CII_new'], r['CII_old']), axis=1)

    # --- 4. Cluster-level impact on remaining graph ---
    # Build cluster mapping from remaining_df (product -> cluster). If missing, fall back to old mapping.
    if 'cluster' in remaining_df.columns:
        cluster_map = remaining_df.set_index('product_id')['cluster'].to_dict()
    else:
        cluster_map = complements_df.set_index('product_id').get('cluster', pd.Series()).to_dict()

    cluster_metrics = []
    for cluster_id in remaining_df['cluster'].unique():
        members = [p for p, c in cluster_map.items() if c == cluster_id]
        cluster_products = updated_CII[updated_CII['product_id'].isin(members)]
        if not cluster_products.empty:
            avg_CII = cluster_products['CII'].mean()
            avg_cohesion = cluster_products['cohesion'].mean()
            cluster_metrics.append({
                'cluster': cluster_id,
                'avg_CII': avg_CII,
                'avg_cohesion': avg_cohesion,
                'num_products': len(cluster_products)
            })
    cluster_impact = pd.DataFrame(cluster_metrics)

    # --- 5. Sales / demand impact estimates (requires baseline_sales, transfer_map optional) ---
    baseline_focus_sales = None
    total_recovered = 0.0
    recovered_by_sub = pd.DataFrame(columns=['substitute_id', 'recovered_amount'])

    # baseline_sales must be indexable by product id
    if baseline_sales is not None:
        # coerce to Series
        baseline_sales_s = pd.Series(baseline_sales)
        baseline_focus_sales = float(baseline_sales_s.get(focus_product_id, 0.0))

        # recovered sales via transfer_map
        if transfer_map is not None and not transfer_map.empty:
            # filter for this focus product
            tmap = transfer_map[transfer_map['focus_product_id'] == focus_product_id].copy()
            if 'transfer_pct' not in tmap.columns or 'substitute_id' not in tmap.columns:
                raise ValueError("transfer_map must contain 'substitute_id' and 'transfer_pct' columns")
            tmap['recovered_amount'] = tmap['transfer_pct'] * baseline_focus_sales
            total_recovered = tmap['recovered_amount'].sum()
            recovered_by_sub = tmap[['substitute_id', 'recovered_amount']].reset_index(drop=True)
        else:
            # No explicit transfer map provided -> assume 0 recovered via direct substitution.
            total_recovered = 0.0
            recovered_by_sub = recovered_by_sub

        # Estimate lost complement sales using percent change in CII:
        # interpret pct_CII_change as proportional change in complement demand.
        # Only consider products that were complements of the focus product (pre-removal).
        pre_complements = pd.concat([
            complements_df.loc[complements_df['product_id'] == focus_product_id, 'complement_id'],
            complements_df.loc[complements_df['complement_id'] == focus_product_id, 'product_id']
        ]).unique()

        delta_complements = delta[delta['product_id'].isin(pre_complements)].copy()
        # for stable estimation, treat inf or nan pct results conservatively
        def capped_pct(x):
            if pd.isna(x) or np.isinf(x):
                return 0.0
            return float(x)
        delta_complements['pct_CII_change_capped'] = delta_complements['pct_CII_change'].map(capped_pct)
        # assume proportional change in CII -> proportional change in sales for these complements
        # only negative pct changes represent lost sales (positive are gains)
        def est_sales_change(row):
            pid = row['product_id']
            pct = row['pct_CII_change_capped']
            base = float(baseline_sales_s.get(pid, 0.0))
            # if pct negative => lost_sales positive
            return pct * base
        delta_complements['estimated_sales_change'] = delta_complements.apply(est_sales_change, axis=1)
        # lost complement sales = sum of negative changes (take abs)
        lost_complements = -delta_complements.loc[delta_complements['estimated_sales_change'] < 0, 'estimated_sales_change'].sum()
        # gained complement sales (optional)
        gained_complements = delta_complements.loc[delta_complements['estimated_sales_change'] > 0, 'estimated_sales_change'].sum()

        estimated_lost_complement_sales = float(lost_complements)
    else:
        baseline_sales_s = None
        baseline_focus_sales = None
        total_recovered = None
        recovered_by_sub = recovered_by_sub
        estimated_lost_complement_sales = None

    # --- 6. Net impact ---
    # Interpret baseline_focus_sales as the immediate sales removed from delisting.
    # Net change = -baseline_focus_sales + total_recovered - estimated_lost_complement_sales
    if baseline_focus_sales is not None:
        net_change = -baseline_focus_sales + float(total_recovered) - float(estimated_lost_complement_sales)
    else:
        net_change = None

    return {
        'updated_CII': updated_CII,
        'old_CII': old_CII,
        'delta_CII': delta,
        'removed_edges': removed_edges,
        'cluster_impact': cluster_impact,
        'recovered_sales_by_sub': recovered_by_sub,
        'total_recovered_sales': float(total_recovered) if total_recovered is not None else None,
        'estimated_lost_complement_sales': float(estimated_lost_complement_sales) if estimated_lost_complement_sales is not None else None,
        'baseline_focus_sales': baseline_focus_sales,
        'net_change': float(net_change) if net_change is not None else None
    }

def compute_multi_level_complements(G, max_hops=3, decay=0.5):
    multi_level_edges = []
    for node in G.nodes():
        visited = {node: 1.0}
        frontier = [(node, 1.0, 0)]  # (current_node, current_weight, hop)
        while frontier:
            current, weight, hop = frontier.pop(0)
            if hop >= max_hops:
                continue
            for nbr in G.neighbors(current):
                edge_weight = G[current][nbr]['weight'] * weight * decay
                if nbr not in visited or visited[nbr] < edge_weight:
                    visited[nbr] = edge_weight
                    frontier.append((nbr, edge_weight, hop+1))
                    multi_level_edges.append((node, nbr, edge_weight))
    return pd.DataFrame(multi_level_edges, columns=['product_id', 'complement_id', 'effective_weight'])

