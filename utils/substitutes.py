import pandas as pd
import numpy as np
import time
import random
from scipy.stats import spearmanr
import psutil, os
from sklearn.metrics import f1_score


def compute_sub_score(product_df, pairwise_df, sampled_products, similarity_df, file_path, weight_si=0.4,
                                      weight_pen=0.1, weight_name=0.6):
    """
    Compute a hybrid substitution score for product pairs and save the top substitutes to a CSV file.

    This function calculates a weighted combination of substitution metrics (substitution index, 
    Jaccard similarity, conditional probability) and applies a similarity weight based on department 
    and aisle alignment to identify likely substitute products. For each product in the sampled list, 
    it ranks potential substitutes and keeps the top 20.

    Parameters:
    -----------
    product_df : pd.DataFrame
        DataFrame containing product metadata, must include 'product_id', 'department_id', 'aisle_id'.
    pairwise_df : pd.DataFrame
        DataFrame of pairwise product metrics, must include 'product_i', 'product_j', 'P_i', 'P_j', 'P_ij'.
    sampled_products : list or pd.Series
        List of product IDs for which to compute substitutes.
    file_path : str
        Path to the CSV file where the top substitutes for all sampled products will be appended.
    weight_si : float, default=0.5
        Weight applied to the substitution index in the hybrid score calculation.
    weight_jc : float, default=0.3
        Weight applied to the Jaccard similarity in the hybrid score calculation (negative weight).
    weight_cond : float, default=0.2
        Weight applied to the conditional probability in the hybrid score calculation (negative weight).
    weight_sim_partial : float, default=0.7
        Similarity weight for product pairs in the same department but different aisle.
    weight_sim_none : float, default=0.4
        Similarity weight for product pairs in different departments.

    Returns:
    --------
    None
        Results are saved directly to the specified CSV file. The CSV includes the following columns:
        - product_id: ID of the reference product
        - substitute_id: ID of the candidate substitute product
        - score: weighted hybrid substitution score
        - rank: rank of substitute for this product (top 20 only)
        - jaccard: normalized Jaccard similarity between products
        - conditional: normalized conditional probability between products
        - substitution_index: normalized substitution index between products
        - same_department: boolean flag if both products are in the same department
        - same_aisle: boolean flag if both products are in the same aisle
        - valid_substitution: boolean flag for substitutes in the same aisle
        - possible_substitution: boolean flag for substitutes in the same department

    Notes:
    ------
    - Pairwise metrics are normalized within each product’s candidate substitutes before weighting.
    - Similarity weight is applied multiplicatively to prioritize same-department and same-aisle pairs.
    - Only the top 20 substitutes per product (by hybrid score) are retained in the output CSV.
    """
        
    with open(file_path, "w") as f:
        f.write("product_id,substitute_id,score,rank,substitution_index\n")

    pairwise_df_i = pairwise_df.groupby("product_i")
    pairwise_df_j = pairwise_df.groupby("product_j")

    for product_id in sampled_products:
        product_probs = pairwise_df_i.get_group(product_id) if product_id in pairwise_df_i.groups else pd.DataFrame()
        product_j_probs = pairwise_df_j.get_group(product_id) if product_id in pairwise_df_j.groups else pd.DataFrame()

        # Reverse relationships, so we have both directions
        products_rev = product_j_probs.rename(columns={
            'product_i': 'product_j',
            'product_j': 'product_i',
            'P_i': 'P_j',
            'P_j': 'P_i'
        })

        product_probs_df = pd.concat([product_probs, products_rev], ignore_index=True)
        if product_probs_df.empty:
            continue

        # Merge product similarity 
        product_probs_df = product_probs_df.merge(
            similarity_df,
            on=['product_i', 'product_j'],
            how='left'
        ).rename(columns={'combined': 'name_similarity'})

        # Fill missing similarities with 0
        product_probs_df['name_similarity'] = product_probs_df['name_similarity'].fillna(0)

         # Merge department and aisle info
        product_probs_df = product_probs_df.merge(
            product_df[['product_id', 'department_id', 'aisle_id']],
            left_on='product_i', right_on='product_id', how='left'
        ).rename(columns={'department_id': 'product_dept', 'aisle_id': 'product_aisle'}).drop(columns=['product_id'])

        product_probs_df = product_probs_df.merge(
            product_df[['product_id', 'department_id', 'aisle_id']],
            left_on='product_j', right_on='product_id', how='left'
        ).rename(columns={'department_id': 'substitute_dept', 'aisle_id': 'substitute_aisle'}).drop(columns=['product_id'])

        # Restrict to same aisle only
        product_probs_df = product_probs_df[product_probs_df["product_aisle"] == product_probs_df["substitute_aisle"]]
        if product_probs_df.empty:
            continue

        # Compute metrics
        product_probs_df['substitution_index'] = product_probs_df.apply(lambda x: ((x.P_i * x.P_j) - x.P_ij) / (x.P_i * x.P_j), axis=1)
        
        # Penetration-based similarity weight (cosine)
        op_map = product_df.set_index("product_id")["order_penetration_pct"].to_dict()
        up_map = product_df.set_index("product_id")["user_penetration_pct"].to_dict()

        product_probs_df["op_i"] = product_probs_df["product_i"].map(op_map)
        product_probs_df["up_i"] = product_probs_df["product_i"].map(up_map)
        product_probs_df["op_j"] = product_probs_df["product_j"].map(op_map)
        product_probs_df["up_j"] = product_probs_df["product_j"].map(up_map)

        # Boost using a combination of order penetration and user penetration of the substitute product
        product_probs_df["penetration_boost"] = 0.5 * product_probs_df["op_j"] + 0.5 * product_probs_df["up_j"]

        # Normalize
        for col in ['substitution_index', 'penetration_boost']:
            min_val = product_probs_df[col].min()
            max_val = product_probs_df[col].max()
            if max_val - min_val != 0:
                product_probs_df[col] = (product_probs_df[col] - min_val) / (max_val - min_val)
            else:
                product_probs_df[col] = 0.0


        # Calculate weighted hybrid score
        linear_score = (
            (weight_si * product_probs_df['substitution_index']) 
            + (weight_name * product_probs_df['name_similarity'])
        )

        product_probs_df['score'] = linear_score * (1 + weight_pen * product_probs_df['penetration_boost'])

        # Rank substitutes
        product_probs_df = product_probs_df.dropna(subset=['score'])
        product_probs_df['rank'] = product_probs_df['score'].rank(method='dense', ascending=False).astype(int)
        
        # Keep only top 20
        top_substitutes_df = product_probs_df[product_probs_df['rank'] <= 20].copy()

        # Rename columns
        top_substitutes_df = top_substitutes_df.rename(columns={
            'product_i': 'product_id',
            'product_j': 'substitute_id'
        })

        # Save results as CSV
        columns_to_save = ['product_id', 'substitute_id', 'score', 'rank','substitution_index']
        top_substitutes_df = top_substitutes_df[columns_to_save]
        top_substitutes_df.to_csv(file_path, mode='a', index=False, header=False)


    print(f"Completed substitute calculations. Saved to {file_path}")


def find_best_threshold(df, score_col='score', label_col='valid_substitution', num_thresholds=100):
    """
    Determine the best threshold for a continuous score to predict binary labels.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the score and label columns.
        score_col (str): Name of the column with continuous scores.
        label_col (str): Name of the binary label column to optimize for (valid_substitution).
        num_thresholds (int): Number of thresholds to test.
        
    Returns:
        float: Best threshold value that maximizes F1 score for the specified label.
    """
    df = df.copy()
    
    # Ensure the label is boolean
    df[label_col] = df[label_col].astype(bool)
    
    # Candidate thresholds
    thresholds = np.linspace(df[score_col].min(), df[score_col].max(), num_thresholds)
    
    best_threshold = thresholds[0]
    best_f1 = -1.0
    
    for t in thresholds:
        pred = df[score_col] >= t
        f1 = f1_score(df[label_col], pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    print(f" Best threshold determined as: {best_threshold:.3f}\n")

    return best_threshold


def compute_transferability(subs_df, top_n):
    """
    Compute transferability % (DTR) for identified substitutes using top-N max-score scaling.
    
    Parameters:
    -----------
    subs_df : pd.DataFrame
        Substitute pairs with columns ['product_id', 'substitute_id', 'score', 'identified_substitute'].
    top_n : int, optional
        If specified, keeps only the top N substitutes per product by score.
        
    Returns:
    --------
    dtr_df : pd.DataFrame
        DataFrame with columns ['product_id', 'substitute_id', 'score', 'transferability_pct']
    """
    
    # Filter identified substitutes and remove duplicates
    filtered_df = subs_df[subs_df["identified_substitute"] == True].drop_duplicates(subset=['product_id', 'substitute_id'])
    
    results = []
    
    for prod, group in filtered_df.groupby('product_id'):
        g = group.copy()
        
        # Keep only top N
        g = g.sort_values('score', ascending=False).head(top_n)
        
        # Max score in top-N
        max_score = g['score'].max()
        total_score = g['score'].sum()
        
        # Avoid division by zero
        if total_score > 0:
            g['transferability_pct'] = g['score'] / total_score * max_score
        else:
            g['transferability_pct'] = 0.0
        
        results.append(g[['product_id', 'substitute_id', 'score', 'transferability_pct']])
    
    dtr_df = pd.concat(results, ignore_index=True)
    
    return dtr_df


def build_product_orders(orders_df):
    """Return dict: product_id -> set(order_id)."""
    return orders_df.groupby("product_id")["order_id"].apply(set).to_dict()


def sample_products_for_validation(transfer_df, product_orders, n_products=100, method="random", seed=42):
    """
    Return a list of product_ids to validate.
    method: "random" or "freq_stratified" (sample across frequency quartiles)
    """
    random.seed(seed)
    np.random.seed(seed)
    candidate_products = transfer_df["product_id"].unique().tolist()
    if method == "random":
        return random.sample(candidate_products, min(n_products, len(candidate_products)))
    elif method == "freq_stratified":
        # stratify by product frequency (orders per product)
        freqs = {p: len(product_orders.get(p, [])) for p in candidate_products}
        df = pd.DataFrame({"product_id": list(freqs.keys()), "freq": list(freqs.values())})
        df["quantile"] = pd.qcut(df["freq"].rank(method="first"), q=4, labels=False, duplicates="drop")
        out = []
        per_quant = max(1, n_products // 4)
        for q in sorted(df["quantile"].unique()):
            pool = df[df["quantile"] == q]["product_id"].tolist()
            pick = random.sample(pool, min(per_quant, len(pool)))
            out.extend(pick)
        # if short, add random
        if len(out) < n_products:
            remaining = set(candidate_products) - set(out)
            add = random.sample(list(remaining), min(n_products - len(out), len(remaining)))
            out.extend(add)
        return out[:n_products]
    else:
        return random.sample(candidate_products, min(n_products, len(candidate_products)))


# Stability test for sample (random split)
def run_stability_test_on_sample(orders_df, sample_products, subs_df):
    """
    Random 50/50 split of orders, compute transferability only for sample_products,
    then compute rank correlation between splits for each product.
    """
    order_ids = orders_df["order_id"].unique()
    np.random.shuffle(order_ids)
    half = len(order_ids) // 2
    splits = [order_ids[:half], order_ids[half:]]
    # precompute product_orders_full once
    correlations = []
    for i, ids in enumerate(splits):
        subset_orders = orders_df[orders_df["order_id"].isin(ids)]
        po = build_product_orders(subset_orders)
        # compute transferability but only for sample products
        dtr = compute_transferability(subs_df, 5)
        dtr["split"] = i
        if i == 0:
            df_all = dtr
        else:
            df_all = pd.concat([df_all, dtr], ignore_index=True)
    for A in sample_products:
        g = df_all[df_all["product_id"] == A]
        g0 = g[g["split"] == 0].set_index("substitute_id")["transferability_pct"]
        g1 = g[g["split"] == 1].set_index("substitute_id")["transferability_pct"]
        common = g0.index.intersection(g1.index)
        if len(common) < 2:
            continue
        corr, _ = spearmanr(g0.loc[common], g1.loc[common])
        correlations.append(corr)
    if len(correlations) == 0:
        return np.nan
    return float(np.nanmean(correlations)), float(np.nanstd(correlations))


# Effectiveness: stochastic simulated delisting for sample
def run_effectiveness_on_sample(product_orders, transfer_df_sample, sample_products, p_switch=0.3, K=3, n_trials=10, seed=42):
    """
    For each product A in sample_products:
      - predicted_top: top-K by transferability_pct
      - simulate n_trials: pick a random subset of A's orders that are 'open to switching'
        (size = p_switch * len(orders_A)), and reassign those orders to substitutes
        according to transferability_pct (multinomial).
      - compute Precision@K averaged across trials.
    Returns DataFrame with precision@K per product and aggregate stats.
    """
    random.seed(seed); np.random.seed(seed)
    results = []
    # pre-index transfer_df_sample by product
    grouped = transfer_df_sample.groupby("product_id")
    for A in sample_products:
        if A not in product_orders or A not in grouped.groups:
            continue
        orders_A = list(product_orders[A])
        n_A = len(orders_A)
        if n_A == 0:
            continue
        group = grouped.get_group(A)
        # predicted top-K
        predicted_top = group.sort_values("transferability_pct", ascending=False).head(K)["substitute_id"].tolist()
        if len(predicted_top) == 0:
            continue
        precisions = []
        # prepare substitute ids and probabilities
        subs = group["substitute_id"].tolist()
        probs = group["transferability_pct"].values
        if probs.sum() == 0:
            # skip if zero probs
            continue
        probs = probs / probs.sum()
        for t in range(n_trials):
            # choose orders open to switching
            m = max(1, int(round(p_switch * n_A)))
            # select orders (by id)
            open_orders = random.sample(orders_A, min(m, len(orders_A)))
            # simulate assignment of each open order to exactly one substitute (multinomial draws)
            assigned = np.random.choice(subs, size=len(open_orders), p=probs)
            # count increases per substitute
            inc_counts = pd.Series(assigned).value_counts()
            observed_top = inc_counts.sort_values(ascending=False).head(K).index.tolist()
            # compute precision@K
            inter = len(set(predicted_top).intersection(set(observed_top)))
            precisions.append(inter / K)
        results.append({"product_id": A, "predicted_top": predicted_top, "precision_at_k_mean": np.mean(precisions),
                        "precision_at_k_std": np.std(precisions)})
    return pd.DataFrame(results)


# Efficiency measurement: measure runtime of computing transferability for sample products
def measure_efficiency_sample(orders_df, subs_df, sample_products, product_orders):
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / (1024**2)
    t0 = time.time()
    dtr = compute_transferability(subs_df, 5)
    t1 = time.time()
    mem_after = proc.memory_info().rss / (1024**2)
    return {"runtime_s": t1 - t0, "mem_before_mb": mem_before, "mem_after_mb": mem_after, "mem_growth_mb": mem_after - mem_before, "n_rows": dtr.shape[0]}


# Master runner: run validation across multiple random subsets
def run_multiple_subsets_validation(orders_df, subs_df, transfer_df, n_subsets=5, sample_size=100, method="random",
                                    p_switch=0.3, K=3, n_trials=10, seed=42):
    """
    Runs validation on n_subsets different sampled sets of products and returns aggregated results.
    """
    product_orders = build_product_orders(orders_df)
    all_results = []
    for s in range(n_subsets):
        ss = seed + s
        sample = sample_products_for_validation(transfer_df, product_orders, n_products=sample_size, method=method, seed=ss)
        # compute transferability for only these products (fast)
        transfer_sample = compute_transferability(subs_df, 5)

        # stability
        stab_mean, stab_std = run_stability_test_on_sample(orders_df, sample, subs_df)
        # effectiveness
        eff_df = run_effectiveness_on_sample(product_orders, transfer_sample, sample, p_switch=p_switch, K=K, n_trials=n_trials, seed=ss)
        eff_mean = float(eff_df["precision_at_k_mean"].mean()) if not eff_df.empty else np.nan
        eff_std = float(eff_df["precision_at_k_mean"].std()) if not eff_df.empty else np.nan
        # efficiency
        eff_metrics = measure_efficiency_sample(orders_df, subs_df, sample, product_orders)
        all_results.append({
            "subset_index": s,
            "n_sample_products": len(sample),
            "stability_mean_corr": stab_mean,
            "stability_std_corr": stab_std,
            "effectiveness_precision_mean": eff_mean,
            "effectiveness_precision_std": eff_std,
            "eff_runtime_s": eff_metrics["runtime_s"],
            "eff_mem_growth_mb": eff_metrics["mem_growth_mb"]
        })
    return pd.DataFrame(all_results)