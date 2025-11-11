import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def compute_hybrid_substitution_score(product_df, pairwise_df, sampled_products, file_path, weight_si=0.5, weight_jc=0.3,
                                      weight_cond=0.2, weight_sim_partial=0.7, weight_sim_none=0.4):
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
        f.write("product_id,substitute_id,score,rank,jaccard,conditional,substitution_index,same_department,same_aisle,valid_substitution,possible_substitution\n")

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

         # Merge department and aisle info
        product_probs_df = product_probs_df.merge(
            product_df[['product_id', 'department_id', 'aisle_id']],
            left_on='product_i', right_on='product_id', how='left'
        ).rename(columns={'department_id': 'product_dept', 'aisle_id': 'product_aisle'}).drop(columns=['product_id'])

        product_probs_df = product_probs_df.merge(
            product_df[['product_id', 'department_id', 'aisle_id']],
            left_on='product_j', right_on='product_id', how='left'
        ).rename(columns={'department_id': 'substitute_dept', 'aisle_id': 'substitute_aisle'}).drop(columns=['product_id'])

        # Compute metrics
        product_probs_df['jaccard'] = product_probs_df.apply(lambda x: x.P_ij / (x.P_i + x.P_j - x.P_ij), axis=1)
        product_probs_df['conditional'] = product_probs_df.apply(lambda x: ((x.P_ij / x.P_i) + (x.P_ij / x.P_j)) / 2, axis=1)
        product_probs_df['substitution_index'] = product_probs_df.apply(lambda x: ((x.P_i * x.P_j) - x.P_ij) / (x.P_i * x.P_j), axis=1)

        # Determine similarity weight
        def get_similarity_weight(x):
            if x.product_dept == x.substitute_dept:
                if x.product_aisle == x.substitute_aisle:
                    return 1.0   # same dept + aisle → strongest
                return weight_sim_partial       # same dept, different aisle
            return weight_sim_none          # different dept

        product_probs_df['similarity_weight'] = product_probs_df.apply(get_similarity_weight, axis=1)

        # Normalize
        for col in ['jaccard', 'conditional', 'substitution_index']:
            min_val = product_probs_df[col].min()
            max_val = product_probs_df[col].max()
            if max_val - min_val != 0:
                product_probs_df[col] = (product_probs_df[col] - min_val) / (max_val - min_val)
            else:
                product_probs_df[col] = 0.0

        # Calculate weighted hybrid score
        product_probs_df['score'] = (
            (weight_si * product_probs_df['substitution_index'] -
             weight_jc * product_probs_df['jaccard'] -
             weight_cond * product_probs_df['conditional'])
            * product_probs_df['similarity_weight']
        )

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

        dept_map = product_df.set_index('product_id')['department_id'].to_dict()
        aisle_map = product_df.set_index('product_id')['aisle_id'].to_dict()

        # Add department and aisle info to subs_df
        top_substitutes_df['product_dept'] = top_substitutes_df['product_id'].map(dept_map)
        top_substitutes_df['substitute_dept'] = top_substitutes_df['substitute_id'].map(dept_map)
        top_substitutes_df['product_aisle'] = top_substitutes_df['product_id'].map(aisle_map)
        top_substitutes_df['substitute_aisle'] = top_substitutes_df['substitute_id'].map(aisle_map)

        # Check matches
        top_substitutes_df['same_department'] = top_substitutes_df['product_dept'] == top_substitutes_df['substitute_dept']
        top_substitutes_df['same_aisle'] = top_substitutes_df['product_aisle'] == top_substitutes_df['substitute_aisle']
        top_substitutes_df['valid_substitution'] = top_substitutes_df['same_aisle']
        top_substitutes_df['possible_substitution'] = top_substitutes_df['same_department']

        # Save results as CSV
        columns_to_save = [
            'product_id', 'substitute_id', 'score', 'rank',
            'jaccard', 'conditional', 'substitution_index',
            'same_department', 'same_aisle', 'valid_substitution', 'possible_substitution'
        ]
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

    adjusted_threshold = best_threshold + df[score_col].std() * 0.15
    
    print(f" Best threshold determined as: {best_threshold:.3f} and adjusted to: {adjusted_threshold}\n")

    return adjusted_threshold


def summarize_substitution_validation(top_substitutes_df, product_df, pairwise_df):
    """
    Summarize substitution validation metrics for easy comparison across weightings.
    
    Returns a single DataFrame with summary metrics.
    Only 'identified_substitute' products are considered for score and co-occurrence metrics.
    """

    subs_df = top_substitutes_df.copy()

    # --- Department / Aisle alignment ---
    valid_frac = subs_df[subs_df['identified_substitute']]['valid_substitution'].mean()
    aisle_frac = subs_df[subs_df['identified_substitute']]['possible_substitution'].mean()  # assuming possible_substitute = dept only

    # Fraction of valid/possible substitutes not identified
    fn_frac = subs_df[(subs_df['valid_substitution'] | subs_df['possible_substitution']) & 
                      (~subs_df['identified_substitute'])].shape[0] / \
              subs_df[(subs_df['valid_substitution'] | subs_df['possible_substitution'])].shape[0]

    # --- Average score and number of substitutes per product ---
    avg_score = subs_df[subs_df['identified_substitute']].groupby('product_id')['score'].mean().mean()
    avg_num_substitutes = subs_df[subs_df['identified_substitute']].groupby('product_id')['substitute_id'].count().mean()

    # --- Co-occurrence metrics ---
    pairwise_df = pairwise_df.copy()
    pairwise_df['joint_freq'] = pairwise_df['P_ij']
    pairwise_df['lift'] = pairwise_df['P_ij'] / (pairwise_df['P_i'] * pairwise_df['P_j'])

    coocc_df = subs_df[subs_df['identified_substitute']].merge(
        pairwise_df[['product_i','product_j','joint_freq','lift']],
        left_on=['product_id','substitute_id'],
        right_on=['product_i','product_j'],
        how='left'
    )
    coocc_corr = coocc_df[['score','joint_freq','lift']].corr()
    score_joint_corr = coocc_corr.loc['score','joint_freq']
    score_lift_corr = coocc_corr.loc['score','lift']

    # --- Feature correlations ---
    feature_cols = ['substitution_index','jaccard','conditional']
    feature_corr = subs_df[subs_df['identified_substitute']][feature_cols].corr()

    # --- Aggregate summary ---
    summary = {
        'avg_dept_frac': valid_frac,
        'avg_aisle_frac': aisle_frac,
        'avg_score': avg_score,
        'avg_num_substitutes': avg_num_substitutes,
        'false_negative_frac': fn_frac,
        'score_joint_freq_corr': score_joint_corr,
        'score_lift_corr': score_lift_corr,
        'feature_corr': feature_corr
    }

    print("***Substitution Validation Summary***\n")
    print(f"Average fraction of substitutes in same department: {summary['avg_dept_frac']:.3f}")
    print(f"Average fraction of substitutes in same aisle: {summary['avg_aisle_frac']:.3f}")
    print(f"Average substitution score: {summary['avg_score']:.3f}")
    print(f"Average number of substitutes per product: {summary['avg_num_substitutes']:.1f}")
    print(f"Fraction of valid/possible substitutes not identified: {summary['false_negative_frac']:.3f}\n")
    
    print("Co-occurrence correlations:")
    print(f" - Score vs joint frequency (P_ij): {summary['score_joint_freq_corr']:.3f}")
    print(f" - Score vs lift: {summary['score_lift_corr']:.3f}\n")
    
    print("Feature correlations:")
    print(summary['feature_corr'])

    # Return as a DataFrame for easier comparison across weightings
    summary_df = pd.DataFrame([summary])
    return summary_df


def get_substitutes_and_evaluate(product_df, pairwise_df, substitutes_df, file_path):
    
    # Find the best substitution score threshold which will identify a product as a true substitute
    best_threshold = find_best_threshold(substitutes_df)

    # Add a new column 'identified_substitute' based on the threshold
    substitutes_df['identified_substitute'] = substitutes_df['score'] >= best_threshold

    # Save results to CSV
    #substitutes_df.to_csv(file_path, index=False)

    evaluation_df = summarize_substitution_validation(substitutes_df, product_df, pairwise_df)
    evaluation_df.to_csv(file_path, index=False)