import pandas as pd


def categorize_by_distribution(col):
    """
    Categorize a numeric Series into quartile/IQR/outlier categories.

    Returns a Series with the same index as input.
    """

    # Calculate quartiles and IQR
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1

    # Define fences for outliers
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    # Function to categorize
    def categorize(x):
        if x < lower_fence or x > upper_fence:
            return "4" #Outliers
        elif x < Q1:
            return "2" #Q2
        elif x > Q3:
            return "3" #Q3
        else:
            return "1" #IQR

    categorized = col.apply(categorize).astype('category')
    categorized.index = col.index  # Ensure alignment
    return categorized


