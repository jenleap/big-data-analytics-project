import unittest
import pandas as pd
from utils.pairwise import compute_pairwise_probabilities_chunked 

class TestComputePairwiseProbabilitiesChunked(unittest.TestCase):

    def setUp(self):
        # Small test dataset
        self.test_data = pd.DataFrame({
            'order_id': [1, 1, 1, 2, 2, 3, 3, 3],
            'product_id': ['A', 'B', 'C', 'A', 'C', 'A', 'B', 'C']
        })

    def test_all_pairs_captured_one_direction(self):
        """
        Test that all non-zero co-occurring pairs are captured (one-directional)
        """
        co_df = compute_pairwise_probabilities_chunked(self.test_data)

        # Get all non-zero co-occurring pairs (one-directional)
        baskets = self.test_data.groupby('order_id')['product_id'].apply(list)
        expected_pairs = set()
        for basket in baskets:
            for i in range(len(basket)):
                for j in range(i + 1, len(basket)):
                    expected_pairs.add((basket[i], basket[j]))

        # Actual pairs from co_df
        actual_pairs = set(zip(co_df['product_i'], co_df['product_j']))

        missing = expected_pairs - actual_pairs
        self.assertTrue(len(missing) == 0, f"Missing non-zero co-occurring pairs: {missing}")

    def test_probabilities_correct_one_direction(self):
        """
        Test that P_i, P_j, P_ij are correctly calculated (allow one-directional)
        """
        co_df = compute_pairwise_probabilities_chunked(self.test_data)

        # Manually compute expected probabilities
        baskets = self.test_data.groupby('order_id')['product_id'].apply(list)
        num_orders = len(baskets)
        product_counts = self.test_data['product_id'].value_counts().to_dict()

        # Check each pair in co_df
        for _, row in co_df.iterrows():
            prod_i = row['product_i']
            prod_j = row['product_j']

            # Count co-occurrences
            count_ij = sum(1 for basket in baskets if prod_i in basket and prod_j in basket)
            expected_P_i = product_counts[prod_i] / num_orders
            expected_P_j = product_counts[prod_j] / num_orders
            expected_P_ij = count_ij / num_orders

            self.assertAlmostEqual(row['P_i'], expected_P_i, places=6)
            self.assertAlmostEqual(row['P_j'], expected_P_j, places=6)
            self.assertAlmostEqual(row['P_ij'], expected_P_ij, places=6)

if __name__ == "__main__":
    unittest.main()
