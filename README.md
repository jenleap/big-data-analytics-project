# Predicting Product Substitutes and Complements Using Basket-Level Data

This project will use the Instacart Online Grocery Shopping Dataset, publicly available and hosted on Kaggle: https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset

Retailers regularly review their product assortment to determine which products to carry and which to delist, in order to maximize profit. When removing a product, its transferability — the percentage of its sales that will potentially be captured by other similar products — is a key consideration. Delisting products with high transferability pose less risk of revenue loss. In addition, a retailer should consider a product’s complements — products frequently bought together. This is relevant because listing or delisting a product may also affect the sales of its complements.

Existing research often uses choice models to estimate diversion ratios (or demand shift), requiring price, promotion and stockout data, which are often proprietary. Less explored in the literature is the use of basket-level data to predict both substitution and complementarity, offering a more accessible and behaviour-based approach. This enables retailers to estimate how delisting or listing a product may affect both substitutes and complements without relying on proprietary pricing data.

### Data Description

#### Order Size Distribution
This graph shows the distribution of basket sizes (number of products per order). Most orders contain 5–14 items, with a few unusually large baskets flagged as outliers, perhaps representing bulk or special orders.

<img src="graphs/order_size_frequency.png" width="200">

#### User Order Frequency
This graph displays the number of orders placed per user. Most users place between 5–19 orders, while a small subset of high-frequency users have been flagged as outliers to prevent skewing average-based analyses.

<img src="graphs/user_order_counts_hist.png" width="200">

#### Product Frequency
This graph illustrates how many baskets each product appears in. Most products are purchased infrequently, while a few popular products appear in a disproportionately large number of baskets. Both extremes have been flagged to aid interpretation and prevent skew in frequency-based metrics.

<img src="graphs/product_order_counts_hist.png" width="200">

### Research Questions

Which products are most likely to absorb sales if a particular product is removed?

What percentage of sales from a removed product will be transferred to each substituting product?

Which products are complementary to a particular product and thus may be affected if it is removed?

What is the impact of a product on the sales of complementary items?

### Project Approach

<img src="graphs/methodology-table.png">


.

