# Predicting Product Substitutes and Complements Using Basket-Level Data

This project will use the Instacart Online Grocery Shopping Dataset, publicly available and hosted on Kaggle: https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset

Retailers regularly review their product assortment to determine which products to carry and which to delist, in order to maximize profit. When removing a product, its transferability — the percentage of its sales that will potentially be captured by other similar products — is a key consideration. Delisting products with high transferability pose less risk of revenue loss. In addition, a retailer should consider a product’s complements — products frequently bought together. This is relevant because listing or delisting a product may also affect the sales of its complements.

Existing research often uses choice models to estimate diversion ratios (or demand shift), requiring price, promotion and stockout data, which are often proprietary. Less explored in the literature is the use of basket-level data to predict both substitution and complementarity, offering a more accessible and behaviour-based approach. This enables retailers to estimate how delisting or listing a product may affect both substitutes and complements without relying on proprietary pricing data.

### Research Questions

Which products are most likely to absorb sales if a particular product is removed?

What percentage of sales from a removed product will be transferred to each substituting product?

Which products are complementary to a particular product and thus may be affected if it is removed?

What is the impact of a product on the sales of complementary items?

### Project Approach

<table>
  <tr>
    <th>Objective</th>
    <th>Methodology</th>
    <th>Reference</th>
  </tr>
  <tr>
    <td>Identify substitute products for a given product</td>
    <td>
        <ul>
            <li>Create a product-order matrix, where rows are orders and columns are products, values indicate whether a product was purchased (binary: 0 or 1)</li>
            <li>Calculate probabilities that product i and j appear together in same basket</li>
            <li>Calculate jaccard similarity and conditional purchase probability to identify products that appear in similar shopping contexts</li>
            <li>Calculate exclusivity-based substitution index (adapted from substitution and transfer models (Mani et al., 2016; Gayle & Lin, 2022))</li>
            <li>Normalize exclusivity, jaccard and conditional probability and combine into a hybrid score</li>
            <li>Rank candidate substitutes</li>
        </ul>
        Validation
        <ul>
            <li>Category consistency validation - substitute pairs should have high category overlap</li>
            <li>Behavioural validation - identify customers who used to buy A but stopped; check if these users later start buying B</li>
        </ul>
    </td>
    <td>
        <ul>
            <li>Sokol & Holý (2022): Demonstrate that basket-level similarity and co-occurrence patterns can infer substitute relationships without price data</li>
        </ul>
    </td>
  </tr>
  <tr>
    <td>Calculate what percentage of sales could be absorbed by substitute products if product A is delisted</td>
    <td>Another cell</td>
    <td>Another cell</td>
  </tr>
  <tr>
    <td>Identify complementary products</td>
    <td>Another cell</td>
    <td>Another cell</td>
  </tr>
  <tr>
    <td>Quantify how delisting a product may affect sales of its complements</td>
    <td>Another cell</td>
    <td>Another cell</td>
  </tr>
</table>


.

