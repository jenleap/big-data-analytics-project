# Predicting Product Substitutes and Complements Using Basket-Level Data

## Context

Retailers regularly review their product assortment to determine which products to carry and which to delist, in order to maximize profit. When removing a product, its transferability — the percentage of its sales that will potentially be captured by other similar products — is a key consideration. Delisting products with high transferability pose less risk of revenue loss. In addition, a retailer should consider a product’s complements — products frequently bought together. This is relevant because listing or delisting a product may also affect the sales of its complements.

The research questions build on one another and leverage different methodological approaches for a comprehensive analysis:

Which products are most likely to act as substitutes for a focus product?
This question aims to identify high-quality substitutes by quantifying similarity in purchase behavior and product characteristics.
What proportion of a removed product’s sales could be transferred to each substituting product?
This examines the distribution of demand transferability (DTR), allowing more precise estimates of revenue redistribution following a delisting.
Which products exhibit strong complementarity with a focus product?
This investigates directional behavioral linkages between products and identifies complements whose sales are influenced by the focal item.
How much impact does a focus product have on its complementary products?
This estimates direct and network-amplified complement impact, enabling more robust category planning.

## Results
Interaticve results are available here in the Product Explorer: https://jenleap.github.io/big-data-analytics-project/index.html

## Dataset

This project uses the Instacart Online Grocery Shopping Dataset, publicly available and hosted on Kaggle: https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset

Full EDA is available here: https://jenleap.github.io/big-data-analytics-project/eda_report_orders.html

## Repository Structure
```
├── exploratory/ # exploratory analysis scripts
├── graphs/ # generated charts
├── json/ # product results JSON to be used in the Product Explorer
├── notebooks/ # Jupyter notebooks for EDA, experimentation, modeling
├── tests/
├── utils/ # utility scripts and helper functions
├── .gitignore # files ignored by git
├── README.md 
├── eda_report_orders.html # EDA report (rendered)
└── index.html # interative Product Explorer
```
## Environment
- Python version: 3.10+
- Install dependencies with: pip install -r requirements.txt

## Getting Started 
To reproduce the analysis, follow these steps:

### Clone the repository  
```bash
git clone https://github.com/jenleap/big-data-analytics-project.git  
cd big-data-analytics-project  
```

### Load the data
Download the full dataset from Kaggle: https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset
Place all raw files in the `/dataset` directory.

### Create directories for generated results
The notebooks are set to save results to `data/cleaned` and `data/validation`. If another directory structure is desired, 
update the filepaths for saving to CSV.

### Run notebooks
Open the relevant notebooks under `/notebooks`. Run them sequentially to reproduce the exploratory analyses, modeling, and result generation.

### Notes
To parse the results into JSON format to use in the Product Explorer, the `build_product_graph` function is available in `/utils/json-helpers.ipynb`.

#### Units Tests

To run the units tests, in the command line run:
`python -m unittest discover -s tests -p "*.py"`


.

