# Home-Decor Plus – Personalized Product Recommendation System

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Description](#data-description)
3. [Installation & Setup](#installation--setup)
4. [Usage](#usage)
5. [Methodology](#methodology)
   - [Feature Engineering (RFCM)](#feature-engineering-rfcm)
   - [Cold-Start Mitigation](#cold-start-mitigation)
   - [Modeling with SVD](#modeling-with-svd)
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Model Validation](#model-validation)
6. [Results & Business Impact](#results--business-impact)
7. [File Structure](#file-structure)
8. [Future Work](#future-work)
9. [Team Members](#team-members)
10. [Acknowledgements](#acknowledgements)

---

## Project Overview
The goal of this project is to develop a customer-focused recommendation system for Home-Decor Plus that delivers highly relevant product suggestions based on historical purchase behavior. By combining Recency-Frequency-Continuity-Monetary (RFCM) feature engineering with matrix factorization (Singular Value Decomposition), we:
- Addressed the cold-start problem for both new users and new products.
- Provided personalized top-N product recommendations to each customer.
- Generated actionable insights about customer segments, product popularity, and temporal trends.

This system is designed to improve engagement, increase upsell/cross-sell opportunities, and ultimately drive incremental revenue for Home-Decor Plus’s e-commerce platform.

---

## Data Description
- **Source**: Publicly available e-commerce transactional dataset (Kaggle).
- **Time Frame**: December 1, 2010 to December 9, 2011.
- **Number of Users**: 4,339 unique customer IDs.
- **Number of Products**: 3,877 unique items across multiple categories.
- **Key Fields**:
  - `CustomerID`: Unique identifier for each customer.
  - `InvoiceNo` & `InvoiceDate`: Transaction IDs and timestamps.
  - `StockCode` & `Description`: Product identifiers and descriptive names.
  - `Quantity` & `UnitPrice`: Quantity purchased and unit price.
  - Derived features (after preprocessing): Recency (days since last purchase), Frequency (total transactions), Continuity (distinct months active), Monetary (total spend), AvgBasketSize, PriceRange, etc.

---

## Installation & Setup
1. **Clone this repository**
   ```bash
   git clone https://github.com/<your-org>/home-decor-plus-recommendation.git
   cd home-decor-plus-recommendation

Create a Python virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate


Install required packages
pip install -r requirements.txt
requirements.txt should include (but is not limited to):
pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
surprise (or another SVD implementation library)
jupyter

Data Location
Place the raw transaction CSV (e.g., ecommerce_data.csv) under data/.

Usage
Data Preprocessing & EDA

Open notebooks/01_data_preprocessing_and_eda.ipynb.

Run each cell sequentially to clean the dataset, engineer RFCM features, and explore key patterns (product popularity, customer clusters, temporal trends).

Model Training & Prediction

Open notebooks/02_modeling_and_recommendation.ipynb.

This notebook constructs the ratings matrix (using RFCM scores), applies Singular Value Decomposition (SVD), and outputs top-N product recommendations for each user.

Model Validation

Inside the same notebook (or notebooks/03_validation.ipynb), you can reproduce cross-validation steps:

Remove a known rating (e.g., a 5-star rating) from the training set.

Check how often that held-out product appears in the top-5 and top-10 recommendations.

Compute RMSE & MAE over k-fold splits to verify stability.

Output

Generated recommendation CSV (outputs/top_n_recommendations.csv) lists the top 10 recommended products per customer.

Validation plots (e.g., error distributions, CV‐fold metrics) are saved under outputs/validation_plots/.


Methodology
Feature Engineering (RFCM)
We assign each (Customer × Product) pair a pseudo-rating (1–5) based on four dimensions:

Recency – Days since customer’s last purchase (binned into 1–5).

Frequency – Total number of transactions (binned into 1–5).

Continuity – Count of distinct purchase months (binned into 1–5).

Monetary – Aggregate spend (binned into 1–5).

These four scores are combined (e.g., via weighted average or binning) to form a single rating that populates our user-item rating matrix.

Cold-Start Mitigation
User Cold-Start (fewer than 3 historical purchases):

Assign them to a “Top N Popular Items” baseline (e.g., top 10 by overall sales).

If demographic or category data is available, match with similar users.

Product Cold-Start (very few sales):

Recommend these items to users who exhibit high affinity for niche or similar product categories (e.g., via content-based filtering on product attributes).

Our feature engineering (e.g., AvgBasketSize, PriceRange, DistinctInvoices, DistinctMonths) further alleviates cold-start gaps by capturing indirect signals.

Modeling with SVD
Construct Rating Matrix

Rows = CustomerID, Columns = ProductID.

Entries = RFCM-derived rating (1 to 5) wherever a purchase occurred; else zero/unobserved.

Apply Singular Value Decomposition

Use a library (e.g., surprise.SVD) to factorize the sparse matrix into user and item latent factors.

Tune hyperparameters (number of factors, learning rate, regularization) via cross‐validation.

Generate Predictions

For each user, predict ratings for all unseen products.

Rank predictions and extract top 10 highest-scoring items.

Exploratory Data Analysis (EDA)
Product Purchase Distribution

Top 10 products account for ~67.83% of all purchases.

“Regency Cake Stand 3 Tier” and “White Hanging Heart T-Light Holder” emerge as #1 and #2.

Customer Segmentation (Clustering)

Cluster labels (K = 4) identify:

Frequent, low spenders (Cluster 0)

VIP high spenders, low frequency (Cluster 1)

Moderate spenders, steady frequency (Cluster 2)

High-frequency, moderate spenders (Cluster 3)

Temporal Trends

Steady growth with volatility (promotion‐driven spikes).

Seasonal patterns (e.g., holiday peaks).

Price vs. Popularity

Weak negative correlation (ρ = –0.045), indicating price alone does not drive demand.

Charts and detailed analyses are available in the EDA notebook.

Model Validation
5-Fold Cross-Validation on known ratings yields:

RMSE ≈ 0.5458 ± 0.0022

MAE ≈ 0.43495 ± 0.0018

Fit time: ~3.86 ± 1.43 seconds / fold

Test time: ~0.54 ± 0.23 seconds / fold

Hold-Out “5-Star” Test

Remove all 5-star ratings from a random subset of users.

Check penetration: % of held-out items appearing in Top 5 or Top 10 recommendations.

High coverage (> 70% in Top 10) demonstrates that SVD generalizes well.

Validation scripts and plots are stored under notebooks/03_validation.ipynb and outputs/validation_plots/.

Results & Business Impact
Improved Product Discovery

Customers receive personalized suggestions beyond the Top 10 bestsellers.

Encourages exploration of related or niche items, potentially increasing basket size.

Enhanced Personalization

High-value customers (Clusters 1 & 2) see premium or high-ticket recommendations.

Frequent buyers (Cluster 0) get loyalty-focused offers.

Cold-Start Users & Products

Even brand-new customers receive relevant “Popular” recommendations.

Newly launched products are strategically recommended to customers with matching profile attributes.

Scalability & Efficiency

Real-time update capability—batch or incremental SVD retraining supports large transaction volumes with minimal latency.

Quantitative Gains (simulated/estimated)

~8–12% uplift in click-through rate (CTR) on recommended items.

~5% increase in average order value (AOV) among target segments.

(Actual A/B testing on Home-Decor Plus platform is recommended to measure live ROI.)



File Structure:
home-decor-plus-recommendation/
│
├── data/
│   ├── raw/
│   │   └── ecommerce_data.csv               # Original Kaggle transactions
│   └── processed/
│       ├── transactions_cleaned.csv         # After null removal & negative quantity filtering
│       └── feature_matrix.pkl               # Precomputed RFCM rating matrix (sparse)
│
├── notebooks/
│   ├── 01_data_preprocessing_and_eda.ipynb  # Data cleaning, feature engineering, clustering, EDA
│   ├── 02_modeling_and_recommendation.ipynb # Construct ratings, train SVD, generate Top-N recommendations
│   ├── 03_validation.ipynb                  # Cross-validation, hold-out testing, error metrics
│   └── 04_visualizations.ipynb              # Additional charts (optional)
│
├── outputs/
│   ├── top_n_recommendations.csv            # Top 10 recommendations/per user
│   ├── validation_plots/                    # RMSE/MAE across folds, penetration curves
│   └── logs/
│       └── model_training.log               # Training hyperparameters & timing
│
├── src/
│   ├── data_preprocessing.py                # Python scripts for cleaning and RFCM feature generation
│   ├── model_utils.py                       # SVD training, prediction, and evaluation functions
│   └── utils.py                             # Miscellaneous helper functions
│
├── requirements.txt                         # List of Python dependencies
├── README.md                                # This file
└── LICENSE                                  # Project license (e.g., MIT)


Future Work
Advanced Feature Engineering

Incorporate time-decay weighting (give greater importance to recent purchases).

Build a Product Affinity Score: co-purchase networks to capture frequently bought-together items.

Engineer contextual features (e.g., day of week, holiday indicator, customer demographics).

Hybrid Modeling

Combine Collaborative Filtering (CF) with Content-Based Filtering (CBF) for a hybrid recommender—this can further improve cold-start coverage.

Experiment with Bayesian Personalized Ranking (BPR) or Factorization Machines to directly optimize ranking metrics.

Deep Learning Approaches

Implement an Autoencoder-based recommender (e.g., Neural Collaborative Filtering) for richer latent representations.

Leverage Graph Neural Networks (GNNs) on a user-product bipartite graph to better capture relational structure.

A/B Testing & Online Evaluation

Deploy a pilot in production to measure CTR, conversion rate, and lift in average order value compared to baseline.

Set up an offline evaluation pipeline that simulates user sessions and compares multiple recommender variants.

Scalability & Real-Time Serving

Transition from batch SVD to incremental or streaming matrix factorization methods.

Integrate the recommendation API into Home-Decor Plus’s web stack (e.g., via Flask/FastAPI) for low-latency predictions.


Acknowledgements:
The original e-commerce dataset was obtained from Kaggle.

Inspired by standard recommender-system tutorials and the SurPRISE library for collaborative filtering.

UC Davis MSBA program for guidance on best practices in model validation and business impact analysis.
