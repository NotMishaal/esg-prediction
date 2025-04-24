# A Globally Representative Data Framework for ESG Score Prediction
This repository contains the full pipeline for ESG (Environmental, Social, and Governance) score prediction using a combination of structured financial data and unstructured news content. The project integrates data collection, preprocessing, exploration, feature engineering, sentiment analysis, model training, and evaluation for robust ESG modeling.


## Folders
- data: All datasets used in the project
    - raw: Unprocessed/raw data
    - processed: Cleaned and preprocessed data
- notebooks: Jupyter notebooks used for data collection
- src: Source code for scripts and reusable modules
- models: Saved models, weights and checkpoints
- scripts: Standalone scripts (e.g., data scraping, automation)
- experiments: Notebooks used to run the experiments


## Project Flow
### 1) Company Screening

Run `notebooks/company_screener.ipynb` to obtain the companies you'll need for the dataset. Methods to screen different companies written in `scripts/screener.py`.

- Regional groupings defined using the world bank classification.
- Companies were bucketed by relative market capitalization.
- Only companies with available ESG scores from Sustainalytics were used.

### 2) Data Collection

#### 2.1) Financial and ESG Data

Run `notebooks/data_extraction.ipynb` to obtain the annual statements for the companies obtained earlier, through the Yahoo Finance API.

Methods to fetch the different income statement, balance sheet, cash flow and ESG data written in `scripts/data_scraper.py`.

#### 2.2) News Articles

Run `notebooks/article_extraction.ipynb` to fetch up to 250 ESG-related articles per company via the GDELT API. This only contains the URLs and other metadata for the articles, not the actual text content.

`notebooks/news.ipynb` filters and parses the articles and gives the text content, using the newspaper3k library.

### 3) Preprocessing

#### 3.1) News Data

Initial cleaning of the news content is performed in `notebooks/news.ipynb`. Next the relevance of each article's content with ESG is evaluated and filtered in `notebooks/relevance.ipynb` using ESGBERT pre-trained models.

After sentiment analysis in conducted in `notebooks/sentiment.ipynb` using FinBERT, and four aggregate sentiment features are computed for each company.

#### 3.2) Financial Data

The financial data preprocessing and final dataset creation takes places in `experiments/data_preprocessing.ipynb`. Here the following preprocessing steps take place:
- Missing Data Imputation:

    - Stage 1: Forward/backward fill using company history

    - Stage 2: Median of peer group (region + size)

    - Stage 3: Global median

- Feature Engineering: 14 key financial ratios created (e.g., ROA, ROE, Debt-to-Equity).

- Sentiment Merging: Sentiment features computed from the news data are merged with the financial data.

- Dimensionality Reduction: Hierarchical clustering on correlation matrix to reduce redundancy.

- Scaling: Log1p transformation for skewed features + standard scaling.

- Encoding: One-hot encoding for categorical features.

The final dataset with the filtered companies and features is obtained from here.

### 4) Experimental Setup
Using the finalized dataset, different variants and partitions necessary for the research methodology are created in `experiments/experimental_setup.ipynb`. This includes:

- Baseline Dataset

    - Skewed towards Large/Mid-Cap firms from North America and Europe.

    - Reflects common practice in ESG research.

- Diversified Datasets

    - Med-Div: Balanced to the median size of strata.

    - Max-Div: All strata up/down-sampled to match the smallest group.

- Partitioning Schemes

    - Within-Sample Stratified Split: 80/20 in-distribution evaluation.

    - Cross-Region Hold-Out: Train on all but one region, test on the hold-out.

    - Cross-Size Hold-Out: Same as above but with company size categories.

A nested dictionary of dictionaries containing the different datasets is obtained from here.

### 5) Model Training and Evaluation
The ML models and experiments actually take place in `experiments/training.ipynb`. For specific results that were needed for the thesis, `experiments/result_analysis.ipynb` was used to run the required analysis.
