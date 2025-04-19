import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, RobustScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor

# Utility function for safe division

def safe_div(numer, denom):
    """Divide two pandas Series, returning NaN for invalid values or zero denominators."""
    return numer.div(denom).replace([np.inf, -np.inf], np.nan)

# Financial ratios definitions from the notebook
financial_ratios = {
    'profit_margin': (['Net Income'], 'Operating Revenue'),
    'ROA': (['Net Income'], 'Total Assets'),
    'ROE': (['Net Income Common Stockholders'], 'Common Stock Equity'),
    'debt_to_equity': (['Total Debt'], 'Total Equity Gross Minority Interest'),
    'interest_coverage': (['EBIT'], 'Interest Expense'),
    'current_ratio': (['Current Assets'], 'Current Liabilities'),
    'quick_ratio': (['Current Assets', 'Inventory'], 'Current Liabilities'),
    'sales_to_assets': (['Total Revenue'], 'Total Assets'),
    'EBIT_to_sales': (['EBIT'], 'Total Revenue'),
    'dividend_yield': (['Cash Dividends Paid'], 'marketCap'),
    'net_income_to_sales': (['Net Income'], 'Total Revenue'),
    'liquidity_ratio': (['Current Assets'], 'Current Liabilities'),
    'solvency_ratio': (['Total Debt'], 'Total Assets'),
    'price_to_earnings': (['marketCap', 'Ordinary Shares Number'], 'Diluted EPS'),
}

# Custom transformer: drop columns with > threshold missing
class MissingThresholdDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.cols_to_drop_ = None
    def fit(self, X, y=None):
        pct_null = X.isnull().mean()
        self.cols_to_drop_ = pct_null[pct_null > self.threshold].index.tolist()
        return self
    def transform(self, X):
        return X.drop(columns=self.cols_to_drop_)

# Custom transformer: remove low-variance features
class ConstantFeatureRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.selector_ = VarianceThreshold(threshold=threshold)
        self.keep_cols_ = None
    def fit(self, X, y=None):
        numeric = X.select_dtypes(include=[np.number]).fillna(0)
        self.selector_.fit(numeric)
        self.keep_cols_ = numeric.columns[self.selector_.get_support()].tolist()
        return self
    def transform(self, X):
        return X[self.keep_cols_]

# Custom transformer: group-based imputation
class GroupImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_cols=['symbol'], median_group_cols=['region','companySize','latest_year']):
        self.group_cols = group_cols
        self.median_group_cols = median_group_cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        # forward/backward fill by symbol
        df[numeric] = df.groupby(self.group_cols)[numeric].ffill().bfill()
        # group median imputation
        for col in numeric:
            df[col] = df.groupby(self.median_group_cols)[col].transform(lambda grp: grp.fillna(grp.median()))
        # global median
        for col in numeric:
            df[col] = df[col].fillna(df[col].median())
        return df

# Custom transformer: keep only latest entry per symbol
class LatestEntryFilter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        return df[df['date'] == df['latest_date']].drop(columns=['latest_date'])

# Custom transformer: compute financial ratios
class RatioCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, ratio_dict):
        self.ratio_dict = ratio_dict
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        for name, (num_cols, denom_col) in self.ratio_dict.items():
            cols = num_cols if isinstance(num_cols, list) else [num_cols]
            numerator = df[cols].sum(axis=1)
            df[name] = safe_div(numerator, df[denom_col])
        return df

# Custom transformer: select representative features via hierarchical clustering
class RepresentativeFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.selected_features_ = None
    def fit(self, X, y):
        df = X.select_dtypes(include=[np.number]).copy()
        # compute absolute correlation matrix
        corr = df.corr().abs()
        # distance matrix
        dist = 1 - corr
        # condensed distance
        condensed = squareform(dist.values, checks=False)
        Z = linkage(condensed, method='average')
        clusters = fcluster(Z, t=self.threshold, criterion='distance')
        # pick feature with highest abs corr with target
        abs_corr = df.apply(lambda col: col.corr(y).abs())
        reps = []
        for cluster_id in np.unique(clusters):
            features = df.columns[clusters == cluster_id]
            reps.append(abs_corr.loc[features].idxmax())
        self.selected_features_ = reps
        return self
    def transform(self, X):
        return X[self.selected_features_]

# Build the preprocessing and modeling pipeline
# Features to one-hot encode
categorical_features = ['region', 'companySize']
# Numeric features will be inferred at fit time

def build_pipeline():
    # Numeric transformer: select representative features, log+scale
    numeric_transformer = Pipeline([
        ('rep_sel', RepresentativeFeatureSelector(threshold=0.3)),
        ('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ('scale', RobustScaler())
    ])

    # Categorical transformer: one-hot encoding
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, slice(0, None)),  # all numeric columns
        ('cat', categorical_transformer, categorical_features)
    ], remainder='drop')

    pipeline = Pipeline([
        ('drop_missing', MissingThresholdDropper(0.7)),
        ('drop_constant', ConstantFeatureRemover(0.1)),
        ('impute', GroupImputer()),
        ('latest', LatestEntryFilter()),
        ('ratios', RatioCalculator(financial_ratios)),
        ('preproc', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    return pipeline

# Usage example:
# df = pd.read_csv('path_to_numerical.csv')\# y = df['totalEsg']
# pipeline = build_pipeline()
# pipeline.fit(df, y)
# preds = pipeline.predict(df_new)
