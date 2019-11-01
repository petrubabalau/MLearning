import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE

import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


# =============================================================================
# Constants
# =============================================================================
DATA_PATH = os.path.join(
	os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
	'data',
	'missing_values'
	)
TRAIN_CSV = os.path.join(DATA_PATH, 'train.csv')
TEST_CSV = os.path.join(DATA_PATH, 'test.csv')

DROP_COLS = ['Street', 'Utilities', 'Alley', 'LandSlope', 'RoofMatl',
			 'PoolQC', 'Fence', 'MiscFeature']

fields = ['Id', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 
          'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 
          'SalePrice', 'TotRmsAbvGrd',
          'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea']

# =============================================================================
# Helpers
# =============================================================================
def create_csv(X_test, preds_test, filename='submission.csv'):
	"""Writes results to a csv file.
	"""
	output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
	output.to_csv(filename, index=False)

def get_categorical_cols(df, nunique=10, dtype='object'):
	"""Returns a list of categorical columns.
	"""
	res = [cname for cname in df.columns if
	       df[cname].nunique() < nunique and
	       df[cname].dtype == dtype]
	return res

def get_numerical_cols(df):
	"""Returns a list of numerical columns.
	"""
	dtypes = ('int64', 'float64')
	res = [cname for cname in df.columns if
	       df[cname].dtype in dtypes]
	return res


# =============================================================================
# Model training
# =============================================================================
# Read the data
X_full = pd.read_csv(TRAIN_CSV, index_col='Id')
X_test_full = pd.read_csv(TEST_CSV, index_col='Id')

# X_full.drop(DROP_COLS, axis=1, inplace=True)
# X_test_full.drop(DROP_COLS, axis=1, inplace=True)


# plt.figure(figsize=(12,10))
# cor = X_full.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()
#Correlation with output variable
# cor_target = abs(cor["SalePrice"])
#Selecting highly correlated features
# relevant_features = cor_target[cor_target>0.4]
# print(relevant_features)


# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(
	X_full, y, train_size=0.8, test_size=0.2, random_state=0
)

# Select numerical columns
numerical_cols = get_numerical_cols(X_train)

# Select categorical columns with relatively low cardinality
categorical_cols = get_categorical_cols(X_train, nunique=26)

# -----------------------------------------------------------------------------
# Investigate cardinality
# object_nunique = list(map(lambda col: X_train[col].nunique(), categorical_cols))
# d = dict(zip(categorical_cols, object_nunique))

# Print number of unique entries by column, in ascending order
# for x in sorted(d.items(), key=lambda x: x[1]):
#    print(x)

X_train_num_cols = X_train[numerical_cols].copy()
X_valid_num_cols = X_valid[numerical_cols].copy()

num_imputer = SimpleImputer(strategy='constant')
imputed_X_train = pd.DataFrame(num_imputer.fit_transform(X_train_num_cols))
imputed_X_valid = pd.DataFrame(num_imputer.transform(X_valid_num_cols))

# Put column names back
imputed_X_train.columns = X_train_num_cols.columns
imputed_X_valid.columns = X_valid_num_cols.columns

# -----------------------------------------------------------------------------
# Apply one-hot encoder to each column with categorical data

X_train_categorical_cols = X_train[categorical_cols].copy()
X_valid_categorical_cols = X_valid[categorical_cols].copy()

categ_imputer = SimpleImputer(strategy='constant')
imputed_X_train_categorical_cols = pd.DataFrame(
	categ_imputer.fit_transform(X_train_categorical_cols)
)
imputed_X_valid_categorical_cols = pd.DataFrame(
	categ_imputer.transform(X_valid_categorical_cols)
)

imputed_X_train_categorical_cols.columns = X_train_categorical_cols.columns
imputed_X_valid_categorical_cols.columns = X_valid_categorical_cols.columns


OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(
	OH_encoder.fit_transform(imputed_X_train_categorical_cols)
)
OH_cols_valid = pd.DataFrame(
	OH_encoder.transform(imputed_X_valid_categorical_cols)
)

# One-hot encoding removed index; put it back
OH_cols_train.index = imputed_X_train_categorical_cols.index
OH_cols_valid.index = imputed_X_valid_categorical_cols.index

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([imputed_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([imputed_X_valid, OH_cols_valid], axis=1)


clf = XGBRegressor(n_estimators=1000, learning_rate=0.05)
clf.fit(OH_X_train, y_train)
preds = clf.predict(OH_X_valid)
print("\nMAE: {}\n".format(mean_absolute_error(preds, y_valid)))


'''
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()


clf = XGBRegressor(n_estimators=1000, learning_rate=0.05)
clf.fit(X_train, y_train)

# create_csv(X_test, preds_test)

preds = clf.predict(X_valid)
print("MAE:", mean_absolute_error(preds, y_valid))
'''
