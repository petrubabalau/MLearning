import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from xgboost import XGBRegressor


DATA_PATH = os.path.join(
	os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
	'data',
	'missing_values'
	)
TRAIN_CSV = os.path.join(DATA_PATH, 'train.csv')
TEST_CSV = os.path.join(DATA_PATH, 'test.csv')

# Read the data
X_full = pd.read_csv(TRAIN_CSV, index_col='Id')
X_test_full = pd.read_csv(TEST_CSV, index_col='Id')


# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
	X_full, y, train_size=0.8, test_size=0.2, random_state=0
)


# Select categorical columns with relatively low cardinality
categorical_cols = [cname for cname in X_train_full.columns if
				    X_train_full[cname].nunique() < 30 and
				    X_train_full[cname].dtype == 'object']

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if
	              X_train_full[cname].dtype in ['int64', 'float64']]


my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# print(X_train.head())

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
	('imputer', SimpleImputer(strategy='constant')),
	('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
	transformers=[
		('num', numerical_transformer, numerical_cols),
		('cat', categorical_transformer, categorical_cols)
])

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)


clf = Pipeline(
	steps=[
		('preprocessor', preprocessor),
		('model', my_model)],
	verbose=True
)

# Preprocessing of training data, fit model

clf.fit(
	X_train, y_train
)


# Preprocessong of validation data, get predictions
preds_test = clf.predict(X_test)


output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
'''

preds = clf.predict(X_valid)
print("MAE:", mean_absolute_error(preds, y_valid))
'''