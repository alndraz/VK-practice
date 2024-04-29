import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import ndcg_score

# Load data
file_path = './intern_task.csv'
data = pd.read_csv(file_path)

# Filter out sessions with only one object
query_ids = data['query_id']
query_counts = query_ids.value_counts()
query_ids_to_keep = query_counts[query_counts > 1].index
filtered_data = data[data['query_id'].isin(query_ids_to_keep)]

# Separate data into features and targets
X_filtered = filtered_data.drop(['rank', 'query_id'], axis=1)
y_filtered = filtered_data['rank']
query_ids_filtered = filtered_data['query_id']

# Split data into training, validation, and test sets with stratification
X_train_val, X_test, y_train_val, y_test, queries_train_val, queries_test = train_test_split(
    X_filtered, y_filtered, query_ids_filtered, test_size=0.2, random_state=42, stratify=query_ids_filtered)

X_train, X_val, y_train, y_val, queries_train, queries_val = train_test_split(
    X_train_val, y_train_val, queries_train_val, test_size=0.25, random_state=42, stratify=queries_train_val)  # 0.25 x 0.8 = 0.2

# Prepare data for LightGBM
train_group = queries_train.value_counts(sort=False).sort_index().values
valid_group = queries_val.value_counts(sort=False).sort_index().values

train_data = lgb.Dataset(data=X_train, label=y_train, group=train_group)
valid_data = lgb.Dataset(data=X_val, label=y_val, group=valid_group)

# Model parameters
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [5],
    'learning_rate': 0.1,
    'num_leaves': 31,
    'verbose': -1
}

# Train model
model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[valid_data], valid_names=['eval'])

# Predict on the test set
y_test_pred = model.predict(X_test)

# Calculate NDCG@5 for the test data
ndcg_test = ndcg_score([y_test], [y_test_pred], k=5)

print(f"NDCG@5 Score on Test Set: {ndcg_test}")
