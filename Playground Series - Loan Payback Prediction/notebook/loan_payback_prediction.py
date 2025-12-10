import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load Data
# Assuming the script is run from the notebook directory
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

print(f'Train Data Shape: {train_df.shape}')
print(f'Test Data Shape: {test_df.shape}')

# Data Types and Null Values
print(train_df.info())
print(train_df.isnull().sum())
print(train_df.describe())

# Feature Separation
numerical_columns = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = train_df.select_dtypes(include=['object']).columns.tolist()

if 'id' in numerical_columns:
    numerical_columns.remove('id')
if 'loan_paid_back' in numerical_columns:
    numerical_columns.remove('loan_paid_back')

print(f"Numerical Columns ({len(numerical_columns)}): {numerical_columns}")
print(f"Categorical Columns ({len(categorical_columns)}): {categorical_columns}")

# Target Variable Analysis
plt.figure(figsize=(5, 4))
train_df['loan_paid_back'].value_counts().plot(kind='bar', color=['#e74c3c', '#2ecc71'])
plt.title('Loan Payback Distribution (Count)', fontsize=14, fontweight='bold')
plt.xlabel('Loan Paid Back (0=No, 1=Yes)')
plt.ylabel('Count')
plt.savefig('target_distribution.png')
# plt.show()

# --- Advanced EDA ---

# Numerical Features Distribution
print("Plotting Numerical Features Distribution...")
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns):
    plt.subplot(2, 3, i+1)
    sns.histplot(data=train_df, x=col, hue='loan_paid_back', kde=True, element="step", stat="density", common_norm=False)
    plt.title(f'{col} Distribution')
plt.tight_layout()
plt.savefig('numerical_distribution.png')
# plt.show() # Commented out for script execution

# Categorical Features Analysis
print("Plotting Categorical Features Analysis...")
plt.figure(figsize=(15, 15))
for i, col in enumerate(categorical_columns):
    plt.subplot(3, 2, i+1)
    sns.countplot(data=train_df, x=col, hue='loan_paid_back')
    plt.title(f'{col} vs Loan Paid Back')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('categorical_distribution.png')
# plt.show()

# Correlation Analysis
print("Plotting Correlation Matrix...")
plt.figure(figsize=(10, 8))
sns.heatmap(train_df[numerical_columns + ['loan_paid_back']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
# plt.show()

print("EDA Completed. Images saved.")

# --- Feature Engineering ---

print("Starting Feature Engineering...")

# 1. Preprocessing
# Drop 'id' if it exists (already handled in numerical_columns logic but good to be explicit for dataframe)
if 'id' in train_df.columns:
    train_df = train_df.drop('id', axis=1)
if 'id' in test_df.columns:
    test_id = test_df['id'] # Save for submission
    test_df = test_df.drop('id', axis=1)

# 2. Encoding
# Identify low and high cardinality categorical features
low_cardinality_cols = [col for col in categorical_columns if train_df[col].nunique() < 10]
high_cardinality_cols = [col for col in categorical_columns if train_df[col].nunique() >= 10]

print(f"Low Cardinality Columns: {low_cardinality_cols}")
print(f"High Cardinality Columns: {high_cardinality_cols}")

# One-Hot Encoding for low cardinality
train_df = pd.get_dummies(train_df, columns=low_cardinality_cols, drop_first=True)
test_df = pd.get_dummies(test_df, columns=low_cardinality_cols, drop_first=True)

# Align columns (in case test set has different categories, though unlikely for these standard cols)
train_df, test_df = train_df.align(test_df, join='left', axis=1)
# Fill NaN in test_df (from alignment) with 0
test_df = test_df.fillna(0)
# Drop target from test_df if it got added by align (it shouldn't but safety first)
if 'loan_paid_back' in test_df.columns:
    test_df = test_df.drop('loan_paid_back', axis=1)

# Label Encoding for high cardinality (e.g., grade_subgrade)
from sklearn.preprocessing import LabelEncoder
for col in high_cardinality_cols:
    le = LabelEncoder()
    # Fit on both train and test to cover all categories
    le.fit(pd.concat([train_df[col], test_df[col]]).astype(str))
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

# 3. New Features
# Interaction terms
train_df['loan_to_income'] = train_df['loan_amount'] / train_df['annual_income']
test_df['loan_to_income'] = test_df['loan_amount'] / test_df['annual_income']

# Advanced Features
train_df['monthly_debt'] = train_df['annual_income'] * train_df['debt_to_income_ratio'] / 12
test_df['monthly_debt'] = test_df['annual_income'] * test_df['debt_to_income_ratio'] / 12

train_df['interest_amount'] = train_df['loan_amount'] * (train_df['interest_rate'] / 100)
test_df['interest_amount'] = test_df['loan_amount'] * (test_df['interest_rate'] / 100)

train_df['total_repayment'] = train_df['loan_amount'] + train_df['interest_amount']
test_df['total_repayment'] = test_df['loan_amount'] + test_df['interest_amount']

# Log Transform for skewed features
for col in ['annual_income', 'loan_amount']:
    train_df[f'log_{col}'] = np.log1p(train_df[col])
    test_df[f'log_{col}'] = np.log1p(test_df[col])

# 4. Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Scale numerical columns (original numericals + new features)
# Update numerical columns list to include new features and exclude target
current_numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'loan_paid_back' in current_numerical_cols:
    current_numerical_cols.remove('loan_paid_back')

train_df[current_numerical_cols] = scaler.fit_transform(train_df[current_numerical_cols])
test_df[current_numerical_cols] = scaler.transform(test_df[current_numerical_cols])

print("Feature Engineering Completed.")
print(f"New Train Shape: {train_df.shape}")
print(f"New Test Shape: {test_df.shape}")

# --- Model Selection and Training ---

print("Starting Model Selection and Training...")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    catboost_available = True
except ImportError:
    print("CatBoost not installed. Skipping.")
    catboost_available = False

# Prepare Data
X = train_df.drop('loan_paid_back', axis=1)
y = train_df['loan_paid_back']

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_evaluate_model(model, model_name):
    print(f"\nTraining {model_name}...")
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test_df))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Handle CatBoost specific fitting if needed, but sklearn API usually works
        model.fit(X_train, y_train)
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        
        test_preds += model.predict_proba(test_df)[:, 1] / skf.get_n_splits()
        
        fold_auc = roc_auc_score(y_val, val_preds)
        print(f"Fold {fold+1} AUC: {fold_auc:.4f}")
        
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"Overall AUC for {model_name}: {overall_auc:.4f}")
    
    return oof_preds, test_preds, overall_auc

# Models with Tuned Hyperparameters
models = {
    'XGBoost': xgb.XGBClassifier(
        n_estimators=2000, 
        learning_rate=0.01, 
        max_depth=8, 
        subsample=0.7, 
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=1,
        use_label_encoder=False, 
        eval_metric='logloss', 
        random_state=42, 
        n_jobs=-1
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=2000, 
        learning_rate=0.01, 
        num_leaves=63,
        max_depth=10,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42, 
        n_jobs=-1
    )
}

if catboost_available:
    models['CatBoost'] = cb.CatBoostClassifier(
        iterations=2000, 
        learning_rate=0.01, 
        depth=8, 
        l2_leaf_reg=5,
        eval_metric='AUC', 
        random_seed=42, 
        verbose=0,
        allow_writing_files=False
    )

results = {}
oof_predictions = {}
test_predictions = {}

for name, model in models.items():
    try:
        # For XGBoost early stopping in sklearn API, we need to pass eval_set in fit
        # But for simplicity in this loop structure, we might skip explicit early stopping 
        # or handle it differently. Let's stick to fixed n_estimators for now or simple fit.
        # To properly use early stopping, we'd need to modify the train_evaluate_model function.
        # Let's use the simple fit for now but with more robust params.
        
        oof, test_pred, auc = train_evaluate_model(model, name)
        results[name] = auc
        oof_predictions[name] = oof
        test_predictions[name] = test_pred
    except Exception as e:
        print(f"Failed to train {name}: {e}")

print("\nModel Results:")
for name, auc in results.items():
    print(f"{name}: {auc:.4f}")

# --- Ensembling ---
print("\nCreating Ensemble...")
# Simple Weighted Average based on performance
# We can give more weight to better models.
# Let's assume XGBoost and CatBoost perform best.

weights = {}
total_auc = sum(results.values())
for name, auc in results.items():
    weights[name] = auc / total_auc # Proportional weighting

print(f"Ensemble Weights: {weights}")

ensemble_oof = np.zeros(len(X))
ensemble_test_preds = np.zeros(len(test_df))

for name, weight in weights.items():
    ensemble_oof += oof_predictions[name] * weight
    ensemble_test_preds += test_predictions[name] * weight

ensemble_auc = roc_auc_score(y, ensemble_oof)
print(f"Ensemble OOF AUC: {ensemble_auc:.4f}")

# Feature Importance Analysis
if 'XGBoost' in models:
    xgb_model = models['XGBoost']
    # Fit on full data for feature importance
    xgb_model.fit(X, y)
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Features (XGBoost):")
    print(importance.head(20))
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance.head(20), x='importance', y='feature')
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')

# --- Submission ---
print("Generating Submission File...")
submission = pd.DataFrame({
    'id': test_id,
    'loan_paid_back': ensemble_test_preds
})
submission.to_csv('submission_ensemble.csv', index=False)
print("submission_ensemble.csv created successfully.")



