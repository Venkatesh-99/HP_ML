import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

def stratified_split(df):
    """
    Perform a stratified split of the dataset into training and testing sets.
    """
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=26)
    for train_idx, test_idx in sss.split(X, y):
        return X.iloc[train_idx].copy(), X.iloc[test_idx].copy(), y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

def preprocess(X_train, y_train, X_test, y_test):
    """
    Preprocess the training and testing datasets.
    Ensures 'Gastric cancer' = 1 and 'Non-gastric cancer' = 0.
    """
    scaler = MinMaxScaler()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    le = LabelEncoder()

    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print(f"Label encoding mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"Gastric cancer count in y_train: {sum(y_train_enc)}")
    print(f"Non-gastric cancer count in y_train: {len(y_train_enc) - sum(y_train_enc)}")

    # ✅ Scale numeric features
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # ✅ Convert specified columns to categorical
    if "homB" in X_train.columns and "vacAs1m1" in X_train.columns:
        X_train_scaled.loc[:, "homB":"vacAs1m1"] = X_train_scaled.loc[:, "homB":"vacAs1m1"].astype('object')
        X_test_scaled.loc[:, "homB":"vacAs1m1"] = X_test_scaled.loc[:, "homB":"vacAs1m1"].astype('object')

    # ✅ One-hot encode categorical features
    cat_cols = X_train_scaled.select_dtypes(include=['object']).columns.tolist()
    one_hot_train = pd.DataFrame(encoder.fit_transform(X_train_scaled[cat_cols]),
                                 columns=encoder.get_feature_names_out(cat_cols),
                                 index=X_train_scaled.index)
    one_hot_test = pd.DataFrame(encoder.transform(X_test_scaled[cat_cols]),
                                columns=encoder.get_feature_names_out(cat_cols),
                                index=X_test_scaled.index)

    X_train_encoded = pd.concat([X_train_scaled.drop(cat_cols, axis=1), one_hot_train], axis=1)
    X_test_encoded = pd.concat([X_test_scaled.drop(cat_cols, axis=1), one_hot_test], axis=1)

    # Convert one-hot columns to category
    cat_features = one_hot_train.columns
    X_train_encoded[cat_features] = X_train_encoded[cat_features].astype('category')
    X_test_encoded[cat_features] = X_test_encoded[cat_features].astype('category')

    return X_train_encoded, y_train_enc, X_test_encoded, y_test_enc, le, encoder
