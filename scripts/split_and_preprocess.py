import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

def stratified_split(df):
    """
    Perform a stratified split of the dataset into training and testing sets.
    Parameters:
    - df: DataFrame, the dataset to split.
    Returns:
    - X_train: DataFrame, training features.
    - X_test: DataFrame, testing features.
    - y_train: Series, training labels.
    - y_test: Series, testing labels.
    """
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=26)
    for train_idx, test_idx in sss.split(X, y):
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

def preprocess(X_train, y_train, X_test, y_test):
    """
    Preprocess the training and testing datasets.
    Parameters:
    - X_train: DataFrame, training features.
    - y_train: Series, training labels.
    - X_test: DataFrame, testing features.
    - y_test: Series, testing labels.
    Returns:
    - X_train_encoded: DataFrame, preprocessed training features.
    - y_train_enc: Series, encoded training labels.
    - X_test_encoded: DataFrame, preprocessed testing features.
    - y_test_enc: Series, encoded testing labels.
    - le: LabelEncoder, fitted label encoder for the labels.
    - encoder: OneHotEncoder, fitted one-hot encoder for categorical features.
    """
    le = LabelEncoder()
    scaler = MinMaxScaler()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="error")

    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    X_train.loc[:, "homB":"vacAs1m1"] = X_train.loc[:, "homB":"vacAs1m1"].astype('object')
    X_test.loc[:, "homB":"vacAs1m1"] = X_test.loc[:, "homB":"vacAs1m1"].astype('object')

    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    cat_cols_test = X_test.select_dtypes(include=['object']).columns.tolist()

    one_hot_train = pd.DataFrame(encoder.fit_transform(X_train[cat_cols]),
                                 columns=encoder.get_feature_names_out(cat_cols),
                                 index=X_train.index)
    one_hot_test = pd.DataFrame(encoder.transform(X_test[cat_cols_test]),
                                columns=encoder.get_feature_names_out(cat_cols_test),
                                index=X_test.index)

    X_train_encoded = pd.concat([X_train.drop(cat_cols, axis=1), one_hot_train], axis=1)
    X_test_encoded = pd.concat([X_test.drop(cat_cols_test, axis=1), one_hot_test], axis=1)

    # Convert one-hot columns to category
    cat_features = one_hot_train.columns
    X_train_encoded[cat_features] = X_train_encoded[cat_features].astype('category')
    X_test_encoded[cat_features] = X_test_encoded[cat_features].astype('category')

    return X_train_encoded, y_train_enc, X_test_encoded, y_test_enc, le, encoder
