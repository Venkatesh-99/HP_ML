from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTENC
import joblib
import os

def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model using SMOTENC for handling categorical features.
    Parameters:
    - X_train: DataFrame, training features.
    - y_train: Series, training labels.
    Returns:
    - pipeline: Pipeline, the trained logistic regression model with SMOTENC.
    """
    os.makedirs("./models", exist_ok=True)

    cat_cols = X_train.select_dtypes(include=['category']).columns
    cat_indices = [X_train.columns.get_loc(col) for col in cat_cols]

    pipeline = make_pipeline(
        SMOTENC(categorical_features=cat_indices, random_state=26),
        LogisticRegression(random_state=26)
    )

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, "./models/baseline_lr_model.pkl")
    print("Logistic Regression model trained and saved as 'baseline_lr_model.pkl'.")
    return pipeline
