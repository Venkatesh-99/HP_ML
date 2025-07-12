import os
import joblib
from sklearn.calibration import CalibratedClassifierCV

def calibrate_classifier(estimator, X_train, y_train, cv_splitter, save_dir="./models"):
    """
    Calibrate a classifier using CalibratedClassifierCV.

    Parameters:
    - estimator: The base classifier or search CV object (e.g., BayesSearchCV) to calibrate.
    - X_train: Training features.
    - y_train: Training labels.
    - cv_splitter: Cross-validation splitter for calibration.
    - save_dir: Directory to save the calibrated model.

    Returns:
    - calibrated_model: The calibrated classifier.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Unwrap BayesSearchCV or GridSearchCV to get the best estimator
    if hasattr(estimator, "best_estimator_"):
        estimator = estimator.best_estimator_

    # Handle nested estimators (e.g., Pipeline)
    if hasattr(estimator, 'estimator'):
        inner_estimator = estimator.estimator
    else:
        inner_estimator = estimator

    if hasattr(inner_estimator, 'named_steps'):
        base_estimator_name = inner_estimator.named_steps['classifier'].__class__.__name__.lower()
    else:
        base_estimator_name = inner_estimator.__class__.__name__.lower()

    save_path = os.path.join(save_dir, f"{base_estimator_name}_calibrated_model.pkl")

    # Calibrate the classifier
    calibrated_model = CalibratedClassifierCV(estimator, method='sigmoid', cv=cv_splitter)
    calibrated_model.fit(X_train, y_train)

    print(f"Calibrated {base_estimator_name} model trained and saved as '{save_path}'.")

    # Save the calibrated model
    joblib.dump(calibrated_model, save_path)

    return calibrated_model
