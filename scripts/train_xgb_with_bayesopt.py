from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from xgboost import XGBClassifier
import joblib

def train_xgb_with_bayes(X_train, y_train, cat_indices, cv_splitter):
    """
    Train an XGBoost model using Bayesian optimization with SMOTENC for handling categorical features.
    Parameters:
    - X_train: DataFrame, training features.
    - y_train: Series, training labels.
    - cat_indices: list, indices of categorical features in X_train.
    - cv_splitter: cross-validation splitter object, e.g., StratifiedKFold.
    Returns:
    - opt: BayesSearchCV object, the trained XGBoost model with hyperparameter optimization.
    """
    pipeline = Pipeline([
        ('smote', SMOTENC(categorical_features=cat_indices, random_state=26)),
        ('classifier', XGBClassifier(
            objective='binary:logistic',
            tree_method='hist',
            enable_categorical=True,
            eval_metric='logloss',
            random_state=26
        ))
    ])

    search_space = {
        'classifier__n_estimators': Integer(50, 1000),
        'classifier__max_depth': Integer(3, 7),
        'classifier__learning_rate': Real(0.01, 0.1),
        'classifier__subsample': Real(0.5, 1),
        'classifier__colsample_bytree': Real(0.5, 1),
        'classifier__min_child_weight': Integer(1, 10),
        'classifier__gamma': Real(0, 5),
        'classifier__reg_alpha': Real(1e-9, 10.0, prior='log-uniform')
    }

    opt = BayesSearchCV(
        estimator=pipeline,
        search_spaces=search_space,
        scoring='recall',
        cv=cv_splitter,
        n_iter=30,
        n_points=2,
        n_jobs=2,
        verbose=2,
        random_state=26
    )

    opt.fit(X_train, y_train)

    joblib.dump(opt, "./models/xgb_bayes_opt_model.pkl")
    print("XGBoost model trained and saved as 'xgb_bayes_opt_model.pkl'.")

    return opt
