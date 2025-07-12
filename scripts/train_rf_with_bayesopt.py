# train_rf_bayes.py
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import joblib

def train_rf_with_bayes(X_train, y_train, categorical_indices, cv_splitter):
    """
    Train a Random Forest model using Bayesian optimization with SMOTENC for handling categorical features.
    Parameters:
    - X_train: DataFrame, training features.
    - y_train: Series, training labels.
    - categorical_indices: list, indices of categorical features in X_train.
    - cv_splitter: cross-validation splitter object, e.g., StratifiedKFold.
    Returns:
    - opt: BayesSearchCV object, the trained Random Forest model with hyperparameter optimization.
    """
    rf_pipeline = Pipeline([
        ('smotenc', SMOTENC(categorical_features=categorical_indices, random_state=26)),
        ('classifier', RandomForestClassifier(random_state=26))
    ])

    params_rf_bayes = {
        'classifier__n_estimators': Integer(50, 2000),
        'classifier__criterion': Categorical(['gini', 'entropy', 'log_loss']),
        'classifier__max_depth': Integer(3, 10),
        'classifier__min_samples_split': Integer(2, 10),
        'classifier__min_samples_leaf': Integer(1, 10),
        'classifier__bootstrap': Categorical([True, False]),
        'classifier__ccp_alpha': Real(1e-6, 0.01, prior='log-uniform'),
    }

    opt = BayesSearchCV(
        estimator=rf_pipeline,
        search_spaces=params_rf_bayes,
        scoring='recall',
        cv=cv_splitter,
        n_iter=30,
        n_points=2,
        n_jobs=2,
        verbose=2,
        random_state=26
    )

    opt.fit(X_train, y_train)

    joblib.dump(opt, './models/rf_bayes_opt_model.pkl')
    print("Random Forest model trained and saved as 'rf_bayes_opt_model.pkl'.")
    return opt
