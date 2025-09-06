from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
import joblib

def train_svm_with_bayes(X_train, y_train, categorical_indices, cv_splitter):
    """
    Train an SVM model using Bayesian optimization with SMOTENC for handling categorical features.
    Parameters:
    - X_train: DataFrame, training features.
    - y_train: Series, training labels.
    - categorical_indices: list, indices of categorical features in X_train.
    - cv_splitter: cross-validation splitter object, e.g., StratifiedKFold.
    Returns:
    - opt: BayesSearchCV object, the trained SVM model with hyperparameter optimization.
    """
    svm_pipeline = Pipeline([
        ('smotenc', SMOTENC(categorical_features=categorical_indices, random_state=26)),
        ('classifier', SVC(random_state=26, probability=True))  # probability=True for predict_proba
    ])

    params_svm_bayes = {
    'classifier__C': Real(1e-3, 1e3, prior='log-uniform'),
    'classifier__kernel': Categorical(['linear', 'rbf']),  # Focused selection
    'classifier__gamma': Categorical(['scale', 'auto']),
    'classifier__class_weight': Categorical([None, 'balanced'])
}

    opt = BayesSearchCV(
        estimator=svm_pipeline,
        search_spaces=params_svm_bayes,
        scoring='recall',
        cv=cv_splitter,
        n_iter=30,
        n_points=2,
        n_jobs=2,
        verbose=2,
        random_state=26
    )

    opt.fit(X_train, y_train)

    joblib.dump(opt, './models/svm_bayes_opt_model.pkl')
    print("SVM model trained and saved as 'svm_bayes_opt_model.pkl'.")
    return opt