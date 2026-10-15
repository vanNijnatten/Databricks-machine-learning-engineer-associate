"""A scikit-learn training pipeline for the churn data (exam objective S3.04).

scikit-learn uses the same idea with different words: a *transformer* has ``fit()`` and
``transform()``; an *estimator* (predictor) has ``fit()`` and ``predict()``. ``fit()`` updates the
object in place instead of returning a separate model object.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from db_ml_associate.common import CHURN_CATEGORICAL, CHURN_NUMERIC


def build_sklearn_churn_pipeline() -> Pipeline:
    """Return an unfitted pipeline: impute + scale numerics, one-hot categoricals, logistic regression."""
    # --8<-- [start:sklearn_pipeline]
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore")  # unseen categories become all-zero vectors
    prepare = ColumnTransformer(
        [
            ("numeric", numeric, list(CHURN_NUMERIC)),
            ("categorical", categorical, list(CHURN_CATEGORICAL)),
        ]
    )
    return Pipeline(
        [
            ("prepare", prepare),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )
    # --8<-- [end:sklearn_pipeline]
