"""Chapter 8 pitfalls, each as a wrong and a correct version (constitution Principle V).

The ``wrong_*`` functions are deliberately incorrect — never copy them.
"""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.tree import DecisionTreeClassifier

from db_ml_associate.ch08_foundations.imbalance import (
    make_churn_classifier,
    random_oversample,
)
from db_ml_associate.common import CHURN_LABEL, DEFAULT_SEED, split_frame


def _tree_f1(train: pd.DataFrame, validation: pd.DataFrame, seed: int) -> float:
    model = make_churn_classifier(estimator=DecisionTreeClassifier(random_state=seed))
    model.fit(train.drop(columns=CHURN_LABEL), train[CHURN_LABEL])
    predictions = model.predict(validation.drop(columns=CHURN_LABEL))
    return float(f1_score(validation[CHURN_LABEL], predictions))


def wrong_resample_before_split(churn: pd.DataFrame, seed: int = DEFAULT_SEED) -> float:
    """⚠️ WRONG: oversample the whole dataset, then split. Returns an inflated validation F1.

    Copies of the same minority row land in both splits, so the validation set contains rows the
    model has memorised: the score measures recall of training data, not generalisation.
    """
    # --8<-- [start:wrong_resample_before_split]
    balanced = random_oversample(churn, CHURN_LABEL, seed)  # duplicates before the split ...
    train, validation = split_frame(balanced, CHURN_LABEL, seed=seed)  # ... leak into validation
    # --8<-- [end:wrong_resample_before_split]
    return _tree_f1(train, validation, seed)


def correct_resample_after_split(churn: pd.DataFrame, seed: int = DEFAULT_SEED) -> float:
    """Split first, then oversample only the training part. Returns an honest validation F1."""
    # --8<-- [start:correct_resample_after_split]
    train, validation = split_frame(churn, CHURN_LABEL, seed=seed)
    train = random_oversample(train, CHURN_LABEL, seed)  # validation keeps the real class ratio
    # --8<-- [end:correct_resample_after_split]
    return _tree_f1(train, validation, seed)


def wrong_accuracy_on_imbalanced(churn: pd.DataFrame, seed: int = DEFAULT_SEED) -> dict[str, float]:
    """⚠️ WRONG: judge a churn model by accuracy. A model that never predicts churn scores ~90 %."""
    train, test = split_frame(churn, CHURN_LABEL, seed=seed)
    always_stays = DummyClassifier(strategy="most_frequent").fit(train.drop(columns=CHURN_LABEL), train[CHURN_LABEL])
    predictions = always_stays.predict(test.drop(columns=CHURN_LABEL))
    return {
        "accuracy": float(accuracy_score(test[CHURN_LABEL], predictions)),
        "recall": float(recall_score(test[CHURN_LABEL], predictions, zero_division=0)),
    }


def correct_use_recall_f1(churn: pd.DataFrame, seed: int = DEFAULT_SEED) -> dict[str, float]:
    """Report recall and F1 of the minority class for a class-weighted model."""
    train, test = split_frame(churn, CHURN_LABEL, seed=seed)
    model = make_churn_classifier(class_weight="balanced")
    model.fit(train.drop(columns=CHURN_LABEL), train[CHURN_LABEL])
    predictions = model.predict(test.drop(columns=CHURN_LABEL))
    return {
        "accuracy": float(accuracy_score(test[CHURN_LABEL], predictions)),
        "recall": float(recall_score(test[CHURN_LABEL], predictions)),
        "f1": float(f1_score(test[CHURN_LABEL], predictions)),
    }
