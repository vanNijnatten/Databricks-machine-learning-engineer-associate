"""Chapter 10 pitfalls, each as a wrong and a correct version (constitution Principle V).

The ``wrong_*`` functions are deliberately incorrect — never copy them.
"""

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, rand, space_eval
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from db_ml_associate.ch08_foundations.imbalance import make_churn_classifier
from db_ml_associate.ch10_tuning.hyperopt_tuning import cross_validated_f1
from db_ml_associate.common import CHURN_LABEL, DEFAULT_SEED, split_frame

DEPTHS = [2, 3, 4, 5, 6, 8, 10, 14]
LEAF_SIZES = [1, 5, 20]
FEATURE_FRACTIONS = [0.3, 0.6, 1.0]


@dataclass(frozen=True)
class TunedScore:
    """The score a workflow reports and the score the chosen model really achieves on new data."""

    reported: float
    on_new_data: float


def _three_way_split(churn: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """60 % train, a small 10 % test set and 30 % "future" data the workflow never sees."""
    train, rest = split_frame(churn, CHURN_LABEL, test_size=0.4, seed=seed)
    test, new_data = split_frame(rest, CHURN_LABEL, test_size=0.75, seed=seed)
    return train, test, new_data


def _tree(depth: int, leaf: int, max_features: float, seed: int) -> Pipeline:
    return make_churn_classifier(
        estimator=DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=leaf,
            max_features=max_features,
            class_weight="balanced",
            random_state=seed,
        )
    )


def _f1(model: Pipeline, frame: pd.DataFrame) -> float:
    return float(f1_score(frame[CHURN_LABEL], model.predict(frame.drop(columns=CHURN_LABEL))))


def _wrong_once(churn: pd.DataFrame, seed: int) -> TunedScore:
    train, test, new_data = _three_way_split(churn, seed)
    features, labels = train.drop(columns=CHURN_LABEL), train[CHURN_LABEL]
    # --8<-- [start:wrong_tune_on_test_set]
    candidates = [  # 8 x 3 x 3 = 72 tree configurations
        _tree(depth, leaf, fraction, seed).fit(features, labels)
        for depth, leaf, fraction in product(DEPTHS, LEAF_SIZES, FEATURE_FRACTIONS)
    ]
    best = max(candidates, key=lambda model: _f1(model, test))  # the test set chooses the model ...
    reported = _f1(best, test)  # ... and then grades its own choice: optimistic
    # --8<-- [end:wrong_tune_on_test_set]
    return TunedScore(reported=reported, on_new_data=_f1(best, new_data))


def _correct_once(churn: pd.DataFrame, seed: int) -> TunedScore:
    train, test, new_data = _three_way_split(churn, seed)
    # --8<-- [start:correct_tune_with_cv_then_test]
    search = GridSearchCV(
        _tree(3, 1, 1.0, seed),
        param_grid={
            "model__max_depth": DEPTHS,
            "model__min_samples_leaf": LEAF_SIZES,
            "model__max_features": FEATURE_FRACTIONS,
        },
        scoring="f1",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
    ).fit(train.drop(columns=CHURN_LABEL), train[CHURN_LABEL])
    reported = _f1(search.best_estimator_, test)  # first and only look at the test set
    # --8<-- [end:correct_tune_with_cv_then_test]
    return TunedScore(reported=reported, on_new_data=_f1(search.best_estimator_, new_data))


def _average(scores: list[TunedScore]) -> TunedScore:
    return TunedScore(
        reported=float(np.mean([score.reported for score in scores])),
        on_new_data=float(np.mean([score.on_new_data for score in scores])),
    )


def wrong_tune_on_test_set(churn: pd.DataFrame, repeats: int = 8, seed: int = DEFAULT_SEED) -> TunedScore:
    """⚠️ WRONG: pick hyperparameters by their test score and report that same score.

    Selection bias is a statistical effect, so the result is averaged over ``repeats`` random
    splits: on average the reported score overstates the score on new data.
    """
    return _average([_wrong_once(churn, seed + repeat) for repeat in range(repeats)])


def correct_tune_with_cv_then_test(churn: pd.DataFrame, repeats: int = 8, seed: int = DEFAULT_SEED) -> TunedScore:
    """Choose with cross-validation on training data; use the test set once (averaged over splits)."""
    return _average([_correct_once(churn, seed + repeat) for repeat in range(repeats)])


@dataclass(frozen=True)
class FminChoice:
    """Parameters chosen by ``fmin`` and their cross-validated F1."""

    params: dict[str, Any]
    f1: float


PITFALL_SPACE = {
    "C": hp.choice("C", [0.0001, 1.0]),
    "class_weight": hp.choice("class_weight", [None, "balanced"]),
}


def _run_fmin(churn: pd.DataFrame, sign: int, seed: int) -> FminChoice:
    train, _ = split_frame(churn, CHURN_LABEL, seed=seed)

    def objective(params: dict[str, Any]) -> dict[str, Any]:
        return {
            "loss": sign * cross_validated_f1(train, params, seed=seed),
            "status": STATUS_OK,
        }

    best = space_eval(
        PITFALL_SPACE,
        fmin(
            objective,
            PITFALL_SPACE,
            algo=rand.suggest,
            max_evals=12,
            trials=Trials(),
            rstate=np.random.default_rng(seed),
            show_progressbar=False,
        ),
    )
    return FminChoice(params=dict(best), f1=cross_validated_f1(train, best, seed=seed))


def wrong_fmin_maximise_metric(churn: pd.DataFrame, seed: int = DEFAULT_SEED) -> FminChoice:
    """⚠️ WRONG: return F1 itself as the loss. ``fmin`` minimises, so it finds the *worst* model.

    The objective body would be ``return {"loss": f1, "status": STATUS_OK}``.
    """
    return _run_fmin(churn, sign=+1, seed=seed)


def correct_return_negative_metric(churn: pd.DataFrame, seed: int = DEFAULT_SEED) -> FminChoice:
    """Return ``-f1`` as the loss so that minimising the loss maximises F1."""
    return _run_fmin(churn, sign=-1, seed=seed)
