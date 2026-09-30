from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .config import ModelConfig, ModelsConfig


MODEL_LABELS = {
    "logistic_l1": "Logistic regression L1",
    "logistic_l2": "Logistic regression L2",
    "random_forest": "Random forest",
    "svm_linear": "SVM lineare",
    "svm_rbf": "SVM RBF",
    "svm_poly": "SVM polinomiale",
    "svm_sigmoid": "SVM sigmoid",
}


def _base_estimator(model_name: str, random_state: int):
    if model_name == "logistic_l1":
        return LogisticRegression(l1_ratio=1.0, random_state=random_state)
    if model_name == "logistic_l2":
        return LogisticRegression(l1_ratio=0.0, random_state=random_state)
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=random_state)
    if model_name.startswith("svm_"):
        kernel = model_name.removeprefix("svm_")
        return SVC(kernel=kernel, random_state=random_state)
    raise ValueError(f"Modello non supportato: {model_name}")


def build_estimator(
    model_name: str,
    model_config: ModelConfig,
    *,
    random_state: int,
):
    estimator = _base_estimator(model_name, random_state)
    parameters = dict(model_config.params)
    if model_name.startswith("svm_"):
        # SVC.probability è deprecato; decision_function è sufficiente per AUC.
        parameters.pop("probability", None)

    valid_parameters = estimator.get_params(deep=True)
    unknown_parameters = sorted(set(parameters) - set(valid_parameters))
    if unknown_parameters:
        raise ValueError(
            f"Parametri non validi per '{model_name}': "
            + ", ".join(unknown_parameters)
        )
    try:
        estimator.set_params(**parameters)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Configurazione non valida per il modello '{model_name}': {error}"
        ) from error
    return estimator


def build_estimators(
    config: ModelsConfig,
    *,
    random_state: int,
) -> dict[str, object]:
    estimators: dict[str, object] = {}
    for model_name, model_config in config.enabled_models().items():
        estimators[model_name] = build_estimator(
            model_name,
            model_config,
            random_state=random_state,
        )

    if not estimators:
        raise ValueError("Nessun modello e' abilitato in config/config_analysis.yaml")
    return estimators
