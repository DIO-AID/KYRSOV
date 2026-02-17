from sklearn.pipeline import Pipeline
from .models_library import (
    get_xgb_model, get_lgbm_model, get_catboost_model,
    get_rf_model, get_extra_trees_model, get_gb_model, get_hist_gb_model,
    get_linear_model, get_ridge_model, get_lasso_model, get_elastic_model
)

def build_model(preprocessor, model_name="xgb"):
    """
    Создает Pipeline с препроцессором + выбранной моделью.
    model_name определяет тип модели:
    "xgb", "lgbm", "catboost", "rf", "extra_trees",
    "gb", "hist_gb", "linear", "ridge", "lasso", "elastic"
    """

    models = {
        "xgb": get_xgb_model,
        "lgbm": get_lgbm_model,
        "catboost": get_catboost_model,
        "rf": get_rf_model,
        "extra_trees": get_extra_trees_model,
        "gb": get_gb_model,
        "hist_gb": get_hist_gb_model,
        "linear": get_linear_model,
        "ridge": get_ridge_model,
        "lasso": get_lasso_model,
        "elastic": get_elastic_model
    }

    if model_name not in models:
        raise ValueError(f"Неизвестная модель: {model_name}")

    model_func = models[model_name]

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model_func())  # модель ещё не обучена
    ])

    return pipeline
