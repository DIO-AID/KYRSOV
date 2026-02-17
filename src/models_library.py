# ===============================
# MODELS LIBRARY
# ===============================

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# =====================================================
# XGBOOST
# =====================================================
def get_xgb_model():
    """XGBoost Regressor — хорошо для табличных данных, устойчив к выбросам"""
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0,
        reg_lambda=1,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42
    )

# =====================================================
# LIGHTGBM
# =====================================================
def get_lgbm_model():
    """LightGBM — быстрый градиентный бустинг"""
    return LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42
    )

# =====================================================
# CATBOOST
# =====================================================
def get_catboost_model():
    """CatBoost — отлично работает с категориальными признаками"""
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        bagging_temperature=1,
        random_strength=1,
        loss_function="RMSE",
        verbose=False,
        random_state=42
    )

# =====================================================
# RANDOM FOREST
# =====================================================
def get_rf_model():
    """Random Forest — стабильная базовая модель"""
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

# =====================================================
# EXTRA TREES
# =====================================================
def get_extra_trees_model():
    """Extra Trees — более случайный RandomForest"""
    return ExtraTreesRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

# =====================================================
# GRADIENT BOOSTING
# =====================================================
def get_gb_model():
    """Классический Gradient Boosting"""
    return GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=0.8,
        random_state=42
    )

# =====================================================
# HIST GRADIENT BOOSTING
# =====================================================
def get_hist_gb_model():
    """Быстрый градиентный бустинг (аналог LightGBM)"""
    return HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42
    )

# =====================================================
# LINEAR REGRESSION
# =====================================================
def get_linear_model():
    return LinearRegression()

# =====================================================
# RIDGE
# =====================================================
def get_ridge_model():
    return Ridge(alpha=1.0, random_state=42)

# =====================================================
# LASSO
# =====================================================
def get_lasso_model():
    return Lasso(alpha=0.1, max_iter=10000, random_state=42)

# =====================================================
# ELASTIC NET
# =====================================================
def get_elastic_model():
    return ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42)
