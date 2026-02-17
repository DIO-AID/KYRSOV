from sklearn.metrics import mean_absolute_error, r2_score


def evaluate(model, X_train, X_test, y_train, y_test):
    """
    Обучает модель и считает метрики.
    """

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }

    return results
