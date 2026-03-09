from sklearn.metrics import mean_absolute_error, r2_score


def evaluate(model, X_train, X_test, y_train, y_test):
    """
    Обучает модель и возвращает метрики
    """

    # обучение модели
    model.fit(X_train, y_train)

    # предсказание
    predictions = model.predict(X_test)

    # метрики
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "MAE": mae,
        "R2": r2,
        "predictions": predictions
    }