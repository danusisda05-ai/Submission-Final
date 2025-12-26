import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

train_path = "MLProject/SmokeHealth_preprocessing/SmokeHealth_train_preprocessed.csv"
test_path = "MLProject/SmokeHealth_preprocessing/SmokeHealth_test_preprocessed.csv"

train_df = pd.read_csv(train_path)
test_df =  pd.read_csv(test_path)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]


with mlflow.start_run(run_name="manual_run"):
    mlflow.sklearn.autolog()

    model = RandomForestRegressor()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")