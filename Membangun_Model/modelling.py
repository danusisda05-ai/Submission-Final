import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt # Tambahkan untuk grafik
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Path dataset
train_path = "SmokeHealth_preprocessing/SmokeHealth_train_preprocessed.csv"
test_path = "SmokeHealth_preprocessing/SmokeHealth_test_preprocessed.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("SmokeHealth_Prediction")

with mlflow.start_run(run_name="manual_run"):
    # Pastikan autolog dipanggil SEBELUM model.fit
    mlflow.sklearn.autolog()

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # --- BAGIAN AGAR METRIK MUNCUL DI TAB METRICS ---
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # --- TAMBAHKAN ARTIFACT GRAFIK (OPTIONAL TAPI BAGUS) ---
    plt.figure(figsize=(10,6))
    feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Top 10 Feature Importances")
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png") # Muncul sebagai gambar di UI

    print(f"MSE: {mse}, R2: {r2}")
    print("Cek tab 'Metrics' untuk angka dan 'Artifacts' untuk file/model.")