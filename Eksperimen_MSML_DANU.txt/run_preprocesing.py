import os
import pandas as pd
from preprocessing.automate_danusisda import preprocess_data

if __name__ == "__main__":
    raw_path = "SmokeHealth_raw.csv"
    save_pipeline_path = "preprocessing/preprocessor.joblib"
    save_header_path = "preprocessing/SmokeHealth_preprocessing/columns.csv"
    save_dataset_path = "preprocessing/SmokeHealth_preprocessing"

    os.makedirs(save_dataset_path, exist_ok=True)

    # 1. Load data
    if not os.path.exists(raw_path):
        print(f"Error: File {raw_path} tidak ditemukan!")
    else:
        df = pd.read_csv(raw_path)
        df.columns = df.columns.str.strip()

        if 'blood_pressure' in df.columns:
            print("Mengonversi kolom blood_pressure menjadi angka...")
            # Pecah menjadi dua kolom baru dan ubah jadi float
            df[['bp_systolic', 'bp_diastolic']] = df['blood_pressure'].str.split('/', expand=True).astype(float)
            # Hapus kolom asli yang berbentuk teks agar tidak membuat error di preprocessing
            df = df.drop(columns=['blood_pressure'])

        # 3. Jalankan preprocessing (Target diganti heart_rate sesuai permintaanmu)
        print("Sedang menjalankan preprocessing...")
        X_train, X_test, y_train, y_test = preprocess_data(
            data=df,
            target_column="heart_rate",
            save_path=save_pipeline_path,
            file_path=save_header_path
        )

        # 4. Simpan hasil ke CSV (Data Train & Test)
        print("Menyimpan file hasil preprocessing...")
        
        # Simpan Train
        train_df = pd.DataFrame(X_train)
        train_df['target'] = y_train.values
        train_df.to_csv(f"{save_dataset_path}/SmokeHealth_train_preprocessed.csv", index=False)

        # Simpan Test
        test_df = pd.DataFrame(X_test)
        test_df['target'] = y_test.values
        test_df.to_csv(f"{save_dataset_path}/SmokeHealth_test_preprocessed.csv", index=False)

        print(f"✅ BERHASIL! File disimpan di: {save_dataset_path}")