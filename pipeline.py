from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model, Artifact, Metrics, ClassificationMetrics


# =========================
# 1. LOAD DATA
# =========================
@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["pandas", "numpy"]
)
def load_data(
    input_csv_path: str,
    raw_dataset: Output[Dataset],
):
    import os
    import pandas as pd
    import numpy as np

    def generate_sample_dataframe() -> pd.DataFrame:
        np.random.seed(42)
        n = 6500

        wiek = np.random.randint(18, 70, n)
        dochod = np.random.randint(3000, 25000, n)
        wizyty_www = np.random.randint(1, 50, n)
        czas_na_stronie = np.random.uniform(0.5, 20.0, n)

        score = (
            0.02 * wiek
            + 0.00015 * dochod
            + 0.12 * wizyty_www
            + 0.35 * czas_na_stronie
            - 8
        )

        prob = 1 / (1 + np.exp(-score))
        kupi = (np.random.rand(n) < prob).astype(int)

        df = pd.DataFrame({
            "wiek": wiek,
            "dochod": dochod,
            "wizyty_www": wizyty_www,
            "czas_na_stronie": czas_na_stronie,
            "kupi": kupi
        })
        return df

    if os.path.exists(input_csv_path):
        df = pd.read_csv(input_csv_path)
        if df.empty:
            raise ValueError(f"CSV file exists but is empty: {input_csv_path}")
        print(f"CSV file found: {input_csv_path}")
    else:
        print(f"CSV file not found: {input_csv_path}")
        print("Generating sample dataset customers.csv ...")
        df = generate_sample_dataframe()

    df.to_csv(raw_dataset.path, index=False)

    print(f"Saved raw dataset to: {raw_dataset.path}")
    print("Data preview:")
    print(df.head())
    print("Shape:", df.shape)


# =========================
# 2. PREPROCESS
# =========================
@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["pandas", "numpy", "scikit-learn", "joblib"]
)
def preprocess_data(
    raw_dataset: Input[Dataset],
    x_train: Output[Dataset],
    x_test: Output[Dataset],
    y_train: Output[Dataset],
    y_test: Output[Dataset],
    scaler_artifact: Output[Artifact],
    preprocess_metadata: Output[Artifact],
):
    import json
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    RANDOM_STATE = 42

    df = pd.read_csv(raw_dataset.path)

    required_columns = ["wiek", "dochod", "wizyty_www", "czas_na_stronie", "kupi"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    feature_columns = ["wiek", "dochod", "wizyty_www", "czas_na_stronie"]
    target_column = "kupi"

    X = df[feature_columns]
    y = df[target_column]

    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    pd.DataFrame(X_train_scaled, columns=feature_columns).to_csv(x_train.path, index=False)
    pd.DataFrame(X_test_scaled, columns=feature_columns).to_csv(x_test.path, index=False)
    pd.DataFrame({"kupi": y_train_df.values}).to_csv(y_train.path, index=False)
    pd.DataFrame({"kupi": y_test_df.values}).to_csv(y_test.path, index=False)

    joblib.dump(scaler, scaler_artifact.path)

    metadata = {
        "feature_columns": feature_columns,
        "target_column": target_column,
        "train_rows": int(len(X_train_df)),
        "test_rows": int(len(X_test_df)),
        "input_features": int(len(feature_columns))
    }

    with open(preprocess_metadata.path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Preprocessing finished")
    print(metadata)


# =========================
# 3. TRAIN
# =========================
@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["pandas", "numpy", "torch"]
)
def train_model(
    x_train: Input[Dataset],
    y_train: Input[Dataset],
    preprocess_metadata: Input[Artifact],
    model_artifact: Output[Model],
    training_metrics: Output[Artifact],
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001,
):
    import json
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    with open(preprocess_metadata.path, "r") as f:
        meta = json.load(f)

    feature_columns = meta["feature_columns"]
    input_dim = meta["input_features"]

    X_train_df = pd.read_csv(x_train.path)
    y_train_df = pd.read_csv(y_train.path)

    X_train_tensor = torch.tensor(X_train_df.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_df.values.reshape(-1, 1), dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    class CustomerClassifier(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            return self.network(x)

    model = CustomerClassifier(input_dim=input_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - loss={avg_loss:.6f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "feature_columns": feature_columns,
            "model_class": "CustomerClassifier",
        },
        model_artifact.path
    )

    metrics = {
        "final_train_loss": float(loss_history[-1]),
        "loss_history": [float(x) for x in loss_history],
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
    }

    with open(training_metrics.path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model saved to:", model_artifact.path)
    print("Training metrics saved to:", training_metrics.path)


# =========================
# 4. EVALUATE
# =========================
@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["pandas", "numpy", "torch", "scikit-learn"]
)
def evaluate_model(
    x_test: Input[Dataset],
    y_test: Input[Dataset],
    model_artifact: Input[Model],
    evaluation_metrics: Output[Artifact],
    scalar_metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics],
):
    import json
    import pandas as pd
    import torch
    import math
    import torch.nn as nn
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_auc_score,
        roc_curve,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    checkpoint = torch.load(model_artifact.path, map_location=device)
    input_dim = checkpoint["input_dim"]

    class CustomerClassifier(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            return self.network(x)

    model = CustomerClassifier(input_dim=input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_test_df = pd.read_csv(x_test.path)
    y_test_df = pd.read_csv(y_test.path)

    X_test_tensor = torch.tensor(X_test_df.values, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    preds = (probs >= 0.5).astype(int)
    y_true = y_test_df["kupi"].values

    accuracy = float(accuracy_score(y_true, preds))
    precision = float(precision_score(y_true, preds, zero_division=0))
    recall = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, probs))
    cm = confusion_matrix(y_true, preds).tolist()

    # Scalar metrics tab
    scalar_metrics.log_metric("accuracy", accuracy)
    scalar_metrics.log_metric("precision", precision)
    scalar_metrics.log_metric("recall", recall)
    scalar_metrics.log_metric("f1", f1)
    scalar_metrics.log_metric("roc_auc", roc_auc)

    # Confusion matrix tab
    classification_metrics.log_confusion_matrix(
        ["0", "1"],
        cm
    )

    # ROC curve tab
    fpr, tpr, thresholds = roc_curve(y_true, probs)

    roc_fpr = []
    roc_tpr = []
    roc_thresholds = []

    for i in range(len(fpr)):
        fp = float(fpr[i])
        tp = float(tpr[i])
        th = float(thresholds[i])

        if math.isfinite(fp) and math.isfinite(tp) and math.isfinite(th):
            roc_fpr.append(fp)
            roc_tpr.append(tp)
            roc_thresholds.append(th)

    if roc_fpr and roc_tpr and roc_thresholds:
        classification_metrics.log_roc_curve(
            roc_fpr,
            roc_tpr,
            roc_thresholds,
        )

    # Full JSON artifact for S3/debugging
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
    }

    with open(evaluation_metrics.path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation metrics:")
    print(json.dumps(metrics, indent=2))


# =========================
# 5. UPLOAD TO S3
# =========================
@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["boto3"]
)
def upload_to_s3(
    model_artifact: Input[Model],
    scaler_artifact: Input[Artifact],
    evaluation_metrics: Input[Artifact],
    s3_endpoint: str,
    s3_bucket: str,
    s3_prefix: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_region: str = "us-east-1",
):
    import os
    import boto3

    session = boto3.session.Session()
    s3 = session.client(
        service_name="s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )

    objects_to_upload = [
        (model_artifact.path, f"{s3_prefix}/model.pt"),
        (scaler_artifact.path, f"{s3_prefix}/scaler.joblib"),
        (evaluation_metrics.path, f"{s3_prefix}/evaluation_metrics.json"),
    ]

    for local_path, s3_key in objects_to_upload:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"File not found for upload: {local_path}")

        s3.upload_file(local_path, s3_bucket, s3_key)
        print(f"Uploaded {local_path} -> s3://{s3_bucket}/{s3_key}")

    print("S3 upload finished")


# =========================
# 6. OPTIONAL PREDICT SAMPLE
# =========================
@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["pandas", "torch", "joblib", "scikit-learn"]
)
def predict_sample(
    model_artifact: Input[Model],
    scaler_artifact: Input[Artifact],
):
    import joblib
    import pandas as pd
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_artifact.path, map_location=device)
    scaler = joblib.load(scaler_artifact.path)

    input_dim = checkpoint["input_dim"]
    feature_columns = checkpoint["feature_columns"]

    class CustomerClassifier(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            return self.network(x)

    model = CustomerClassifier(input_dim=input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample = pd.DataFrame([
        {"wiek": 28, "dochod": 5000, "wizyty_www": 2, "czas_na_stronie": 1.1},
        {"wiek": 44, "dochod": 14000, "wizyty_www": 16, "czas_na_stronie": 7.5},
        {"wiek": 51, "dochod": 22000, "wizyty_www": 28, "czas_na_stronie": 15.3},
    ])

    X = sample[feature_columns].values
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)

    sample["purchase_probability"] = probs
    sample["prediction"] = preds

    print(sample.to_string(index=False))


# =========================
# 7. PIPELINE
# =========================
@dsl.pipeline(
    name="enterprise-pytorch-training-pipeline",
    description="Enterprise style pipeline for PyTorch training on OpenShift AI"
)
def enterprise_pytorch_pipeline(
    input_csv_path: str = "/opt/app-root/src/data/customers.csv",
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    upload_enabled: bool = True,
    s3_endpoint: str = "http://minio.minio.svc.cluster.local:9000",
    s3_bucket: str = "models",
    s3_prefix: str = "customer-classifier/run-001",
    aws_access_key_id: str = "minioadmin",
    aws_secret_access_key: str = "minioadmin",
    aws_region: str = "us-east-1",
):
    load_task = load_data(input_csv_path=input_csv_path)

    preprocess_task = preprocess_data(
        raw_dataset=load_task.outputs["raw_dataset"]
    )

    train_task = train_model(
        x_train=preprocess_task.outputs["x_train"],
        y_train=preprocess_task.outputs["y_train"],
        preprocess_metadata=preprocess_task.outputs["preprocess_metadata"],
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    evaluate_task = evaluate_model(
        x_test=preprocess_task.outputs["x_test"],
        y_test=preprocess_task.outputs["y_test"],
        model_artifact=train_task.outputs["model_artifact"],
    )

    predict_task = predict_sample(
        model_artifact=train_task.outputs["model_artifact"],
        scaler_artifact=preprocess_task.outputs["scaler_artifact"],
    )
    
    # Disable caching
    load_task.set_caching_options(enable_caching=False)
    preprocess_task.set_caching_options(enable_caching=False)
    train_task.set_caching_options(enable_caching=False)
    evaluate_task.set_caching_options(enable_caching=False)
    predict_task.set_caching_options(enable_caching=False)

    with dsl.If(upload_enabled == True):
        upload_task = upload_to_s3(
            model_artifact=train_task.outputs["model_artifact"],
            scaler_artifact=preprocess_task.outputs["scaler_artifact"],
            evaluation_metrics=evaluate_task.outputs["evaluation_metrics"],
            s3_endpoint=s3_endpoint,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region=aws_region,
        )
        upload_task.set_caching_options(enable_caching=False)


# =========================
# 8. COMPILE
# =========================
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=enterprise_pytorch_pipeline,
        package_path="enterprise_pytorch_pipeline.yaml",
        pipeline_name="enterprise-pytorch-pipeline"
    )