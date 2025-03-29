from typing import Dict, List

import kfp
from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Model, Output, component


# Step 1: Load Dataset
@dsl.component(base_image="python:3.11-slim")
def load_data(output_csv: Output[Dataset]):
    import subprocess

    subprocess.run(["pip", "install", "pandas", "scikit-learn"], check=True)

    import pandas as pd
    from sklearn.datasets import load_iris

    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    # Save the dataset to the output artifact path
    df.to_csv(output_csv.path, index=False)


# Step 2: Preprocess Data
@dsl.component(base_image="python:3.11-slim")
def preprocess_data(
    input_csv: Input[Dataset],
    output_train: Output[Dataset],
    output_test: Output[Dataset],
    output_ytrain: Output[Dataset],
    output_ytest: Output[Dataset],
):
    import subprocess

    subprocess.run(["pip", "install", "pandas", "scikit-learn"], check=True)

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Load dataset
    df = pd.read_csv(input_csv.path)

    # Debug: Check for NaN values
    print("Initial dataset shape:", df.shape)
    print("Missing values before preprocessing:\n", df.isnull().sum())

    # Handle missing values
    if df.isnull().values.any():
        print("Missing values detected. Handling them...")
        df = df.dropna()  # Drop rows with any NaN values

    # Validate that there are no NaNs in the target column
    assert (
        not df["target"].isnull().any()
    ), "Target column contains NaN values after handling missing values."

    features = df.drop(columns=["target"])
    target = df["target"]

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, target, test_size=0.2, random_state=42
    )

    # Debug: Validate splits
    print("Shapes after train-test split:")
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("y_train:", y_train.shape, "y_test:", y_test.shape)
    print("Missing values in y_train:", y_train.isnull().sum())

    # Ensure no NaNs in the split data
    assert not y_train.isnull().any(), "y_train contains NaN values."
    assert not y_test.isnull().any(), "y_test contains NaN values."

    # Create DataFrames for train and test sets
    X_train_df = pd.DataFrame(X_train, columns=features.columns)
    print("X_train_df:", X_train_df)

    y_train_df = pd.DataFrame(y_train)
    print("y_train_df: ", y_train_df)

    X_test_df = pd.DataFrame(X_test, columns=features.columns)
    print("X_test_df:", X_test_df)

    y_test_df = pd.DataFrame(y_test)
    print("y_test_df: ", y_test_df)

    # Save processed train and test data
    X_train_df.to_csv(output_train.path, index=False)
    X_test_df.to_csv(output_test.path, index=False)

    y_train_df.to_csv(output_ytrain.path, index=False)
    y_test_df.to_csv(output_ytest.path, index=False)


# Step 3: Train Model
@dsl.component(base_image="python:3.11-slim")
def train_model(
    train_data: Input[Dataset], ytrain_data: Input[Dataset], model_output: Output[Model]
):

    import subprocess

    subprocess.run(
        ["pip", "install", "pandas", "scikit-learn", "joblib"], check=True
    )  # Add joblib to the list of dependencies

    import pandas as pd
    from joblib import dump
    from sklearn.linear_model import LogisticRegression

    # Load training data
    train_df = pd.read_csv(train_data.path)
    print("Shape of train_df:", train_df.shape)
    print("train_df:", train_df)
    X_train = train_df

    y_train = pd.read_csv(ytrain_data.path)
    print("Shape of ytrain_df:", y_train.shape)
    print("y_train_df:", y_train)

    # Debug: Validate splits
    print("Shapes of X_train and y_train: ")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("Missing values in X_train:", X_train.isnull().sum())
    print("Missing values in y_train:", y_train.isnull().sum())

    # Ensure no NaN values
    assert not X_train.isnull().values.any(), "X_train contains NaN values."
    assert not y_train.isnull().values.any(), "y_train contains NaN values."

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save model
    dump(model, model_output.path)


# Step 4: Evaluate Model
@dsl.component(base_image="python:3.11-slim")
def evaluate_model(
    test_data: Input[Dataset],
    ytest_data: Input[Dataset],
    model: Input[Model],
    metrics_output: Output[Dataset],
):
    import subprocess

    subprocess.run(
        ["pip", "install", "pandas", "scikit-learn", "matplotlib", "joblib"], check=True
    )

    import matplotlib.pyplot as plt
    import pandas as pd
    from joblib import load
    from sklearn.metrics import classification_report, confusion_matrix

    # Load test data
    X_test = pd.read_csv(test_data.path)

    y_test = pd.read_csv(ytest_data.path)

    # Load model
    model = load(model.path)

    # Predict
    y_pred = model.predict(X_test)

    # Generate metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Save metrics to a file
    metrics_path = metrics_output.path
    with open(metrics_path, "w") as f:
        f.write(str(report))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(metrics_path.replace(".txt", ".png"))


# Define the pipeline
@dsl.pipeline(name="ml-pipeline")
def ml_pipeline():
    # Step 1: Load Dataset
    load_op = load_data()

    # Step 2: Preprocess Data
    preprocess_op = preprocess_data(input_csv=load_op.outputs["output_csv"])

    # Step 3: Train Model
    train_op = train_model(
        train_data=preprocess_op.outputs["output_train"],
        ytrain_data=preprocess_op.outputs["output_ytrain"],
    )

    # Step 4: Evaluate Model
    evaluate_op = evaluate_model(
        test_data=preprocess_op.outputs["output_test"],
        ytest_data=preprocess_op.outputs["output_ytest"],
        model=train_op.outputs["model_output"],
    )


# Compile the pipeline
if __name__ == "__main__":
    # compiler.Compiler().compile(
    #     pipeline_func=ml_pipeline, package_path="kubeflow_pipeline.yaml"
    # )
    client = kfp.Client(host="http://192.168.1.106:8080")
    run = client.create_run_from_pipeline_func(
        ml_pipeline,
    )
