import joblib
import numpy as np
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "xgboost_credit_model.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def predict_credit_risk(input_data):
    model = load_model()

    # Get fitted preprocessing pipeline
    preprocessor = model.named_steps["preprocessor"]
    xgb_model = model.named_steps["classifier"]

    # Exact feature order expected by the trained model
    feature_names = [
        "Attribute1", "Attribute2", "Attribute3", "Attribute4", "Attribute5",
        "Attribute6", "Attribute7", "Attribute8", "Attribute9", "Attribute10",
        "Attribute11", "Attribute12", "Attribute13", "Attribute14", "Attribute15",
        "Attribute16", "Attribute17", "Attribute18", "Attribute19", "Attribute20"
    ]

    # Build numpy input without pandas
    X = np.array(
        [[input_data[col] for col in feature_names]],
        dtype=object
    )

    # Column positions used during training
    numeric_indices = [1, 4, 7, 10, 12, 15, 17]
    categorical_indices = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]

    # Extract fitted transformers
    numeric_transformer = preprocessor.transformers_[0][1]
    categorical_transformer = preprocessor.transformers_[1][1]

    # Apply the same preprocessing manually
    X_numeric = X[:, numeric_indices].astype(float)
    X_categorical = X[:, categorical_indices]

    X_numeric_scaled = numeric_transformer.transform(X_numeric)
    X_categorical_encoded = categorical_transformer.transform(X_categorical)

    # Combine transformed features
    if hasattr(X_categorical_encoded, "toarray"):
        X_categorical_encoded = X_categorical_encoded.toarray()

    X_transformed = np.hstack([
        X_numeric_scaled,
        X_categorical_encoded
    ])

    # Make prediction using the trained XGBoost model
    prediction = xgb_model.predict(X_transformed)[0]
    probability = xgb_model.predict_proba(X_transformed)[0][1]

    return prediction, probability


if __name__ == "__main__":

    sample_data = {
        "Attribute1": "A11",
        "Attribute2": 6,
        "Attribute3": "A34",
        "Attribute4": "A43",
        "Attribute5": 1169,
        "Attribute6": "A65",
        "Attribute7": "A75",
        "Attribute8": 4,
        "Attribute9": "A93",
        "Attribute10": "A101",
        "Attribute11": 4,
        "Attribute12": "A121",
        "Attribute13": 67,
        "Attribute14": "A143",
        "Attribute15": "A152",
        "Attribute16": 2,
        "Attribute17": "A173",
        "Attribute18": 1,
        "Attribute19": "A192",
        "Attribute20": "A201"
    }

    prediction, probability = predict_credit_risk(sample_data)

    print("Prediction:", prediction)
    print("Risk Probability:", f"{probability * 100:.2f}%")