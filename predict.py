import joblib
import pandas as pd

from feature_extraction import (
    load_signal,
    normalize_signal,
    generate_iq,
    extract_features
)

# -----------------------------
# Load Models
# -----------------------------

brand_model = joblib.load("models/brand_model.pkl")
model_model = joblib.load("models/model_model.pkl")
user_model = joblib.load("models/user_stack.pkl")

scaler = joblib.load("models/scaler.pkl")

brand_encoder = joblib.load("models/brand_encoder.pkl")
model_encoder = joblib.load("models/model_encoder.pkl")
user_encoder = joblib.load("models/user_encoder.pkl")


def predict(file_path):

    # Load signal
    signal = load_signal(file_path)

    # Normalize
    signal = normalize_signal(signal)

    # IQ
    I, Q = generate_iq(signal)

    # Features
    features = extract_features(I, Q)

    X = pd.DataFrame([features])

    X = scaler.transform(X)

    # Predictions
    brand = brand_encoder.inverse_transform(
        brand_model.predict(X)
    )[0]

    model = model_encoder.inverse_transform(
        model_model.predict(X)
    )[0]

    user = user_encoder.inverse_transform(
        user_model.predict(X)
    )[0]

    # Confidence
    brand_conf = brand_model.predict_proba(X).max() * 100
    model_conf = model_model.predict_proba(X).max() * 100
    user_conf = user_model.predict_proba(X).max() * 100

    return {
        "brand": brand,
        "model": model,
        "user": user,

        "brand_conf": brand_conf,
        "model_conf": model_conf,
        "user_conf": user_conf,

        "signal": signal,
        "features": features
    }