# ==================================================stre
# AI-BASED BREAST CANCER EXPERT SYSTEM (HYBRID)
# Model-driven rule threshold extraction (FIXED)
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import export_text

# ==================================================
# DATA LOADING & CLEANING
# ==================================================
@st.cache_data
def get_clean_data():
    data = pd.read_csv("data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


# ==================================================
# SIDEBAR INPUTS
# ==================================================
def add_sidebar():
    st.sidebar.header("🔬 Cell Nuclei Measurements")
    data = get_clean_data()

    features = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (worst)", "radius_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Area (worst)", "area_worst"),
    ]

    input_dict = {}
    for label, key in features:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict


# ==================================================
# MODEL TRAINING & THRESHOLD EXTRACTION
# ==================================================
@st.cache_resource
def train_model():
    # load data
    data = get_clean_data()
    features = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean",
        "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_worst", "concavity_worst", "area_worst"
    ]

    X = data[features].copy()
    y = data['diagnosis'].copy()

    # scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # train
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        criterion='entropy',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # evaluation
    acc = model.score(X_test, y_test)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')

    # extract thresholds and convert them back to original units
    thresholds = extract_thresholds(model, features)
    thresholds_original = convert_thresholds_to_original(thresholds, features, scaler, X)

    return model, scaler, features, acc, np.mean(cv_scores), thresholds_original


# ==================================================
# AUTOMATIC RULE THRESHOLD EXTRACTION (SCALED SPACE)
# ==================================================
def extract_thresholds(model, features):
    all_thresholds = {f: [] for f in features}

    for estimator in model.estimators_:
        # export_text gives human-readable rules with numeric thresholds (in model feature space)
        try:
            tree_rules = export_text(estimator, feature_names=features)
        except Exception:
            # fallback: some versions might not accept feature_names in this context
            tree_rules = export_text(estimator)
        for line in tree_rules.split('\n'):
            # looking for patterns like "feature_name <= 0.123" or "feature_name > 0.123"
            for f in features:
                if f in line and ("<=" in line or ">" in line):
                    # attempt to parse the numeric value after <= or >
                    if "<=" in line:
                        token = "<="
                    else:
                        token = ">"
                    try:
                        val = float(line.split(token)[1].strip().split()[0])
                        all_thresholds[f].append(val)
                    except Exception:
                        # skip lines that don't parse cleanly
                        continue

    # convert lists -> average scaled threshold (or leave empty list)
    avg_scaled_thresholds = {}
    for f in features:
        if len(all_thresholds[f]) > 0:
            avg_scaled_thresholds[f] = float(np.mean(all_thresholds[f]))
        else:
            avg_scaled_thresholds[f] = None  # no threshold found in any tree
    return avg_scaled_thresholds


def convert_thresholds_to_original(scaled_thresholds, features, scaler, X_original):
    thresholds_original = {}
    # scaler.scale_ and scaler.mean_ correspond to order of features used to fit scaler
    # We'll map features to their index
    feature_index = {f: i for i, f in enumerate(features)}

    for f in features:
        scaled_val = scaled_thresholds.get(f, None)
        idx = feature_index[f]
        if scaled_val is not None:
            # inverse transform single feature: original = scaled * scale + mean
            orig_val = scaled_val * scaler.scale_[idx] + scaler.mean_[idx]
            thresholds_original[f] = float(orig_val)
        else:
            # fallback to mean of original data
            thresholds_original[f] = float(X_original[f].mean())
    return thresholds_original


# ==================================================
# RADAR CHART
# ==================================================
def get_scaled_values(input_dict):
    data = get_clean_data()
    features = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean",
        "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_worst", "concavity_worst", "area_worst"
    ]
    X = data[features]
    scaled_dict = {}
    for key, value in input_dict.items():
        min_val, max_val = X[key].min(), X[key].max()
        if max_val == min_val:
            scaled_dict[key] = 0.0
        else:
            scaled_dict[key] = (value - min_val) / (max_val - min_val)
    return scaled_dict


def get_radar_chart(input_data):
    scaled = get_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area',
                  'Smoothness', 'Compactness', 'Concavity',
                  'Concave Points', 'Symmetry', 'Fractal Dim',
                  'Radius (Worst)', 'Concavity (Worst)', 'Area (Worst)']

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[scaled[k] for k in scaled],
        theta=categories,
        fill='toself',
        name='Feature Values'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False
    )
    return fig


# ==================================================
# MODEL-DRIVEN EXPERT REASONING (uses ORIGINAL-SPACE thresholds)
# ==================================================
def expert_reasoning(input_data, model, scaler, features, thresholds):
   
    input_array = np.array([input_data[f] for f in features]).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    proba = model.predict_proba(input_scaled)[0]
    malignant_prob = float(proba[1])
    benign_prob = float(proba[0])

    importances = model.feature_importances_
    feature_influence = pd.Series(importances, index=features).sort_values(ascending=False)

    rules_triggered = []
    top_n = min(6, len(features))
    for f in feature_influence.index[:top_n]:
        threshold = thresholds.get(f, None)
        value = float(input_data[f])
        if threshold is None:
            
            rules_triggered.append(f"{f}: no model threshold available; value = {value:.3f}.")
            continue

        if value > threshold:
            rules_triggered.append(f"{f} ({value:.3f}) > learned threshold ({threshold:.3f}) → indicator toward malignancy.")
        else:
            rules_triggered.append(f"{f} ({value:.3f}) ≤ learned threshold ({threshold:.3f}) → indicator toward benign.")

    decision = "Malignant" if malignant_prob > 0.5 else "Benign"

    if malignant_prob > 0.9 or benign_prob > 0.9:
        confidence = "High"
    elif 0.6 <= malignant_prob <= 0.9 or 0.6 <= benign_prob <= 0.9:
        confidence = "Moderate"
    else:
        confidence = "Low"

    return decision, confidence, rules_triggered


# ==================================================
# PREDICTIONS DISPLAY
# ==================================================
def add_predictions(input_data, model, scaler, features):
    input_array = np.array([input_data[f] for f in features]).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)[0]

    st.subheader("ML Model Prediction")
    if int(prediction[0]) == 0:
        st.success("Prediction: **Benign**")
    else:
        st.error("Prediction: **Malignant**")

    st.write(f"**Probability (Benign):** {proba[0]:.2f}")
    st.write(f"**Probability (Malignant):** {proba[1]:.2f}")
    return int(prediction[0]), proba


# ==================================================
# MAIN APP
# ==================================================
def main():
    st.set_page_config(page_title="Breast Cancer Expert System (Model-Aided)", layout="wide")
    st.title("AI-BASED BREAST CANCER DIAGNOSIS SYSTEM (Model-Aided Reasoning)")
    st.write("""
    This enhanced expert system uses **Machine Learning** to both **predict diagnosis**
    and **derive rule thresholds** automatically from the Random Forest model structure.
    Thresholds are converted back to original feature units so rule comparisons match the input sliders.
    """)

    input_data = add_sidebar()
    model, scaler, features, acc, cv_acc, thresholds = train_model()

    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(get_radar_chart(input_data))

    with col2:
       

        pred, proba = add_predictions(input_data, model, scaler, features)

        st.subheader("Model-Driven Expert Reasoning")
        decision, confidence, rules = expert_reasoning(input_data, model, scaler, features, thresholds)
        st.write(f"**Expert Decision:** {decision} ({confidence} confidence)")
        st.write("**Triggered Rules:**")
        for r in rules:
            st.write(f"- {r}")

        st.info("⚕️ This hybrid system blends AI prediction with explainable, model-guided reasoning.")

if __name__ == "__main__":
    main()
