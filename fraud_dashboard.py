import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# App Config

st.set_page_config(page_title="Credit Card Fraud Detection Dashboard", layout="wide")


# Load data
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
else:
    st.warning("No file uploaded. Using default `creditcard.csv` dataset.")
    @st.cache_data
    def load_data(path="creditcard.csv"):
        return pd.read_csv(path)
    df = load_data()


# Basic preprocessing
required_cols = ["Class"] + [f"V{i}" for i in range(1,29)]
if not all(col in df.columns for col in required_cols):
    st.error(f"Uploaded dataset must have columns: {required_cols}")
else:
    y = df["Class"]
    X = df.drop(["Class", "Time", "Amount"], axis=1, errors='ignore')

    # Train-validation-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, shuffle=True, random_state=8
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.17, random_state=8
    )

# Sidebar: global options

st.sidebar.header("⚙️ Settings")

# Shared feature selection
features_selected = st.sidebar.multiselect(
    "Select features for GMM / Prediction",
    options=X.columns.tolist() if 'X' in locals() else [],
    default=["V14", "V17", "V12"] if 'X' in locals() else []
)


st.title("Credit Card Fraud Detection Dashboard")

# Initialize session state for GMM persistence
if "gmm_model" not in st.session_state:
    st.session_state.gmm_model = None
if "gmm_features" not in st.session_state:
    st.session_state.gmm_features = None
if "rf_model" not in st.session_state:
    st.session_state.rf_model = None
if "rf_features" not in st.session_state:
    st.session_state.rf_features = None

# Tabs

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🔍 EDA", "⚙️ Model Training", "📈 Model Comparison", "🧾 Prediction"]
)


# TAB 1: Overview

with tab1:
    st.subheader("Dataset Overview")

    # Metrics
    col1, col2 = st.columns(2)
    col1.metric("Total Transactions", df.shape[0])
    col2.metric("Features (excluding Time/Amount/Class)", df.shape[1] - 3)

    st.write("### Transaction Class Distribution (Balanced Donut Chart)")

    # Actual counts
    class_counts = df["Class"].value_counts().sort_index()
    labels = ["Non-Fraud", "Fraud"]
    colors = ["#2ca02c", "#d62728"]  # Dark green / dark red

    # Exaggerate fraud slice visually for donut only
    plot_counts = class_counts.copy()
    plot_counts[1] = plot_counts[1] * 20  # exaggerate visually

    # Create donut chart
    fig, ax = plt.subplots(figsize=(6,6))
    wedges, texts = ax.pie(
        plot_counts.values,
        labels=labels,
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor='w')
    )

    # Add actual counts and percentages manually
    total = class_counts.sum()
    for i, p in enumerate(wedges):
        angle = (p.theta2 + p.theta1)/2.
        x = 0.7 * np.cos(np.deg2rad(angle))
        y = 0.7 * np.sin(np.deg2rad(angle))
        pct = class_counts[i] / total * 100
        ax.text(x, y, f"{class_counts[i]} ({pct:.2f}%)", ha='center', va='center', fontsize=12, fontweight='bold')

    # Total transactions in center
    ax.text(0, 0, f"Total\n{df.shape[0]}", ha='center', va='center', fontsize=14, fontweight='bold')

    # Title
    ax.set_title("Fraud vs Non-Fraud Transactions", fontsize=16, fontweight='bold')
    st.pyplot(fig)


# TAB 2: EDA

with tab2:
    st.subheader("Feature Distribution (Train set)")

    # Color palette
    nonfraud_color = "#4dd0e1"  # cyan/light blue
    fraud_color = "#f06292"     # soft pink

    feature = st.selectbox("Select a feature to visualize", X.columns.tolist())

    fig, ax = plt.subplots(figsize=(9, 4))

    # Non-Fraud histogram (density)
    sns.histplot(
        X_train[feature][y_train == 0],
        bins=20,
        stat="density",
        color=nonfraud_color,
        label="Non-Fraud",
        kde=True,
        ax=ax,
        alpha=0.6
    )

    # Fraud histogram (density)
    sns.histplot(
        X_train[feature][y_train == 1],
        bins=20,
        stat="density",
        color=fraud_color,
        label="Fraud",
        kde=True,
        ax=ax,
        alpha=0.6
    )

    ax.set_title(f"Distribution of {feature}", fontsize=14, fontweight='bold')
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(title="Transaction Type", fontsize=11)
    sns.despine()

    st.pyplot(fig)

    # Correlation heatmap (full dataset)
    st.write("### Correlation Heatmap (Full Dataset)")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df.corr(),
        ax=ax2,
        cmap="coolwarm",
        annot=False,
        linewidths=0.5,
        linecolor='gray'
    )
    ax2.set_title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
    st.pyplot(fig2)


# TAB 3: Model Training (Baseline = Random Forest)

with tab3:
    st.subheader("Train Random Forest Classifier (Baseline Model)")

    if not features_selected:
        st.warning("Select at least one feature in the sidebar to train the Random Forest.")
    else:
        st.write("Using features:", features_selected)

        n_estimators = st.number_input("Number of Trees", min_value=10, max_value=500, value=100, step=10)
        max_depth = st.number_input("Max Depth", min_value=0, max_value=50, value=0, step=1)

        if st.button("🚀 Train & Save Random Forest"):
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None if max_depth == 0 else max_depth,
                random_state=0,
                class_weight="balanced"
            )
            rf.fit(X_train[features_selected], y_train)

            # Save to session_state
            st.session_state.rf_model = rf
            st.session_state.rf_features = features_selected

            # Evaluate on validation set
            preds_val = rf.predict(X_val[features_selected])
            probas_val = rf.predict_proba(X_val[features_selected])[:, 1]

            prec = precision_score(y_val, preds_val, zero_division=0)
            rec = recall_score(y_val, preds_val, zero_division=0)
            f1 = f1_score(y_val, preds_val, zero_division=0)
            try:
                auc = roc_auc_score(y_val, probas_val)
            except Exception:
                auc = np.nan

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Precision", round(prec, 4))
            c2.metric("Recall", round(rec, 4))
            c3.metric("F1", round(f1, 4))
            c4.metric("ROC AUC", round(auc, 4) if not np.isnan(auc) else "N/A")

            st.success("Random Forest trained and saved to session.")


# TAB 4: Model Comparison

with tab4:
    st.subheader("Compare Models on Test Set")
    if not features_selected:
        st.warning("Select at least one feature in the sidebar to run comparisons.")
    else:
        # define models
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0, class_weight="balanced"),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Gaussian Mixture (GMM)": "gmm",
        }

        results = []
        for name, model_obj in models.items():
            if name.startswith("Gaussian"):
                # Use stored GMM if available & same features, else train quickly
                if st.session_state.gmm_model is not None and st.session_state.gmm_features == features_selected:
                    gmm = st.session_state.gmm_model
                else:
                    gmm = GaussianMixture(n_components=1, random_state=0)
                    gmm.fit(X_train[features_selected][y_train == 0].values)

                scores = gmm.score_samples(X_test[features_selected].values)
                preds = np.array([1 if s < (-50) else 0 for s in scores])
                score_for_auc = scores

            elif name.startswith("Random Forest"):
                rf = model_obj
                rf.fit(X_train[features_selected], y_train)
                preds = rf.predict(X_test[features_selected])
                try:
                    score_for_auc = rf.predict_proba(X_test[features_selected])[:, 1]
                except Exception:
                    score_for_auc = preds

            else:  # Logistic Regression
                clf = model_obj
                clf.fit(X_train[features_selected], y_train)
                preds = clf.predict(X_test[features_selected])
                try:
                    score_for_auc = clf.predict_proba(X_test[features_selected])[:, 1]
                except Exception:
                    score_for_auc = preds

            # Safely compute metrics
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            try:
                auc = roc_auc_score(y_test, score_for_auc)
            except Exception:
                auc = np.nan

            results.append([name, round(prec, 4), round(rec, 4), round(f1, 4),
                           (round(auc, 4) if not np.isnan(auc) else "N/A")])

        res_df = pd.DataFrame(results, columns=["Model", "Precision", "Recall", "F1", "ROC AUC"])
        st.dataframe(res_df.set_index("Model"))

        try:
            chart_df = res_df.set_index("Model")[["Precision", "Recall", "F1"]].astype(float)
            st.bar_chart(chart_df)
        except Exception:
            pass


# TAB 5: Prediction (Playground)

with tab5:
    st.subheader("Fraud Detection Playground")

    if not features_selected:
        st.warning("Select at least one feature in the sidebar to make a prediction.")
    else:
        st.write("Using features:", features_selected)
        model_choice = st.selectbox(
            "Choose model for prediction",
            ["Random Forest", "Logistic Regression", "Gaussian Mixture (GMM)"]
        )

        # Generate default input from a random sample
        sample_input = X_train[features_selected].sample(1, random_state=42).iloc[0].to_dict()

        # Dynamic numeric inputs for each selected feature
        input_vals = {}
        cols = st.columns(len(features_selected))
        for i, feat in enumerate(features_selected):
            input_vals[feat] = cols[i].number_input(
                feat,
                value=float(sample_input[feat]),  # default to sampled value
                format="%.6f"
            )

        input_df = pd.DataFrame([input_vals])

        if st.button("🔎 Predict Fraud"):
            if model_choice.startswith("Gaussian"):
                # use stored gmm or train on the fly
                if st.session_state.gmm_model is not None and st.session_state.gmm_features == features_selected:
                    gmm = st.session_state.gmm_model
                else:
                    gmm = GaussianMixture(n_components=1, random_state=0)
                    gmm.fit(X_train[features_selected][y_train == 0].values)
                    st.session_state.gmm_model = gmm
                    st.session_state.gmm_features = features_selected

                score_input = gmm.score_samples(input_df[features_selected].values)[0]
                pred_label = 1 if score_input < (-50) else 0

                st.metric("Prediction", "Fraud" if pred_label == 1 else "Non-Fraud")
                st.write("GMM score (log-likelihood):", float(score_input))
                st.write(f"Decision threshold used: {(-50)}")

            elif model_choice.startswith("Random Forest"):
                # use stored RF model or train on the fly
                if "rf_model" in st.session_state and st.session_state.rf_features == features_selected:
                    rf = st.session_state.rf_model
                else:
                    rf = RandomForestClassifier(
                        n_estimators=100,
                        random_state=0,
                        class_weight="balanced"
                    )
                    rf.fit(X_train[features_selected], y_train)
                    st.session_state.rf_model = rf
                    st.session_state.rf_features = features_selected

                prob = rf.predict_proba(input_df[features_selected])[0][1]
                pred_label = int(rf.predict(input_df[features_selected])[0])

                st.metric("Prediction", "Fraud" if pred_label == 1 else "Non-Fraud")
                st.progress(float(prob))
                st.write("Fraud probability (Random Forest):", round(prob, 4))

            else:  # Logistic Regression
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_train[features_selected], y_train)
                prob = clf.predict_proba(input_df[features_selected])[0][1]
                pred_label = int(clf.predict(input_df[features_selected])[0])

                st.metric("Prediction", "Fraud" if pred_label == 1 else "Non-Fraud")
                st.progress(float(prob))
                st.write("Fraud probability (Logistic):", round(prob, 4))

