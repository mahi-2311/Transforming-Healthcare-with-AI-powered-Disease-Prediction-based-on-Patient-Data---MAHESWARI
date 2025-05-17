import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    confusion_matrix,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Disease Prediction Based on Symptoms")
st.write(
    "Upload your cleaned disease dataset CSV file with features as symptoms and a 'prognosis' target column."
)

# File uploader
uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")
    st.write("### Dataset Sample")
    st.write(df.head())

    # Prepare features and target
    if "prognosis" not in df.columns:
        st.error("The dataset must contain the target column named 'prognosis'.")
        st.stop()

    X = df.drop("prognosis", axis=1)
    y = df["prognosis"]

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predicted on test set
    y_pred = model.predict(X_test_scaled)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Predict Disease",
            "Evaluation Metrics",
            "Confusion Matrix",
            "ROC Curve",
        ],
    )

    symptoms_list = list(X.columns)

    def get_user_input():
        st.write("Please answer Yes or No for the following symptoms:")
        user_symptoms = []
        for symptom in symptoms_list:
            ans = st.radio(f"Do you have {symptom.replace('_', ' ')}?", ("Yes", "No"), key=symptom)
            user_symptoms.append(1 if ans == "Yes" else 0)
        return [user_symptoms]

    if page == "Predict Disease":
        st.header("Predict Disease from Symptoms")
        user_input = get_user_input()
        if st.button("Predict"):
            user_scaled = scaler.transform(user_input)
            pred_encoded = model.predict(user_scaled)
            disease = le.inverse_transform(pred_encoded)
            st.success(f"🔴 Based on your symptoms, you might have: {disease[0]}")

    elif page == "Evaluation Metrics":
        st.header("Evaluation Metrics")
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.write(f"**Accuracy:** {accuracy * 100:.2f}%")
        st.write(f"**F1 Score:** {f1:.2f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")

    elif page == "Confusion Matrix":
        st.header("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        st.pyplot(plt)
        plt.clf()

    elif page == "ROC Curve":
        st.header("ROC Curve")

        # ROC curve multi-class with one-hot encoding of labels
        from sklearn.preprocessing import label_binarize

        n_classes = len(le.classes_)
        y_test_binarized = label_binarize(y_test, classes=range(n_classes))
        y_score = model.predict_proba(X_test_scaled)

        plt.figure(figsize=(12, 8))

        colors = plt.cm.get_cmap('tab10', n_classes)  # distinct colors colormap with n classes

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr,
                tpr,
                color=colors(i),
                lw=3,
                label=f"Class {le.classes_[i]} (area = {roc_auc:.2f})",
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small')  # legend outside plot right
        plt.tight_layout(rect=[0,0,0.8,1])  # adjust layout to include legend outside
        st.pyplot(plt)
        plt.clf()

else:
    st.info("🔹 Please upload a CSV file to start.")

