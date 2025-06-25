import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Configuration ----
st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="wide")
USER_CSV = "user.csv"
PREDICTION_LOG_CSV = "predictions_log.csv"

# ---- Utility: Set Background ----
def set_background(image_path):
    if not os.path.exists(image_path):
        return
    with open(image_path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode()
    bg_style = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64}");
            background-size: cover;
        }}
        </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

# ---- User Handling ----
def load_users():
    if os.path.exists(USER_CSV):
        return pd.read_csv(USER_CSV)
    return pd.DataFrame(columns=["username", "password"])

def save_user(username, password):
    users = load_users()
    if username in users["username"].values:
        return False
    users = pd.concat([users, pd.DataFrame({"username": [username], "password": [password]})], ignore_index=True)
    users.to_csv(USER_CSV, index=False)
    return True

def validate_user(username, password):
    users = load_users()
    return ((users.username == username) & (users.password == password)).any()

# ---- Model & Prediction ----
@st.cache_resource
def load_model(model_dir='model_files'):
    m = os.path.join(model_dir, 'best_model.pkl')
    s = os.path.join(model_dir, 'scaler.pkl')
    if not os.path.exists(m) or not os.path.exists(s):
        st.error("Model or scaler not found.")
        return None, None
    return joblib.load(m), joblib.load(s)

def predict_fraud(features, model, scaler):
    return model.predict_proba(scaler.transform(features))[:, 1][0]

def log_prediction(data):
    if os.path.exists(PREDICTION_LOG_CSV):
        df = pd.read_csv(PREDICTION_LOG_CSV)
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    else:
        df = pd.DataFrame([data])
    df.to_csv(PREDICTION_LOG_CSV, index=False)

def plot_confusion_matrix(tp, fp, fn, tn):
    fig, ax = plt.subplots()
    sns.heatmap([[tp, fp], [fn, tn]], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Legit", "Pred Fraud"],
                yticklabels=["Actual Legit", "Actual Fraud"], ax=ax)
    ax.set_title("Confusion Matrix")
    return fig

# ---- Pages ----
def register_page():
    set_background("backgrounds/register_bg.png")
    st.title("🔐 Register")
    u = st.text_input("Create Username")
    p = st.text_input("Create Password", type="password")
    if st.button("Register"):
        if save_user(u, p):
            st.success("Registration successful.")
        else:
            st.error("Username exists.")

def login_page():
    set_background("backgrounds/login_bg.png")
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if validate_user(u, p):
            st.session_state.authenticated = True
            st.session_state.username = u
            st.success("Logged in.")
        else:
            st.error("Invalid credentials.")

def home_page():
    set_background("backgrounds/home_bg.png")
    st.title("🏠 Home")
    st.markdown("Welcome to the **Credit Card Fraud Detection App**.")

def predict_page():
    set_background("backgrounds/predict_bg.png")
    st.title("💳 Credit Card Fraud Detection")
    model, scaler = load_model()
    if model is None:
        return

    with st.form("form"):
        amt = st.number_input("Amount", 0.0, 10000.0, 100.0)
        lat = st.number_input("Cust Lat", -90.0, 90.0, 40.0)
        lon = st.number_input("Cust Long", -180.0, 180.0, -74.0)
        pop = st.number_input("City Pop", 0, 1000000, 100000)
        ts = st.number_input("Unix Time", 0, 2**31 - 1, 1577836800)
        mlat = st.number_input("Merch Lat", -90.0, 90.0, 40.0)
        mlon = st.number_input("Merch Long", -180.0, 180.0, -74.0)

        if st.form_submit_button("Predict"):
            f = pd.DataFrame({
                "amt": [amt], "lat": [lat], "long": [lon],
                "city_pop": [pop], "unix_time": [ts],
                "merch_lat": [mlat], "merch_long": [mlon]
            })
            prob = predict_fraud(f, model, scaler)
            cls = 1 if prob > 0.5 else 0
            pred = "🚨 FRAUD" if cls == 1 else "✅ LEGIT"
            risk = "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"

            st.subheader("Results")
            st.metric("Fraud Prob", f"{prob:.2%}")
            st.metric("Status", pred)
            st.metric("Risk", risk)

            st.subheader("🮢 Confusion Matrix")
            st.pyplot(plot_confusion_matrix(1 if cls == 1 else 0, 0, 0, 1))

            mets = {
                "accuracy": (cls == cls) / 1.0,
                "precision": 1 if cls == 1 else 0,
                "recall": 1 if cls == 1 else 0,
                "f1_score": 1 if cls == 1 else 0
            }

            log_prediction({
                "username": st.session_state.get("username", "guest"),
                "prob": round(prob, 4),
                "pred": pred,
                "risk": risk,
                "accuracy": round(mets['accuracy'], 2),
                "precision": round(mets['precision'], 2),
                "recall": round(mets['recall'], 2),
                "f1_score": round(mets['f1_score'], 2),
                "amt": amt,
                "lat": lat,
                "long": lon,
                "city_pop": pop,
                "unix_time": ts,
                "merch_lat": mlat,
                "merch_long": mlon
            })

def dashboard_page():
    st.title("📊 Dashboard")

    if os.path.exists(PREDICTION_LOG_CSV):
        df = pd.read_csv(PREDICTION_LOG_CSV)
        st.subheader("📋 Recent Predictions")
        st.dataframe(df.tail(10))

        st.subheader("📈 Metric Trends Over Time")

        metrics = ["accuracy", "precision", "recall", "f1_score"]
        cols = st.columns(2)

        for idx, metric in enumerate(metrics):
            with cols[idx % 2]:
                st.markdown(f"#### {metric.capitalize()}")
                df_plot = df[[metric]].reset_index().rename(columns={metric: "Value"})
                line_chart = alt.Chart(df_plot).mark_line(point=True).encode(
                    x=alt.X("index:Q", title="Prediction Index"),
                    y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1])),
                    tooltip=["Value"]
                ).properties(height=300)
                st.altair_chart(line_chart, use_container_width=True)
    else:
        st.info("No predictions yet.")

def main():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        o = st.sidebar.radio("Menu", ["Login", "Register"])
        login_page() if o == "Login" else register_page()
    else:
        p = st.sidebar.selectbox("Navigate", ["Home", "Predict", "Dashboard", "Logout"])
        {"Home": home_page, "Predict": predict_page, "Dashboard": dashboard_page}.get(
            p, lambda: st.session_state.update({"authenticated": False}) or st.experimental_rerun()
        )()

if __name__ == "__main__":
    main()
