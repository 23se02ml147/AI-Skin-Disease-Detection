import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import re
import tempfile
import json
from datetime import datetime
from collections import Counter
import pandas as pd

from database import (
    create_database,
    register_user,
    login_user,
    save_prediction,
    get_prediction_history,
    get_prediction_count,
    get_user_profile,
    update_profile,
    change_password,
    delete_prediction,
    get_activity_log,
)

from report import create_prediction_report


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Skin Disease Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

os.makedirs("uploads", exist_ok=True)


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding-top: 20px;
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.1);
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 12px 20px;
        font-size: 16px;
        font-weight: 500;
        width: 100%;
        text-align: left;
        transition: all 0.3s ease;
        margin-bottom: 5px;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.25);
        border-color: #4fc3f7;
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(79, 195, 247, 0.3);
    }
    .sidebar-user {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .sidebar-user h3 { color: #ffffff !important; margin-bottom: 5px; font-size: 18px; }
    .sidebar-user p { color: rgba(255, 255, 255, 0.7) !important; font-size: 14px; }
    .sidebar-user .avatar { font-size: 50px; margin-bottom: 10px; }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px 0;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }
    .card {
        background: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .card h2, .card h3 { color: #1a1a2e; }

    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 25px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        color: white;
    }
    .prediction-card h2 { color: white !important; }
    .prediction-card p { color: rgba(255, 255, 255, 0.9) !important; }
    .disease-name {
        font-size: 36px;
        font-weight: 800;
        color: #ffffff;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }
    .metric-card h3 { color: #1a1a2e; font-size: 28px; margin: 10px 0; }
    .metric-card p { color: #666; font-size: 14px; }

    .history-card {
        background: white;
        padding: 18px 22px;
        border-radius: 15px;
        margin-bottom: 12px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    .history-card:hover {
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        transform: translateX(5px);
    }
    .history-card h3 { color: #1a1a2e; margin-bottom: 8px; }
    .history-card p { color: #666; margin: 4px 0; }

    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        border: none;
        padding: 12px 24px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    .stFileUploader {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 18px;
        padding: 20px;
        border: 2px dashed #667eea;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    hr { border-color: rgba(255, 255, 255, 0.1); margin: 15px 0; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATABASE INIT
# ============================================================

create_database()


# ============================================================
# LOAD CLASSES
# ============================================================

def load_classes():
    if os.path.exists("classes.json"):
        with open("classes.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return [
        "Acne", "Eczema", "Psoriasis", "Rosacea",
        "Vitiligo", "Tinea / Ringworm", "Impetigo",
        "Seborrheic Dermatitis"
    ]

CLASS_NAMES = load_classes()


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "name" not in st.session_state:
    st.session_state.name = ""
if "page" not in st.session_state:
    st.session_state.page = "Login"
if "last_prediction_key" not in st.session_state:
    st.session_state.last_prediction_key = None


# ============================================================
# VALIDATION
# ============================================================

def valid_email(email):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, email) is not None

def valid_phone(phone):
    return phone.isdigit() and len(phone) == 10


# ============================================================
# LOAD MODEL (UPDATED FOR SPLIT BASE + HEAD)
# ============================================================

@st.cache_resource
def load_skin_model():
    head_path = "skin_model_head.keras"
    
    # Check if the new split model exists
    if os.path.exists(head_path):
        try:
            # 1. Load the base model (Feature Extractor)
            base_model = tf.keras.applications.EfficientNetB0(
                include_top=False, weights="imagenet", pooling="avg", input_shape=(224, 224, 3)
            )
            base_model.trainable = False
            
            # 2. Load the trained custom head
            head_model = tf.keras.models.load_model(head_path)
            
            # 3. Combine them into one seamless model for prediction
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = tf.keras.applications.efficientnet.preprocess_input(inputs)
            x = base_model(x, training=False)
            outputs = head_model(x)
            
            final_model = tf.keras.Model(inputs, outputs)
            return final_model, "EfficientNetB0 + Custom Head"
            
        except Exception as error:
            st.error(f"Unable to load the split model: {error}")
            st.stop()
            
    # Fallback to old models if split model doesn't exist
    possible_models = [
        "best_skin_model.keras",
        "skin_model.keras",
        "best_skin_model.h5",
        "skin_model.h5",
    ]
    for model_path in possible_models:
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                return model, model_path
            except Exception as error:
                st.error(f"Unable to load {model_path}: {error}")
                st.stop()
                
    st.error("❌ No trained skin disease model found. Please run train_fast.py first.")
    st.stop()


# ============================================================
# GRAD-CAM HELPERS
# ============================================================

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    try:
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer_name = layer.name
                    break
                if hasattr(layer, "layers"):
                    for sub in reversed(layer.layers):
                        if isinstance(sub, tf.keras.layers.Conv2D):
                            last_conv_layer_name = sub.name
                            break
                    if last_conv_layer_name:
                        break

        if last_conv_layer_name is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception:
        return None


def overlay_heatmap(image, heatmap):
    try:
        import cv2
    except ImportError:
        return None
    img = np.array(image.resize((224, 224)))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(superimposed)


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():
    st.markdown('<div class="main-title">🩺 AI Skin Disease Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-Powered Skin Disease Analysis System</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔐 Login")

        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        if st.button("🔑 Login", use_container_width=True):
            if not username or not password:
                st.error("Please enter username and password.")
            else:
                user = login_user(username.strip(), password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.username = user[1]
                    st.session_state.name = user[2]
                    st.session_state.page = "Dashboard"
                    st.success(f"Welcome, {user[2]}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        st.divider()
        if st.button("📝 New User? Create Account", use_container_width=True):
            st.session_state.page = "Sign Up"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# SIGN UP PAGE
# ============================================================

def signup_page():
    st.markdown('<div class="main-title">🩺 Create Your Account</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Register to use the AI Skin Disease Detection System</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 Sign Up")

        username = st.text_input("👤 Username")
        name = st.text_input("🧑 Full Name")
        email = st.text_input("📧 Email Address")
        phone = st.text_input("📱 Phone Number")
        password = st.text_input("🔐 Create Password", type="password")
        confirm_password = st.text_input("🔐 Confirm Password", type="password")

        if st.button("✅ Create Account", use_container_width=True):
            if not username.strip():
                st.error("Please enter username.")
            elif not name.strip():
                st.error("Please enter your name.")
            elif not valid_email(email):
                st.error("Please enter a valid email.")
            elif not valid_phone(phone):
                st.error("Phone number must contain 10 digits.")
            elif len(password) < 6:
                st.error("Password must contain at least 6 characters.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = register_user(
                    username.strip(), name.strip(),
                    email.strip(), phone.strip(), password
                )
                if success:
                    st.success("✅ Account created successfully!")
                    st.info("Please login using your new account.")
                    st.session_state.page = "Login"
                else:
                    st.error(message)

        st.divider()
        if st.button("← Back to Login", use_container_width=True):
            st.session_state.page = "Login"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

def sidebar():
    st.sidebar.markdown(f"""
    <div class="sidebar-user">
        <div class="avatar">👤</div>
        <h3>{st.session_state.name}</h3>
        <p>@{st.session_state.username}</p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    nav_items = {
        "🏠 Dashboard": "Dashboard",
        "🔬 Skin Prediction": "Prediction",
        "📸 Live Camera": "Camera",
        "📋 Prediction History": "History",
        "📊 Analytics": "Analytics",
        "📚 Disease Information": "Disease Info",
        "🤖 Model Information": "Model Info",
        "👤 My Profile": "Profile",
        "⚙️ Settings": "Settings",
    }

    for label, page in nav_items.items():
        if st.sidebar.button(label, use_container_width=True):
            st.session_state.page = page
            st.rerun()

    st.sidebar.markdown("---")

    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.name = ""
        st.session_state.page = "Login"
        st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

def dashboard_page():
    st.markdown('<div class="main-title">🏠 Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">Welcome back, {st.session_state.name} 👋</div>', unsafe_allow_html=True)

    prediction_count = get_prediction_count(st.session_state.username)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>🔬</h2>
            <h3>{prediction_count}</h3>
            <p>Total Predictions</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2>🧬</h2>
            <h3>{len(CLASS_NAMES)}</h3>
            <p>Disease Classes</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>🤖</h2>
            <h3>CNN</h3>
            <p>AI Model</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown(f"""
    <div class="card">
        <h2>🔬 AI Skin Disease Detection</h2>
        <p>Upload a skin image and use the trained CNN model to generate an AI-based classification.</p>
        <p>This system can detect <b>{len(CLASS_NAMES)}</b> different skin diseases.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔬 Start New Prediction", use_container_width=True):
        st.session_state.page = "Prediction"
        st.rerun()

    logs = get_activity_log(st.session_state.username, limit=5)
    if logs:
        st.subheader("🕐 Recent Activity")
        for action, ts in logs:
            st.write(f"- **{action}** — {ts}")


# ============================================================
# PREDICTION CORE
# ============================================================

def run_prediction(image, image_name):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📷 Uploaded Image")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    img_resized = image.resize((224, 224))
    # NOTE: We do NOT rescale here because the combined model handles it
    img_array_raw = np.array(img_resized).astype("float32")
    img_array = np.expand_dims(img_array_raw, axis=0)

    model, model_name = load_skin_model()

    with st.spinner("Analyzing image..."):
        try:
            predictions = model.predict(img_array, verbose=0)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

    top_indices = np.argsort(predictions)[::-1][:3]
    top3 = [
        (CLASS_NAMES[i], float(predictions[i] * 100))
        for i in top_indices if i < len(CLASS_NAMES)
    ]

    if not top3:
        st.error("Unable to generate prediction.")
        return

    predicted_class, confidence = top3[0]

    saved_filename = f"{st.session_state.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    saved_path = os.path.join("uploads", saved_filename)
    try:
        image.save(saved_path)
    except Exception:
        saved_path = None

    prediction_key = (
        st.session_state.username,
        image_name,
        predicted_class,
        round(confidence, 2),
    )

    if confidence >= 50 and st.session_state.last_prediction_key != prediction_key:
        save_prediction(
            st.session_state.username,
            predicted_class,
            confidence,
            image_name,
            saved_path,
        )
        st.session_state.last_prediction_key = prediction_key

    with col2:
        st.markdown(f"""
        <div class="prediction-card">
            <h2>🩺 Prediction Result</h2>
            <p>Predicted Disease</p>
            <div class="disease-name">{predicted_class}</div>
            <h2>Confidence: {confidence:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    if confidence < 40:
        st.warning("⚠️ Model confidence is low. The prediction may be unreliable.")
    elif confidence < 50:
        st.info("ℹ️ Confidence below 50% — prediction not saved to history.")

    st.write("")
    st.markdown('<div class="card"><h2>📊 Top Predictions</h2></div>', unsafe_allow_html=True)
    for rank, (label, score) in enumerate(top3, start=1):
        st.write(f"**{rank}. {label} — {score:.2f}%**")
        st.progress(min(score / 100, 1.0))

    with st.expander("🔥 AI Attention Heatmap (Grad-CAM)", expanded=False):
        heatmap = make_gradcam_heatmap(img_array, model)
        if heatmap is not None:
            overlay = overlay_heatmap(image, heatmap)
            if overlay is not None:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(image, caption="Original", use_container_width=True)
                with c2:
                    st.image(overlay, caption="Where AI looked", use_container_width=True)
            else:
                st.info("Install `opencv-python` to enable heatmap overlay.")
        else:
            st.info("Heatmap unavailable for this model architecture.")

    with st.expander("🤖 Model Information"):
        st.write(f"Model file: `{model_name}`")
        st.write("Input size: 224 × 224 × 3")
        st.write("Model type: Convolutional Neural Network (CNN)")
        st.write(f"Number of disease classes: {len(CLASS_NAMES)}")

    st.write("")
    if st.button("📄 Generate PDF Report", use_container_width=True):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                report_path = temp_file.name

            create_prediction_report(
                report_path,
                st.session_state.name,
                st.session_state.username,
                predicted_class,
                confidence,
                image_name,
                top3,
            )

            with open(report_path, "rb") as pdf_file:
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_file,
                    file_name="skin_disease_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Could not generate PDF: {e}")

    st.warning("⚕️ Medical Disclaimer: This application is for educational and research purposes only. It is not a substitute for professional medical diagnosis.")


# ============================================================
# PREDICTION PAGE
# ============================================================

def prediction_page():
    st.markdown('<div class="main-title">🔬 Skin Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a skin image for AI analysis</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📷 Upload Skin Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.markdown("""
        <div class="card">
            <h3>📷 Image Requirements</h3>
            <ul>
                <li>Use a clear skin image.</li>
                <li>JPG, JPEG, or PNG format.</li>
                <li>Avoid extremely dark or blurry images.</li>
                <li>Make sure the affected area is visible.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return

    image = Image.open(uploaded_file).convert("RGB")
    run_prediction(image, uploaded_file.name)


# ============================================================
# CAMERA PAGE
# ============================================================

def camera_page():
    st.markdown('<div class="main-title">📸 Live Camera Capture</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Take a photo with your device camera</div>', unsafe_allow_html=True)

    picture = st.camera_input("📸 Take a photo of the affected area")

    if picture is None:
        st.info("Position your skin in good lighting and press the capture button.")
        return

    image = Image.open(picture).convert("RGB")
    run_prediction(image, "camera_capture.jpg")


# ============================================================
# HISTORY PAGE
# ============================================================

def history_page():
    st.markdown('<div class="main-title">📋 Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your previous AI skin disease predictions</div>', unsafe_allow_html=True)

    records = get_prediction_history(st.session_state.username)

    if not records:
        st.info("No prediction history found.")
        return

    total = len(records)
    avg_conf = sum(r[2] for r in records) / total
    common = Counter(r[1] for r in records).most_common(1)[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Predictions", total)
    c2.metric("Avg Confidence", f"{avg_conf:.1f}%")
    c3.metric("Most Common", common[0])

    st.divider()

    for record in records:
        rid, disease, confidence, image_name, image_path, date = record
        col1, col2 = st.columns([1, 4])
        with col1:
            if image_path and os.path.exists(image_path):
                st.image(image_path, use_container_width=True)
            else:
                st.markdown("### 🖼️")
        with col2:
            st.markdown(f"### 🩺 {disease}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.write(f"**File:** {image_name}")
            st.write(f"**Date:** {date}")
            if st.button("🗑️ Delete", key=f"del_{rid}"):
                delete_prediction(rid, st.session_state.username)
                st.success("Deleted.")
                st.rerun()
        st.divider()


# ============================================================
# ANALYTICS PAGE
# ============================================================

def analytics_page():
    st.markdown('<div class="main-title">📊 Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Insights from your prediction history</div>', unsafe_allow_html=True)

    records = get_prediction_history(st.session_state.username)
    if not records:
        st.info("No data yet. Make some predictions first!")
        return

    df = pd.DataFrame(records, columns=["id", "disease", "confidence", "image", "path", "date"])
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.date

    try:
        import plotly.express as px

        c1, c2 = st.columns(2)
        with c1:
            counts = df["disease"].value_counts().reset_index()
            counts.columns = ["disease", "count"]
            fig = px.bar(counts, x="disease", y="count",
                         title="Disease Distribution",
                         color="count", color_continuous_scale="purples")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            daily = df.groupby("day")["confidence"].mean().reset_index()
            fig2 = px.line(daily, x="day", y="confidence",
                           title="Average Confidence Over Time", markers=True)
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.histogram(df, x="confidence", nbins=20,
                            title="Confidence Distribution",
                            color_discrete_sequence=["#764ba2"])
        st.plotly_chart(fig3, use_container_width=True)

    except ImportError:
        st.warning("Install plotly to see charts: `pip install plotly`")
        st.dataframe(df[["disease", "confidence", "date"]])

    csv = df[["disease", "confidence", "image", "date"]].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download History as CSV", csv,
                       file_name="prediction_history.csv", mime="text/csv")


# ============================================================
# DISEASE INFORMATION
# ============================================================

def disease_info_page():
    st.markdown('<div class="main-title">📚 Disease Information</div>', unsafe_allow_html=True)

    disease_info = {
        "Acne": "Acne causes pimples, blackheads, and whiteheads. Treat with gentle cleansers and topical retinoids.",
        "Eczema": "Eczema (Atopic Dermatitis) causes dry, itchy, inflamed skin. Moisturizers and steroids help.",
        "Psoriasis": "Psoriasis causes red, scaly patches. Manage with topical steroids, phototherapy, or biologics.",
        "Rosacea": "Rosacea causes facial redness and flushing. Avoid triggers, use sunscreen, see a dermatologist.",
        "Vitiligo": "Vitiligo causes loss of skin pigment. Treat with phototherapy or topical creams.",
        "Tinea / Ringworm": "Fungal infection with ring-shaped rash. Treat with antifungal creams.",
        "Impetigo": "Bacterial infection with sores. Treat with antibiotics.",
        "Seborrheic Dermatitis": "Flaky, scaly patches on oily areas. Use medicated shampoos.",
        "Melanoma": "Serious skin cancer. Immediate medical attention required.",
        "Actinic Keratosis": "Precancerous skin growth from sun exposure. Treat early.",
        "Inflammatory": "General class for inflammatory skin conditions like Eczema, Psoriasis, and Dermatitis. Treatment depends on the specific condition.",
        "Pigmentation": "General class for pigmented lesions like Nevi (moles), Benign Keratosis, and Melanocytic conditions.",
        "Tumors": "General class for skin tumors including Melanoma and Basal Cell Carcinoma. Immediate medical attention required.",
    }

    st.markdown(f'<div class="subtitle">Detected classes: {len(CLASS_NAMES)}</div>', unsafe_allow_html=True)

    cols = st.columns(2)
    for idx, disease in enumerate(CLASS_NAMES):
        with cols[idx % 2]:
            with st.expander(f"🩺 {disease}"):
                st.write(disease_info.get(disease, f"Information for {disease} is being added."))
                st.info("⚠️ Educational purposes only. Consult a doctor.")


# ============================================================
# MODEL INFORMATION
# ============================================================

def model_info_page():
    st.markdown('<div class="main-title">🤖 Model Information</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h2>🧠 CNN Skin Disease Classification</h2>
        <p>The application uses a Convolutional Neural Network for image-based skin disease classification.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", "CNN (EfficientNetB0)")
    with col2:
        st.metric("Input Size", "224 × 224 × 3")
    with col3:
        st.metric("Disease Classes", len(CLASS_NAMES))

    st.write("")
    st.subheader("📊 Disease Classes")
    cols = st.columns(4)
    for idx, disease in enumerate(CLASS_NAMES):
        with cols[idx % 4]:
            st.markdown(f"• {disease}")


# ============================================================
# PROFILE PAGE
# ============================================================

def profile_page():
    st.markdown('<div class="main-title">👤 My Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Manage your personal account information</div>', unsafe_allow_html=True)

    profile = get_user_profile(st.session_state.username)
    if not profile:
        st.error("Unable to load profile.")
        return

    current_username, current_name, email, phone, created_at = profile

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div style="font-size:70px;">👤</div>
            <h2>{current_name}</h2>
            <p>@{current_username}</p>
            <p>🗓️ Member since<br>{created_at}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("✏️ Edit Profile")

        new_username = st.text_input("Username", value=current_username)
        new_name = st.text_input("Full Name", value=current_name)
        st.text_input("Email", value=email, disabled=True)
        st.text_input("Phone Number", value=phone, disabled=True)

        if st.button("💾 Save Changes", use_container_width=True):
            success, message = update_profile(current_username, new_username.strip(), new_name.strip())
           