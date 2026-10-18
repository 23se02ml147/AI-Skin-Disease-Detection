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
    create_database, register_user, login_user, save_prediction,
    get_prediction_history, get_prediction_count, get_user_profile,
    update_profile, change_password, delete_prediction, get_activity_log
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
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); min-height: 100vh; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); padding-top: 20px; }
    section[data-testid="stSidebar"] * { color: #ffffff !important; }
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.1); color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 12px;
        padding: 10px 20px; font-size: 16px; font-weight: 500; width: 100%;
        text-align: left; transition: all 0.3s ease; margin-bottom: 5px;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.25); border-color: #4fc3f7; transform: translateX(5px);
    }
    .sidebar-user { background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 20px; }
    .sidebar-user h3 { margin-bottom: 5px; font-size: 18px; }
    .sidebar-user p { font-size: 14px; }
    .sidebar-user .avatar { font-size: 50px; margin-bottom: 10px; }
    .main-title {
        font-size: 42px; font-weight: 800; text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; padding: 20px 0;
    }
    .subtitle { text-align: center; font-size: 18px; color: #555; margin-bottom: 30px; }
    .card { background: rgba(255, 255, 255, 0.95); padding: 25px; border-radius: 20px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08); margin-bottom: 20px; }
    .card h2, .card h3 { color: #1a1a2e; }
    .prediction-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 25px; text-align: center; color: white; }
    .prediction-card h2 { color: white !important; }
    .prediction-card p { color: rgba(255, 255, 255, 0.9) !important; }
    .disease-name { font-size: 36px; font-weight: 800; color: #ffffff; text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2); }
    .metric-card { background: white; padding: 20px; border-radius: 18px; text-align: center; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06); border: 1px solid rgba(0, 0, 0, 0.05); }
    .metric-card h3 { color: #1a1a2e; font-size: 28px; margin: 10px 0; }
    .metric-card p { color: #666; font-size: 14px; }
    .history-card { background: white; padding: 18px 22px; border-radius: 15px; margin-bottom: 12px; box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06); border-left: 4px solid #667eea; }
    .history-card h3 { color: #1a1a2e; margin-bottom: 8px; }
    .history-card p { color: #666; margin: 4px 0; }
    .stButton > button { border-radius: 12px; font-weight: 600; border: none; padding: 10px 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; transition: all 0.3s ease; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4); }
    .stFileUploader { background: rgba(255, 255, 255, 0.95); border-radius: 18px; padding: 20px; border: 2px dashed #667eea; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATABASE & CLASSES
# ============================================================
create_database()

def load_classes():
    if os.path.exists("classes.json"):
        with open("classes.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return ["inflammatory", "pigmentation", "tumors"]

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

def valid_email(email): return re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email) is not None
def valid_phone(phone): return phone.isdigit() and len(phone) == 10

# ============================================================
# LOAD MODEL (Split Base + Head)
# ============================================================
@st.cache_resource
def load_skin_model():
    head_path = "skin_model_head.keras"
    
    if not os.path.exists(head_path):
        st.error(f"❌ Model file '{head_path}' not found. Please run train.py first.")
        st.stop()
        
    try:
        # 1. Load Base Model
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", pooling="avg", input_shape=(224, 224, 3)
        )
        base_model.trainable = False
        
        # 2. Load Trained Head
        head_model = tf.keras.models.load_model(head_path)
        
        # 3. Combine them
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        x = base_model(x, training=False)
        outputs = head_model(x)
        
        final_model = tf.keras.Model(inputs, outputs)
        return final_model, "EfficientNetB0 + Custom Head"
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# ============================================================
# GRAD-CAM HELPERS
# ============================================================
def make_gradcam_heatmap(img_array, model):
    try:
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if hasattr(layer, "layers"):
                for sub in reversed(layer.layers):
                    if isinstance(sub, tf.keras.layers.Conv2D):
                        last_conv_layer_name = sub.name
                        break
                if last_conv_layer_name: break
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

        if not last_conv_layer_name: return None

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
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
        img = np.array(image.resize((224, 224)))
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        return Image.fromarray(superimposed)
    except:
        return None

# ============================================================
# AUTHENTICATION UI
# ============================================================
def login_page():
    st.markdown('<div class="main-title">🩺 AI Skin Disease Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-Powered Skin Disease Analysis System</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("🔑 Login", use_container_width=True):
            user = login_user(username.strip(), password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = user[1]
                st.session_state.name = user[2]
                st.session_state.page = "Dashboard"
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.divider()
        if st.button("📝 New User? Create Account", use_container_width=True):
            st.session_state.page = "Sign Up"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

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
            if not username or not name or not valid_email(email) or not valid_phone(phone) or len(password) < 6 or password != confirm_password:
                st.error("Please check all fields. Password must be 6+ chars, phone 10 digits.")
            else:
                success, msg = register_user(username.strip(), name.strip(), email.strip(), phone.strip(), password)
                if success:
                    st.success("Account created! Please login.")
                    st.session_state.page = "Login"
                    st.rerun()
                else:
                    st.error(msg)
        st.divider()
        if st.button("← Back to Login", use_container_width=True):
            st.session_state.page = "Login"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR & DASHBOARD
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
    
    nav = {"🏠 Dashboard": "Dashboard", "🔬 Prediction": "Prediction", "📸 Live Camera": "Camera",
           "📋 History": "History", "📊 Analytics": "Analytics", "📚 Disease Info": "Disease Info",
           "🤖 Model Info": "Model Info", "👤 Profile": "Profile", "⚙️ Settings": "Settings"}
           
    for label, page in nav.items():
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

def dashboard_page():
    st.markdown('<div class="main-title">🏠 Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">Welcome back, {st.session_state.name} 👋</div>', unsafe_allow_html=True)
    
    count = get_prediction_count(st.session_state.username)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric-card'><h2>🔬</h2><h3>{count}</h3><p>Total Predictions</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><h2>🧬</h2><h3>{len(CLASS_NAMES)}</h3><p>Disease Classes</p></div>", unsafe_allow_html=True)
    c3.markdown("<div class='metric-card'><h2>🤖</h2><h3>CNN</h3><p>AI Model</p></div>", unsafe_allow_html=True)
    
    st.write("")
    if st.button("🔬 Start New Prediction", use_container_width=True):
        st.session_state.page = "Prediction"
        st.rerun()
        
    logs = get_activity_log(st.session_state.username, limit=5)
    if logs:
        st.subheader("🕐 Recent Activity")
        for action, ts in logs:
            st.write(f"- **{action}** — {ts}")

# ============================================================
# PREDICTION
# ============================================================
def run_prediction(image, image_name):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📷 Uploaded Image")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    img_resized = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized).astype("float32"), axis=0)
    
    model, model_name = load_skin_model()
    
    with st.spinner("Analyzing..."):
        preds = model.predict(img_array, verbose=0)[0]
        
    top_idx = np.argsort(preds)[::-1][:3]
    top3 = [(CLASS_NAMES[i], float(preds[i]*100)) for i in top_idx if i < len(CLASS_NAMES)]
    
    if not top3: return
    
    disease, conf = top3[0]
    
    saved_path = os.path.join("uploads", f"{st.session_state.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    image.save(saved_path)
    
    prediction_key = (st.session_state.username, image_name, disease, round(conf, 2))
    if conf >= 50 and st.session_state.last_prediction_key != prediction_key:
        save_prediction(st.session_state.username, disease, conf, image_name, saved_path)
        st.session_state.last_prediction_key = prediction_key
    
    with col2:
        st.markdown(f"<div class='prediction-card'><h2>🩺 Prediction</h2><p>Predicted Disease</p><div class='disease-name'>{disease}</div><h2>Confidence: {conf:.2f}%</h2></div>", unsafe_allow_html=True)
        
    if conf < 50:
        st.warning("⚠️ Confidence is below 50%. Prediction may be unreliable.")
        
    st.write("")
    st.markdown('<div class="card"><h2>📊 Top Predictions</h2></div>', unsafe_allow_html=True)
    for rank, (label, score) in enumerate(top3, 1):
        st.write(f"**{rank}. {label} — {score:.2f}%**")
        st.progress(min(score/100, 1.0))
        
    with st.expander("🔥 Grad-CAM Heatmap"):
        hm = make_gradcam_heatmap(img_array, model)
        if hm is not None:
            ov = overlay_heatmap(image, hm)
            if ov:
                c1, c2 = st.columns(2)
                c1.image(image, caption="Original")
                c2.image(ov, caption="AI Focus Area")
            else:
                st.info("Install opencv-python for heatmap.")
        else:
            st.info("Heatmap unavailable.")
            
    with st.expander("🤖 Model Information"):
        st.write(f"Model file: `{model_name}`")
        st.write("Input size: 224 × 224 × 3")
        st.write(f"Number of disease classes: {len(CLASS_NAMES)}")
            
    if st.button("📄 Generate PDF Report", use_container_width=True):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                path = f.name
            create_prediction_report(path, st.session_state.name, st.session_state.username, disease, conf, image_name, top3)
            with open(path, "rb") as f:
                st.download_button("⬇️ Download PDF", data=f, file_name="report.pdf", mime="application/pdf", use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate PDF: {e}")
            
    st.warning("⚕️ Medical Disclaimer: Educational purposes only.")

def prediction_page():
    st.markdown('<div class="main-title">🔬 Skin Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a skin image for AI analysis</div>', unsafe_allow_html=True)
    file = st.file_uploader("📷 Upload Skin Image", type=["jpg", "jpeg", "png"])
    if file:
        image = Image.open(file).convert("RGB")
        run_prediction(image, file.name)
    else:
        st.markdown("""
        <div class="card">
            <h3>📷 Image Requirements</h3>
            <ul><li>Use a clear skin image.</li><li>JPG, JPEG, or PNG format.</li><li>Avoid blurry images.</li></ul>
        </div>
        """, unsafe_allow_html=True)

def camera_page():
    st.markdown('<div class="main-title">📸 Live Camera Capture</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Take a photo with your device camera</div>', unsafe_allow_html=True)
    pic = st.camera_input("Take a photo")
    if pic:
        image = Image.open(pic).convert("RGB")
        run_prediction(image, "camera_capture.jpg")

# ============================================================
# HISTORY & ANALYTICS
# ============================================================
def history_page():
    st.markdown('<div class="main-title">📋 Prediction History</div>', unsafe_allow_html=True)
    records = get_prediction_history(st.session_state.username)
    if not records:
        st.info("No history found.")
        return
    
    total = len(records)
    avg_conf = sum(r[2] for r in records) / total
    common = Counter(r[1] for r in records).most_common(1)[0][0]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Predictions", total)
    c2.metric("Avg Confidence", f"{avg_conf:.1f}%")
    c3.metric("Most Common", common)
    
    st.divider()
    for r in records:
        rid, disease, conf, name, path, date = r
        c1, c2 = st.columns([1, 4])
        with c1:
            if path and os.path.exists(path): st.image(path, use_container_width=True)
            else: st.markdown("### 🖼️")
        with c2:
            st.markdown(f"### 🩺 {disease}")
            st.write(f"**Confidence:** {conf:.2f}%")
            st.write(f"**File:** {name} | **Date:** {date}")
            if st.button("🗑️ Delete", key=f"del_{rid}"):
                delete_prediction(rid, st.session_state.username)
                st.success("Deleted.")
                st.rerun()
        st.divider()

def analytics_page():
    st.markdown('<div class="main-title">📊 Analytics</div>', unsafe_allow_html=True)
    records = get_prediction_history(st.session_state.username)
    if not records:
        st.info("No data yet.")
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
            st.plotly_chart(px.bar(counts, x="disease", y="count", title="Disease Distribution", color="count"), use_container_width=True)
        with c2:
            daily = df.groupby("day")["confidence"].mean().reset_index()
            st.plotly_chart(px.line(daily, x="day", y="confidence", title="Confidence Over Time", markers=True), use_container_width=True)
    except ImportError:
        st.warning("Install plotly for charts: `pip install plotly`")
        st.dataframe(df[["disease", "confidence", "date"]])
        
    csv = df[["disease", "confidence", "image", "date"]].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, file_name="history.csv", mime="text/csv")

# ============================================================
# INFO & SETTINGS
# ============================================================
def disease_info_page():
    st.markdown('<div class="main-title">📚 Disease Information</div>', unsafe_allow_html=True)
    info = {
        "Inflammatory": "General class for inflammatory skin conditions like Eczema, Psoriasis, and Dermatitis.",
        "Pigmentation": "General class for pigmented lesions like Nevi (moles) and Benign Keratosis.",
        "Tumors": "General class for skin tumors including Melanoma and Basal Cell Carcinoma. Requires medical attention."
    }
    cols = st.columns(2)
    for idx, disease in enumerate(CLASS_NAMES):
        with cols[idx % 2]:
            with st.expander(f"🩺 {disease}"):
                st.write(info.get(disease, f"Information for {disease} is being added."))
                st.info("⚠️ Educational purposes only.")

def model_info_page():
    st.markdown('<div class="main-title">🤖 Model Information</div>', unsafe_allow_html=True)
    st.markdown("<div class='card'><h2>🧠 CNN Skin Disease Classification</h2><p>The application uses a Convolutional Neural Network (EfficientNetB0) for image-based classification.</p></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Model Type", "EfficientNetB0")
    c2.metric("Input Size", "224 × 224 × 3")
    c3.metric("Disease Classes", len(CLASS_NAMES))

def profile_page():
    st.markdown('<div class="main-title">👤 My Profile</div>', unsafe_allow_html=True)
    profile = get_user_profile(st.session_state.username)
    if not profile: return
    curr_u, curr_n, email, phone, created = profile
    
    c1, c2 = st.columns([1, 2])
    c1.markdown(f"<div class='card' style='text-align:center;'><div style='font-size:70px;'>👤</div><h2>{curr_n}</h2><p>@{curr_u}</p><p>🗓️ Member since<br>{created}</p></div>", unsafe_allow_html=True)
    
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("✏️ Edit Profile")
        new_u = st.text_input("Username", value=curr_u)
        new_n = st.text_input("Full Name", value=curr_n)
        st.text_input("Email", value=email, disabled=True)
        if st.button("💾 Save Changes"):
            s, m = update_profile(curr_u, new_u.strip(), new_n.strip())
            if s:
                st.session_state.username = new_u.strip()
                st.session_state.name = new_n.strip()
                st.success(m)
                st.rerun()
            else: st.error(m)
        st.markdown('</div>', unsafe_allow_html=True)

def settings_page():
    st.markdown('<div class="main-title">⚙️ Settings</div>', unsafe_allow_html=True)
    st.subheader("🔐 Change Password")
    old = st.text_input("Current Password", type="password")
    new = st.text_input("New Password", type="password")
    conf = st.text_input("Confirm New Password", type="password")
    if st.button("Change Password"):
        if not old: st.error("Enter current password.")
        elif len(new) < 6: st.error("Must be 6+ chars.")
        elif new != conf: st.error("Passwords do not match.")
        else:
            s, m = change_password(st.session_state.username, old, new)
            if s: st.success(m)
            else: st.error(m)

# ============================================================
# MAIN ROUTER
# ============================================================
if not st.session_state.logged_in:
    if st.session_state.page == "Sign Up": signup_page()
    else: login_page()
else:
    sidebar()
    pages = {
        "Dashboard": dashboard_page, "Prediction": prediction_page, "Camera": camera_page,
        "History": history_page, "Analytics": analytics_page, "Disease Info": disease_info_page,
        "Model Info": model_info_page, "Profile": profile_page, "Settings": settings_page
    }
    pages.get(st.session_state.page, dashboard_page)()