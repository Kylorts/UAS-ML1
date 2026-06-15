"""
Demo Klasifikasi Sampah — Streamlit (folder mandiri)
Model: ShuffleNet V2 (PyTorch) & ResNet50 + Random Forest (TF + sklearn)
Fitur: prediksi + confidence, akurasi model, Grad-CAM / feature-importance CAM.
"""

import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Konfigurasi (semua path relatif terhadap folder app ini)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 224
CLASSES = ["Sampah Kardus", "Sampah Kertas", "Sampah Plastik"]

PATHS = {
    "shufflenet": os.path.join(BASE_DIR, "models", "shufflenet_final.pt"),
    "resnet_fe":  os.path.join(BASE_DIR, "models", "resnet_fe.keras"),
    "rf":         os.path.join(BASE_DIR, "models", "rf_baseline.joblib"),
    "dataset":    os.path.join(BASE_DIR, "dataset", "standardized_dataset"),
    "cam_stats":  os.path.join(BASE_DIR, "cam_stats.npz"),
}

# Akurasi test (dari evaluasi 80/20, test set 52 gambar)
TEST_ACC = {
    "ShuffleNet V2": 0.9231,
    "ResNet50 + Random Forest": 0.8654,
}

st.set_page_config(page_title="Klasifikasi Sampah", page_icon="♻️", layout="wide")


# ---------------------------------------------------------------------------
# Util gambar
# ---------------------------------------------------------------------------
def load_images(pil_img):
    """Kembalikan (gambar resolusi penuh untuk tampilan, gambar 224 untuk model)."""
    full = np.array(pil_img.convert("RGB"))
    small = cv2.resize(full, (IMG_SIZE, IMG_SIZE))
    return full, small


def overlay_cam(raw_img, cam):
    # resize heatmap ke ukuran gambar tampilan (resolusi penuh) -> tidak pecah
    cam = cv2.resize(cam, (raw_img.shape[1], raw_img.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(raw_img, 0.6, heat, 0.4, 0)


# ---------------------------------------------------------------------------
# ShuffleNet (PyTorch) + Grad-CAM
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_shufflenet():
    import torch
    import torch.nn as nn
    import torchvision.models as models

    model = models.shufflenet_v2_x1_0(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    state = torch.load(PATHS["shufflenet"], map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def shufflenet_predict_cam(model, img_rgb):
    import torch
    import torch.nn.functional as F
    from torchvision import transforms as T

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    x = normalize(x).unsqueeze(0)
    x.requires_grad_(True)

    acts, grads = {}, {}
    h1 = model.conv5.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o))
    h2 = model.conv5.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0]))

    out = model(x)
    probs = F.softmax(out, dim=1)[0].detach().numpy()
    pred = int(out.argmax(1).item())

    model.zero_grad()
    out[0, pred].backward()
    w = grads["v"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["v"]).sum(1)).squeeze().detach().numpy()
    cam = cam / (cam.max() + 1e-8)

    h1.remove(); h2.remove()
    return pred, probs, cam


# ---------------------------------------------------------------------------
# ResNet50 + RF (TF + sklearn) + feature-importance CAM
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_resnet_rf():
    import joblib
    from tensorflow.keras.models import load_model, Model

    fe = load_model(PATHS["resnet_fe"], compile=False)
    rf = joblib.load(PATHS["rf"])
    cam_model = Model(fe.input, fe.get_layer("conv5_block3_out").output)
    return fe, rf, cam_model


@st.cache_resource(show_spinner=False)
def get_cam_stats(_fe):
    """class_mean & overall_mean dari fitur dataset (untuk bobot CAM per-kelas)."""
    if os.path.exists(PATHS["cam_stats"]):
        d = np.load(PATHS["cam_stats"])
        return d["class_mean"], d["overall_mean"]

    from tensorflow.keras.applications.resnet50 import preprocess_input
    feats, labels = [], []
    for ci, cls in enumerate(CLASSES):
        cdir = os.path.join(PATHS["dataset"], cls)
        if not os.path.isdir(cdir):
            continue
        imgs = []
        for fn in os.listdir(cdir):
            img = cv2.imread(os.path.join(cdir, fn))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        if not imgs:
            continue
        X = preprocess_input(np.array(imgs, dtype=np.float32))
        f = _fe.predict(X, batch_size=32, verbose=0)
        feats.append(f); labels += [ci] * len(f)

    feats = np.concatenate(feats); labels = np.array(labels)
    class_mean = np.array([feats[labels == c].mean(0) for c in range(len(CLASSES))])
    overall_mean = feats.mean(0)
    np.savez(PATHS["cam_stats"], class_mean=class_mean, overall_mean=overall_mean)
    return class_mean, overall_mean


def resnet_rf_predict_cam(fe, rf, cam_model, class_mean, overall_mean, img_rgb):
    from tensorflow.keras.applications.resnet50 import preprocess_input

    x = preprocess_input(np.expand_dims(img_rgb.astype(np.float32), 0))
    feat = fe.predict(x, verbose=0)
    probs = rf.predict_proba(feat)[0]
    pred = int(np.argmax(probs))

    w = rf.feature_importances_ * (class_mean[pred] - overall_mean)
    w = np.maximum(w, 0)
    fmap = cam_model.predict(x, verbose=0)[0]
    cam = np.dot(fmap, w)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    return pred, probs, cam


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("Klasifikasi Sampah")
st.caption("Kardus · Kertas · Plastik dengan visualisasi Grad-CAM / CAM")

with st.sidebar:
    st.header("Pengaturan")
    model_name = st.selectbox("Pilih model", ["ShuffleNet V2", "ResNet50 + Random Forest"])
    st.metric("Akurasi test model", f"{TEST_ACC[model_name]*100:.2f}%")
    st.caption("Akurasi pada test set (80/20, 52 gambar).")
    uploaded = st.file_uploader("Upload gambar sampah", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Silakan upload gambar di panel kiri untuk memulai prediksi.")
    st.stop()

img_full, img_rgb = load_images(Image.open(uploaded))

with st.spinner("Memuat model & memprediksi (load pertama bisa agak lama)..."):
    if model_name == "ShuffleNet V2":
        model = load_shufflenet()
        pred, probs, cam = shufflenet_predict_cam(model, img_rgb)
        cam_label = "Grad-CAM"
    else:
        fe, rf, cam_model = load_resnet_rf()
        class_mean, overall_mean = get_cam_stats(fe)
        pred, probs, cam = resnet_rf_predict_cam(
            fe, rf, cam_model, class_mean, overall_mean, img_rgb)
        cam_label = "CAM (feature importance)"

heat = overlay_cam(img_full, cam)

# --- Hasil prediksi (di atas) ---
st.subheader(f"Prediksi: {CLASSES[pred]}  ·  Keyakinan: {probs[pred]*100:.1f}%")
st.caption("Probabilitas per kelas:")
for i, cls in enumerate(CLASSES):
    st.write(cls)
    st.progress(float(probs[i]), text=f"{probs[i]*100:.1f}%")

st.markdown("---")

# --- Gambar input + Grad-CAM/CAM (di bawah) ---
c1, c2 = st.columns(2)
with c1:
    st.subheader("Gambar Input")
    st.image(img_full, use_container_width=True)
with c2:
    st.subheader(cam_label)
    st.image(heat, use_container_width=True)

st.caption(f"Model: {model_name}  ·  {cam_label} menyorot area yang menjadi dasar keputusan model.")
