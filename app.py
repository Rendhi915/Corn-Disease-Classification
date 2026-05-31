import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # max 16MB

# ============================================================
# KONFIGURASI
# ============================================================
MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'best_corn_model.pt')
CLASS_NAMES = ['Bercak Daun', 'Daun Sehat', 'Hawar Daun', 'Karat Daun']
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_INFO = {
    'Bercak Daun': {
        'en': 'Gray Leaf Spot',
        'desc': 'Penyakit yang disebabkan oleh jamur Cercospora zeae-maydis. Ditandai dengan bercak abu-abu memanjang pada daun.',
        'color': '#f59e0b',
        'icon': '🍂'
    },
    'Daun Sehat': {
        'en': 'Healthy',
        'desc': 'Daun jagung dalam kondisi sehat, tidak terinfeksi penyakit apapun.',
        'color': '#10b981',
        'icon': '🌿'
    },
    'Hawar Daun': {
        'en': 'Northern Leaf Blight',
        'desc': 'Penyakit yang disebabkan oleh jamur Exserohilum turcicum. Ditandai dengan lesio besar berbentuk cerutu.',
        'color': '#ef4444',
        'icon': '🔥'
    },
    'Karat Daun': {
        'en': 'Common Rust',
        'desc': 'Penyakit yang disebabkan oleh jamur Puccinia sorghi. Ditandai dengan pustul berwarna oranye kecokelatan.',
        'color': '#8b5cf6',
        'icon': '🟠'
    }
}

# ============================================================
# LOAD MODEL
# ============================================================
def load_model():
    model = models.mobilenet_v3_small()
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()
print(f"Model loaded on {DEVICE}")

# ============================================================
# TRANSFORM
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ============================================================
# GRAD-CAM
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        input_tensor = input_tensor.to(DEVICE)
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        output = self.model(input_tensor)
        probs  = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        pred_prob  = probs[0, pred_class].item()
        all_probs  = probs[0].tolist()

        if target_class is None:
            target_class = pred_class

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()

        return cam, pred_class, pred_prob, all_probs

gradcam = GradCAM(model, model.features[-1])

# ============================================================
# VALIDASI 3 LAPIS — PASTIKAN GAMBAR ADALAH DAUN JAGUNG
# ============================================================

# Threshold confidence minimum
CONFIDENCE_THRESHOLD = 88.0
MARGIN_THRESHOLD = 15.0  # 최소 top-1 vs top-2 gap (percent)
ENTROPY_THRESHOLD = 1.00  # softmax entropy threshold

def check_green_dominant(img_np):
    """
    Lapis 1: Cek apakah gambar didominasi warna daun.
    Return: (passed, metrics)
    """
    img_uint8 = (img_np * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)

    # Range warna daun (lebih toleran untuk daun sakit/kekuningan)
    mask_green = cv2.inRange(hsv, (30, 25, 25), (95, 255, 255))
    mask_yellow = cv2.inRange(hsv, (15, 20, 20), (40, 255, 255))
    mask_brown = cv2.inRange(hsv, (5, 30, 20), (25, 255, 220))

    total_pixels = img_np.shape[0] * img_np.shape[1]
    green_ratio = (mask_green > 0).sum() / total_pixels
    yellow_ratio = (mask_yellow > 0).sum() / total_pixels
    brown_ratio = (mask_brown > 0).sum() / total_pixels

    leaf_color_ratio = green_ratio + yellow_ratio * 0.7 + brown_ratio * 0.5

    leaf_mask = (mask_green > 0) | (mask_yellow > 0) | (mask_brown > 0)
    if leaf_mask.any():
        hue_vals = hsv[:, :, 0][leaf_mask]
        sat_vals = hsv[:, :, 1][leaf_mask]
        hue_std = float(np.std(hue_vals))
        sat_mean = float(np.mean(sat_vals))
    else:
        hue_std = 0.0
        sat_mean = 0.0

    dense_leaf = leaf_color_ratio >= 0.50
    min_ratio = 0.18
    sat_threshold = 30 if dense_leaf else 35
    hue_threshold = 2.0

    if dense_leaf:
        passed = sat_mean >= sat_threshold
    else:
        passed = (
            (leaf_color_ratio >= min_ratio)
            and (sat_mean >= sat_threshold)
            and (hue_std >= hue_threshold)
        )

    return (
        passed,
        {
            'leaf_ratio': round(leaf_color_ratio * 100, 1),
            'sat_mean': round(sat_mean, 1),
            'hue_std': round(hue_std, 1),
            'min_ratio': round(min_ratio * 100, 1),
            'sat_threshold': sat_threshold,
            'hue_threshold': hue_threshold,
            'dense_leaf': dense_leaf
        }
    )


def check_leaf_texture(img_np):
    """
    Lapis 2: Cek apakah gambar punya tekstur daun.
    Return: (passed, edge_score, variance, line_count)
    """
    img_uint8 = (img_np * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 20, 80)
    edge_ratio = (edges > 0).sum() / edges.size

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25,
                             minLineLength=15, maxLineGap=12)
    line_count = 0 if lines is None else len(lines)
    has_lines = line_count >= 5

    gray_float = gray.astype(float)
    variance = float(np.var(gray_float))

    passed = (edge_ratio >= 0.03) and (variance > 180) and (has_lines or edge_ratio > 0.08)
    edge_score = round(edge_ratio * 100, 1)
    return passed, edge_score, round(variance, 1), line_count


def check_aspect_ratio(img):
    """
    Lapis 3 tambahan: Cek apakah gambar tidak terlalu ekstrem aspect ratio-nya.
    Foto daun jagung yang valid biasanya tidak terlalu memanjang.
    """
    w, h = img.size
    ratio = max(w, h) / min(w, h)
    return ratio <= 5.0  # tolak kalau lebih dari 5:1


def validate_corn_leaf(img, img_np):
    """
    Jalankan semua validasi dan kembalikan hasil.
    Return: (is_valid, reason, details)
    """
    # Cek aspect ratio
    if not check_aspect_ratio(img):
        return False, "Proporsi gambar tidak sesuai untuk foto daun jagung.", {}

    # Lapis 1: Warna hijau
    green_passed, green_metrics = check_green_dominant(img_np)
    leaf_ratio = green_metrics['leaf_ratio']
    sat_mean = green_metrics['sat_mean']
    hue_std = green_metrics['hue_std']

    if not green_passed:
        min_ratio = green_metrics['min_ratio']
        sat_threshold = green_metrics['sat_threshold']
        hue_threshold = green_metrics['hue_threshold']
        dense_leaf = green_metrics['dense_leaf']

        reasons = []
        if leaf_ratio < min_ratio:
            reasons.append("proporsi warna daun rendah")
        if sat_mean < sat_threshold:
            reasons.append("saturasi warna daun rendah")
        if (not dense_leaf) and (hue_std < hue_threshold):
            reasons.append("warna terlalu seragam")

        reason_text = " / ".join(reasons) if reasons else "validasi warna daun gagal"
        return False, (
            f"Gambar tidak terdeteksi sebagai daun jagung. "
            f"{reason_text} "
            f"(leaf {leaf_ratio}%, sat {sat_mean}, hue std {hue_std}). "
            f"Pastikan foto menampilkan daun jagung dengan jelas."
        ), {
            'green_ratio': leaf_ratio,
            'sat_mean': sat_mean,
            'hue_std': hue_std,
            'min_ratio': min_ratio,
            'sat_threshold': sat_threshold,
            'hue_threshold': hue_threshold
        }

    # Lapis 2: Tekstur daun
    texture_passed, edge_score, variance, line_count = check_leaf_texture(img_np)
    if not texture_passed:
        return False, (
            f"Tekstur gambar tidak sesuai dengan daun jagung. "
            f"Pastikan foto fokus pada permukaan daun jagung."
        ), {
            'green_ratio': leaf_ratio,
            'sat_mean': sat_mean,
            'hue_std': hue_std,
            'edge_score': edge_score,
            'variance': variance,
            'line_count': line_count
        }

    return True, "OK", {
        'green_ratio': leaf_ratio,
        'sat_mean': sat_mean,
        'hue_std': hue_std,
        'edge_score': edge_score,
        'variance': variance,
        'line_count': line_count
    }


def prepare_validation_image(img):
    """
    Downscale untuk validasi agar stabil di berbagai resolusi.
    """
    img_val = img.copy()
    img_val.thumbnail((512, 512))
    return np.array(img_val) / 255.0


# ============================================================
# HELPER — IMAGE TO BASE64
# ============================================================
def img_to_base64(img_np):
    img_uint8 = (img_np * 255).astype(np.uint8)
    img_pil   = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diupload'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    allowed = {'png', 'jpg', 'jpeg', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': 'Format file tidak didukung. Gunakan PNG/JPG/JPEG/WEBP'}), 400

    try:
        # Load gambar
        img = Image.open(file.stream).convert('RGB')
        img_resized = img.resize((224, 224))
        img_np = np.array(img_resized) / 255.0
        img_val_np = prepare_validation_image(img)

        # ============================================================
        # VALIDASI 3 LAPIS SEBELUM PREDIKSI
        # ============================================================
        is_valid, reason, val_details = validate_corn_leaf(img, img_val_np)

        if not is_valid:
            original_b64 = img_to_base64(img_np)
            return jsonify({
                'success': True,
                'is_corn_leaf': False,
                'reason': reason,
                'images': {'original': original_b64}
            })

        # ============================================================
        # PREDIKSI MODEL
        # ============================================================
        img_tensor = transform(img).unsqueeze(0)
        cam, pred_idx, pred_prob, all_probs = gradcam.generate(img_tensor)

        pred_label  = CLASS_NAMES[pred_idx]
        confidence  = pred_prob * 100

        # ============================================================
        # GENERATE GRAD-CAM & RESPONSE
        # ============================================================
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = np.clip(0.5 * img_np + 0.5 * heatmap, 0, 1)

        original_b64 = img_to_base64(img_np)
        heatmap_b64  = img_to_base64(heatmap)
        overlay_b64  = img_to_base64(overlay)

        class_probs = [
            {
                'name': CLASS_NAMES[i],
                'prob': round(all_probs[i] * 100, 2),
                'color': CLASS_INFO[CLASS_NAMES[i]]['color']
            }
            for i in range(len(CLASS_NAMES))
        ]
        class_probs.sort(key=lambda x: x['prob'], reverse=True)

        return jsonify({
            'success': True,
            'is_corn_leaf': True,
            'prediction': pred_label,
            'confidence': round(confidence, 2),
            'description': CLASS_INFO[pred_label]['desc'],
            'color': CLASS_INFO[pred_label]['color'],
            'icon': CLASS_INFO[pred_label]['icon'],
            'en_name': CLASS_INFO[pred_label]['en'],
            'class_probs': class_probs,
            'images': {
                'original': original_b64,
                'heatmap': heatmap_b64,
                'overlay': overlay_b64
            }
        })

    except Exception as e:
        return jsonify({'error': f'Gagal memproses gambar: {str(e)}'}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("  Corn Leaf Disease Detector")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
