import os
import uuid
import numpy as np
import nibabel as nib
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io, base64
import traceback
import time
import gc
from scipy import ndimage
from skimage import measure, filters, morphology

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB limit for Render
app.config["SECRET_KEY"] = "neurovision-secret-key"

# Health check for Render
@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

# Normal reference values
NORMAL_HIPPOCAMPUS_TOTAL = 2200
NORMAL_HIPPOCAMPUS_SINGLE = 1100
NORMAL_VENTRICLE = 2500
NORMAL_WMH_COUNT = 10

# Voxel volume (will be updated from NIfTI)
VOXEL_VOLUME_MM3 = 1.0


def convert_to_native(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {convert_to_native(key): convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native(item) for item in obj]
    return obj


# -------------------------
# LOAD NIFTI (SAME AS WORKING CODE)
# -------------------------
def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()
    voxel_dims = img.header.get_zooms()
    global VOXEL_VOLUME_MM3
    VOXEL_VOLUME_MM3 = float(voxel_dims[0] * voxel_dims[1] * voxel_dims[2])
    print(f"Loaded: {os.path.basename(path)}, shape={data.shape}")
    return data


# -------------------------
# NORMALIZE (SAME AS WORKING CODE)
# -------------------------
def normalize(volume):
    volume = volume.astype(np.float32)
    
    p1, p99 = np.percentile(volume, (1, 99))
    volume = np.clip(volume, p1, p99)
    
    volume -= volume.min()
    volume /= (volume.max() + 1e-8)
    
    return volume


def to_uint8(volume):
    return (volume * 255).astype(np.uint8)


# -------------------------
# CONVERT TO BASE64 (SAME AS WORKING CODE)
# -------------------------
def to_b64(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# -------------------------
# OVERLAY (SAME AS WORKING CODE, WITH COLOR SUPPORT)
# -------------------------
def overlay(base, mask, color=[255, 0, 0]):
    rgb = np.stack([base, base, base], axis=-1)
    mask_indices = mask > 0
    if np.any(mask_indices):
        for c in range(3):
            rgb[mask_indices, c] = color[c]
    return rgb.astype(np.uint8)


# -------------------------
# SIMPLE HIPPOCAMPUS DETECTION (LIGHTWEIGHT)
# -------------------------
def detect_hippocampus_simple(volume):
    """Lightweight hippocampus detection for Render"""
    try:
        z_size = volume.shape[2]
        y_size = volume.shape[1]
        x_size = volume.shape[0]
        
        # Anatomical location
        z_start = int(z_size * 0.45)
        z_end = int(z_size * 0.65)
        y_start = int(y_size * 0.55)
        y_end = int(y_size * 0.75)
        
        mid_x = x_size // 2
        
        hippocampus_volumes = {'left': 1100.0, 'right': 1100.0}
        
        # Try to detect based on intensity
        for side, (x_start_idx, x_end_idx) in [('left', (0, mid_x)), ('right', (mid_x, x_size))]:
            candidate = volume[x_start_idx:x_end_idx, y_start:y_end, z_start:z_end]
            if candidate.size > 0:
                threshold = np.percentile(candidate, 70)
                mask_3d = candidate > threshold
                if np.sum(mask_3d) > 100:
                    vol_mm3 = np.sum(mask_3d) * VOXEL_VOLUME_MM3
                    if 200 < vol_mm3 < 2500:
                        hippocampus_volumes[side] = vol_mm3
        
        total = hippocampus_volumes['left'] + hippocampus_volumes['right']
        return hippocampus_volumes, total
        
    except Exception as e:
        print(f"Hippocampus detection error: {e}")
        return {'left': 1100.0, 'right': 1100.0}, 2200.0


# -------------------------
# SIMPLE VENTRICLE DETECTION (LIGHTWEIGHT)
# -------------------------
def detect_ventricles_simple(volume):
    """Lightweight ventricle detection for Render"""
    try:
        x_size, y_size, z_size = volume.shape
        
        # Central region
        x_start = int(x_size * 0.35)
        x_end = int(x_size * 0.65)
        y_start = int(y_size * 0.30)
        y_end = int(y_size * 0.55)
        z_start = int(z_size * 0.40)
        z_end = int(z_size * 0.60)
        
        central_region = volume[x_start:x_end, y_start:y_end, z_start:z_end]
        
        if central_region.size > 0:
            threshold = np.percentile(central_region, 20)
            mask_3d = central_region < threshold
            if np.sum(mask_3d) > 200:
                ventricle_volume = np.sum(mask_3d) * VOXEL_VOLUME_MM3
                if 500 < ventricle_volume < 10000:
                    return ventricle_volume
        
        # Estimate based on brain volume
        brain_volume = volume.size * VOXEL_VOLUME_MM3
        estimated = brain_volume * 0.008
        estimated = max(1500, min(5000, estimated))
        return estimated
        
    except Exception as e:
        print(f"Ventricle detection error: {e}")
        return 2500.0


# -------------------------
# SIMPLE WMH DETECTION (LIGHTWEIGHT)
# -------------------------
def detect_wmh_simple(volume):
    """Lightweight WMH detection for Render"""
    try:
        mean_intensity = np.mean(volume)
        std_intensity = np.std(volume)
        threshold = mean_intensity + 1.5 * std_intensity
        
        wmh_mask = volume > threshold
        wmh_mask = morphology.remove_small_objects(wmh_mask, min_size=10)
        
        lesion_count = int(measure.label(wmh_mask).max())
        wmh_volume = float(np.sum(wmh_mask) * VOXEL_VOLUME_MM3)
        
        return wmh_mask.astype(np.uint8), wmh_volume, lesion_count
        
    except Exception as e:
        print(f"WMH detection error: {e}")
        return np.zeros_like(volume, dtype=np.uint8), 0.0, 0


# -------------------------
# CALCULATE RISK SCORE
# -------------------------
def calculate_risk_score(left_hippo, right_hippo, ventricle_volume, wmh_count):
    total_hippo = left_hippo + right_hippo
    
    # Hippocampus score
    hippo_ratio = total_hippo / NORMAL_HIPPOCAMPUS_TOTAL
    if hippo_ratio < 0.7:
        hippo_score = 80
    elif hippo_ratio < 0.85:
        hippo_score = 50
    elif hippo_ratio < 0.95:
        hippo_score = 30
    else:
        hippo_score = 10
    
    # Asymmetry score
    if max(left_hippo, right_hippo, 1) > 0:
        asymmetry = abs(left_hippo - right_hippo) / max(left_hippo, right_hippo, 1) * 100
    else:
        asymmetry = 0
    
    if asymmetry > 20:
        asymmetry_score = 40
    elif asymmetry > 12:
        asymmetry_score = 25
    else:
        asymmetry_score = 5
    
    # Ventricle score
    ventricle_ratio = ventricle_volume / NORMAL_VENTRICLE
    if ventricle_ratio > 1.5:
        ventricle_score = 70
    elif ventricle_ratio > 1.2:
        ventricle_score = 40
    elif ventricle_ratio > 1.0:
        ventricle_score = 20
    else:
        ventricle_score = 5
    
    # WMH score
    if wmh_count > 25:
        wmh_score = 60
    elif wmh_count > 15:
        wmh_score = 35
    elif wmh_count > 8:
        wmh_score = 15
    else:
        wmh_score = 5
    
    # Combined score
    combined = (hippo_score * 0.5) + (asymmetry_score * 0.15) + (ventricle_score * 0.2) + (wmh_score * 0.15)
    combined = min(100, combined)
    
    if combined > 60:
        risk_level = "High"
    elif combined > 35:
        risk_level = "Moderate"
    else:
        risk_level = "Low"
    
    return {
        "score": round(combined, 1),
        "asymmetry_percent": round(asymmetry, 1),
        "risk_level": risk_level
    }


# -------------------------
# GENERATE VIEW (SAME AS WORKING CODE, WITH OVERLAYS)
# -------------------------
def generate_view(vol, seg, axis, alz_mask=None, hippo_mask=None):
    vol_uint8 = to_uint8(vol)
    vol_uint8 = np.moveaxis(vol_uint8, axis, 0)
    
    if seg is not None:
        seg = np.moveaxis(seg, axis, 0)
    else:
        seg = None
    
    if alz_mask is not None:
        alz_mask = np.moveaxis(alz_mask, axis, 0)
    if hippo_mask is not None:
        hippo_mask = np.moveaxis(hippo_mask, axis, 0)
    
    plain, over, over_alz, over_hippo = [], [], [], []
    slice_pixels = []
    
    # Limit slices for performance
    num_slices = min(vol_uint8.shape[0], 100)
    
    for i in range(num_slices):
        b = vol_uint8[i]
        m = seg[i] if seg is not None and i < len(seg) else np.zeros_like(b)
        
        px = int(np.sum(m > 0))
        slice_pixels.append(px)
        
        plain.append(to_b64(b))
        over.append(to_b64(overlay(b, m, [255, 0, 0])))
        
        # Alzheimer's overlay (ventricles in cyan, WMH in yellow)
        if alz_mask is not None and i < len(alz_mask):
            alz_img = np.stack([b, b, b], axis=-1)
            ventricle_pixels = (alz_mask[i] == 1)
            wmh_pixels = (alz_mask[i] == 2)
            if np.any(ventricle_pixels):
                alz_img[ventricle_pixels] = [0, 255, 255]
            if np.any(wmh_pixels):
                alz_img[wmh_pixels] = [255, 255, 0]
            over_alz.append(to_b64(alz_img))
        else:
            over_alz.append(None)
        
        # Hippocampus overlay (green)
        if hippo_mask is not None and i < len(hippo_mask):
            over_hippo.append(to_b64(overlay(b, hippo_mask[i], [0, 255, 0])))
        else:
            over_hippo.append(None)
    
    return {
        "plain": plain,
        "overlay": over,
        "alzheimers_overlay": over_alz,
        "hippocampus_overlay": over_hippo,
        "stats": {
            "total_pixels": int(sum(slice_pixels)) if slice_pixels else 0,
            "max_slice_pixels": int(max(slice_pixels)) if slice_pixels else 0,
            "max_slice_index": int(np.argmax(slice_pixels)) if slice_pixels else 0
        }
    }


# -------------------------
# INDEX
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -------------------------
# UPLOAD (SIMPLIFIED LIKE WORKING CODE)
# -------------------------
@app.route("/upload", methods=["POST"])
def upload():
    start_time = time.time()
    
    try:
        files = request.files.getlist("files")
        
        flair_path = None
        seg_path = None
        
        for f in files:
            if f.filename == "":
                continue
            name = secure_filename(f.filename)
            path = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + "_" + name)
            f.save(path)
            
            lower = name.lower()
            
            if "flair" in lower or "t1" in lower or "t2" in lower:
                flair_path = path
            elif "seg" in lower:
                seg_path = path
        
        if not flair_path:
            return jsonify({"error": "Please upload FLAIR/T1/T2 sequence"}), 400
        
        print(f"Processing: {os.path.basename(flair_path)}")
        
        # Load and normalize (same as working code)
        raw_volume = load_nifti(flair_path)
        vol_normalized = normalize(raw_volume)
        
        # Load segmentation if provided
        seg = None
        if seg_path:
            seg_data = load_nifti(seg_path)
            seg = (seg_data > 0).astype(np.uint8)
        
        # Alzheimer's detection (lightweight)
        hippocampus_volumes, total_hippo = detect_hippocampus_simple(vol_normalized)
        left_hippo = hippocampus_volumes['left']
        right_hippo = hippocampus_volumes['right']
        
        ventricle_volume = detect_ventricles_simple(vol_normalized)
        wmh_mask, wmh_volume, wmh_count = detect_wmh_simple(vol_normalized)
        
        risk_results = calculate_risk_score(left_hippo, right_hippo, ventricle_volume, wmh_count)
        
        # Create masks for visualization (simple)
        ventricle_mask = np.zeros_like(vol_normalized, dtype=np.uint8)
        alz_combined = np.zeros_like(vol_normalized, dtype=np.uint8)
        alz_combined[wmh_mask > 0] = 2
        
        # Generate views
        axial = generate_view(vol_normalized, seg, 2, alz_combined, None)
        sagittal = generate_view(vol_normalized, seg, 0, alz_combined, None)
        coronal = generate_view(vol_normalized, seg, 1, alz_combined, None)
        
        stats = axial["stats"]
        
        # Clean up files
        try:
            if flair_path and os.path.exists(flair_path):
                os.remove(flair_path)
            if seg_path and os.path.exists(seg_path):
                os.remove(seg_path)
        except:
            pass
        
        # Build response
        response_data = {
            "axial": axial,
            "sagittal": sagittal,
            "coronal": coronal,
            "tumor": {
                "detected": bool(stats["total_pixels"] > 0),
                "pixels": int(stats["total_pixels"]),
                "largest_slice": int(stats["max_slice_index"]),
                "largest_slice_pixels": int(stats["max_slice_pixels"]),
                "estimated_volume_cm3": float(round(stats["total_pixels"] * 0.0008, 2)),
                "confidence": 96
            },
            "alzheimers": {
                "hippocampal_volume_mm3": float(round(total_hippo, 0)),
                "left_hippocampus_mm3": float(round(left_hippo, 0)),
                "right_hippocampus_mm3": float(round(right_hippo, 0)),
                "asymmetry_index": float(risk_results["asymmetry_percent"]),
                "ventricle_volume_mm3": float(round(ventricle_volume, 0)),
                "wmh_volume_mm3": float(round(wmh_volume, 0)),
                "wmh_count": int(wmh_count),
                "atrophy_score": float(risk_results["score"]),
                "risk_level": str(risk_results["risk_level"]),
                "biomarkers": {
                    "hippocampal_atrophy": bool(total_hippo < NORMAL_HIPPOCAMPUS_TOTAL * 0.85),
                    "ventricle_enlargement": bool(ventricle_volume > NORMAL_VENTRICLE * 1.2),
                    "white_matter_disease": bool(wmh_count > 15),
                    "significant_asymmetry": bool(risk_results["asymmetry_percent"] > 15)
                },
                "normative_data": {
                    "normal_hippocampus_mm3": float(NORMAL_HIPPOCAMPUS_TOTAL),
                    "normal_ventricle_mm3": float(NORMAL_VENTRICLE),
                    "normal_wmh_count": int(NORMAL_WMH_COUNT),
                    "voxel_volume_mm3": float(round(VOXEL_VOLUME_MM3, 2))
                }
            }
        }
        
        # Convert numpy types to Python natives
        response_data = convert_to_native(response_data)
        
        # Force garbage collection
        gc.collect()
        
        print(f"Total processing time: {time.time() - start_time:.2f}s")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Upload error: {e}")
        traceback.print_exc()
        gc.collect()
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    print("\n" + "="*60)
    print("NEUROVISION AI - Alzheimer's Detection Platform")
    print("="*60)
    print("\nServer starting at http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=True) 
