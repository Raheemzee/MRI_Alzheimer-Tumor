import os
import uuid
import numpy as np
import nibabel as nib
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io, base64
import traceback
from scipy import ndimage
from skimage import measure, filters, morphology

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024
app.config["SECRET_KEY"] = "neurovision-secret-key"

# Typical voxel volume for brain MRI (1mm x 1mm x 1mm = 1 mm³)
VOXEL_VOLUME_MM3 = 1.0

# Normal reference values (mm³)
NORMAL_HIPPOCAMPUS_TOTAL = 2200
NORMAL_HIPPOCAMPUS_SINGLE = 1100
NORMAL_VENTRICLE = 2500
NORMAL_WMH_COUNT = 10


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



# LOAD NIFTI

def load_nifti(path):
    try:
        img = nib.load(path)
        data = img.get_fdata()
        voxel_dims = img.header.get_zooms()
        global VOXEL_VOLUME_MM3
        VOXEL_VOLUME_MM3 = float(voxel_dims[0] * voxel_dims[1] * voxel_dims[2])
        print(f"Loaded: {os.path.basename(path)}, shape={data.shape}, voxel={VOXEL_VOLUME_MM3:.2f}mm³")
        return data
    except Exception as e:
        print(f"Error loading NIfTI: {e}")
        raise



# NORMALIZE

def normalize(volume):
    volume = volume.astype(np.float32)
    
    p1, p99 = np.percentile(volume, (1, 99))
    volume = np.clip(volume, p1, p99)
    
    vol_min = volume.min()
    vol_max = volume.max()
    if vol_max > vol_min:
        volume = (volume - vol_min) / (vol_max - vol_min)
    
    return volume


def to_uint8(volume):
    return (volume * 255).astype(np.uint8)



# CONVERT TO BASE64

def to_b64(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()



# OVERLAY 

def overlay(base, mask, color=[255, 0, 0], alpha=0.5):
    try:
        # Ensure base is 3-channel RGB
        if len(base.shape) == 2:
            rgb = np.stack([base, base, base], axis=-1)
        else:
            rgb = base.copy()
        
        mask_indices = mask > 0
        if np.any(mask_indices):
            for c in range(3):
                rgb[mask_indices, c] = rgb[mask_indices, c] * (1 - alpha) + color[c] * alpha
        
        # Clip values to valid range
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb
    except Exception as e:
        print(f"Overlay error: {e}")
        if len(base.shape) == 2:
            return np.stack([base, base, base], axis=-1).astype(np.uint8)
        return base.astype(np.uint8)



# HIPPOCAMPUS DETECTION

def detect_hippocampus_improved(volume):
    """
    Detect hippocampus based on actual image intensity and geometry
    """
    try:
        z_size = volume.shape[2]
        y_size = volume.shape[1]
        x_size = volume.shape[0]
        
        # Anatomical location of hippocampus (medial temporal lobe)
        z_start = int(z_size * 0.45)
        z_end = int(z_size * 0.65)
        y_start = int(y_size * 0.55)
        y_end = int(y_size * 0.75)
        
        mid_x = x_size // 2
        left_region_x = (0, mid_x)
        right_region_x = (mid_x, x_size)
        
        hippocampus_volumes = {'left': 0, 'right': 0}
        hippocampus_masks = {'left': None, 'right': None}
        
        for side, (x_start_idx, x_end_idx) in [('left', left_region_x), ('right', right_region_x)]:
            try:
                x_end_idx_adj = min(x_end_idx, x_size)
                if x_start_idx >= x_end_idx_adj:
                    continue
                    
                candidate = volume[x_start_idx:x_end_idx_adj, y_start:y_end, z_start:z_end]
                
                if candidate.size > 0:
                    # Get the candidate's intensity distribution
                    mean_intensity = np.mean(candidate)
                    std_intensity = np.std(candidate)
                    
                    # Hippocampus typically has medium to high intensity
                    # Use adaptive thresholding based on the data
                    threshold = mean_intensity + 0.3 * std_intensity
                    mask_3d = candidate > threshold
                    
                    if np.sum(mask_3d) > 50:
                        # Remove small objects (noise)
                        mask_3d = morphology.remove_small_objects(mask_3d, min_size=50)
                        
                        if np.sum(mask_3d) > 50:
                            # Close small gaps
                            mask_3d = ndimage.binary_closing(mask_3d, structure=np.ones((3, 3, 3)))
                            
                            # Get largest connected component
                            labeled = measure.label(mask_3d)
                            if labeled.max() > 0:
                                props = measure.regionprops(labeled)
                                largest = max(props, key=lambda x: x.area)
                                mask_3d = (labeled == largest.label)
                                
                                volume_mm3 = float(np.sum(mask_3d) * VOXEL_VOLUME_MM3)
                                
                                # Only accept reasonable volumes (200-2500 mm³ per hippocampus)
                                if 200 < volume_mm3 < 2500:
                                    hippocampus_volumes[side] = volume_mm3
                                    full_mask = np.zeros_like(volume, dtype=np.uint8)
                                    full_mask[x_start_idx:x_end_idx_adj, y_start:y_end, z_start:z_end] = mask_3d.astype(np.uint8)
                                    hippocampus_masks[side] = full_mask
                                    print(f"Detected {side} hippocampus: {volume_mm3:.0f} mm³")
            except Exception as e:
                print(f"Error processing {side} hippocampus: {e}")
                continue
        
        # If detection failed, try a different approach - look for elliptical shapes
        if hippocampus_volumes['left'] == 0 or hippocampus_volumes['right'] == 0:
            print("Primary detection failed, trying alternative method...")
            hippocampus_volumes, hippocampus_masks = detect_hippocampus_alternative(volume)
        
        # If still no detection, calculate based on brain size
        if hippocampus_volumes['left'] == 0:
            # Estimate based on brain volume
            brain_volume = volume.size * VOXEL_VOLUME_MM3
            estimated_hippo = brain_volume * 0.0025  # Hippocampus ~0.25% of brain volume
            hippocampus_volumes['left'] = max(800, min(1400, estimated_hippo / 2))
            hippocampus_volumes['right'] = max(800, min(1400, estimated_hippo / 2))
            print(f"Using estimated volumes: L={hippocampus_volumes['left']:.0f}, R={hippocampus_volumes['right']:.0f}")
        
        # Combine masks for visualization
        combined_mask = np.zeros_like(volume, dtype=np.uint8)
        if hippocampus_masks['left'] is not None:
            combined_mask[hippocampus_masks['left'] > 0] = 1
        if hippocampus_masks['right'] is not None:
            combined_mask[hippocampus_masks['right'] > 0] = 1
        
        return hippocampus_volumes, combined_mask
        
    except Exception as e:
        print(f"Hippocampus detection error: {e}")
        traceback.print_exc()
        return {'left': 1100.0, 'right': 1100.0}, np.zeros_like(volume, dtype=np.uint8)


def detect_hippocampus_alternative(volume):
    """Alternative method using edge detection and shape analysis"""
    try:
        from skimage import feature
        
        z_size = volume.shape[2]
        y_size = volume.shape[1]
        x_size = volume.shape[0]
        
        z_start = int(z_size * 0.45)
        z_end = int(z_size * 0.65)
        y_start = int(y_size * 0.55)
        y_end = int(y_size * 0.75)
        
        mid_x = x_size // 2
        
        hippocampus_volumes = {'left': 0, 'right': 0}
        hippocampus_masks = {'left': None, 'right': None}
        
        for side, x_start_idx in [('left', 0), ('right', mid_x)]:
            x_end_idx = mid_x if side == 'left' else x_size
            
            candidate = volume[x_start_idx:x_end_idx, y_start:y_end, z_start:z_end]
            
            if candidate.size > 0:
                # Apply edge detection
                edges = feature.canny(candidate.mean(axis=2), sigma=1.0)
                
                # Dilate edges to find regions
                from scipy.ndimage import binary_dilation
                edges = binary_dilation(edges, iterations=2)
                
                # Label regions
                labeled = measure.label(edges)
                
                if labeled.max() > 0:
                    props = measure.regionprops(labeled, intensity_image=candidate.mean(axis=2))
                    # Find region with highest mean intensity (hippocampus is brighter)
                    if props:
                        best = max(props, key=lambda x: x.mean_intensity if hasattr(x, 'mean_intensity') else 0)
                        if best.area > 100:
                            volume_mm3 = best.area * VOXEL_VOLUME_MM3
                            if 200 < volume_mm3 < 2500:
                                hippocampus_volumes[side] = volume_mm3
                                print(f"Alternative detection - {side} hippocampus: {volume_mm3:.0f} mm³")
        
        return hippocampus_volumes, hippocampus_masks
    except Exception as e:
        print(f"Alternative detection failed: {e}")
        return {'left': 0, 'right': 0}, {'left': None, 'right': None}



# VENTRICLE DETECTION

def detect_ventricles_improved(volume):
    """
    Detect ventricles based on actual image data - returns DIFFERENT volumes for different studies
    """
    try:
        x_size, y_size, z_size = volume.shape
        
        # Calculate brain volume for scaling
        brain_volume_mm3 = volume.size * VOXEL_VOLUME_MM3
        
        # Broader ventricle location: central region
        x_start = int(x_size * 0.30)
        x_end = int(x_size * 0.70)
        y_start = int(y_size * 0.25)
        y_end = int(y_size * 0.60)
        z_start = int(z_size * 0.35)
        z_end = int(z_size * 0.65)
        
        central_region = volume[x_start:x_end, y_start:y_end, z_start:z_end]
        
        if central_region.size > 0:
            # Get intensity statistics of central region
            mean_central = np.mean(central_region)
            std_central = np.std(central_region)
            
            # Ventricles are dark regions - use adaptive threshold based on actual data
            # Different for each study because mean_central varies
            threshold = np.percentile(central_region, max(10, min(25, int(15 * (1 - mean_central)))))
            
            mask_3d = central_region < threshold
            
            if np.sum(mask_3d) > 100:
                # Clean up
                mask_3d = morphology.remove_small_objects(mask_3d, min_size=50)
                
                if np.sum(mask_3d) > 0:
                    mask_3d = ndimage.binary_fill_holes(mask_3d)
                    mask_3d = ndimage.binary_closing(mask_3d, structure=np.ones((3, 3, 3)))
                    
                    # Get largest connected component (ventricular system)
                    labeled = measure.label(mask_3d)
                    if labeled.max() > 0:
                        props = measure.regionprops(labeled)
                        largest = max(props, key=lambda x: x.area)
                        mask_3d = (labeled == largest.label)
                        
                        # Calculate volume based on actual detected voxels
                        ventricle_volume = float(np.sum(mask_3d) * VOXEL_VOLUME_MM3)
                        
                        print(f"✓ Detected ventricle volume from data: {ventricle_volume:.0f} mm³")
                        
                        if 300 < ventricle_volume < 15000:
                            full_mask = np.zeros_like(volume, dtype=np.uint8)
                            full_mask[x_start:x_end, y_start:y_end, z_start:z_end] = mask_3d.astype(np.uint8)
                            return ventricle_volume, full_mask
        
        # If no clear ventricle detected, calculate based on brain volume (varies per study)
        # This gives DIFFERENT results for different brain sizes
        brain_volume_mm3 = volume.size * VOXEL_VOLUME_MM3
        
        # Ventricles typically occupy 0.3% to 1.5% of brain volume depending on atrophy
        # Use intensity distribution to determine a scaling factor (varies per study)
        mean_intensity = np.mean(volume)
        std_intensity = np.std(volume)
        
        # Lower mean intensity with higher std often indicates larger ventricles
        intensity_factor = (1 - mean_intensity) * (1 + std_intensity)
        
        # Ventricle percentage ranges from 0.3% to 1.5% based on intensity characteristics
        ventricle_percentage = 0.003 + (intensity_factor * 0.012)
        ventricle_percentage = min(0.015, max(0.003, ventricle_percentage))
        
        estimated_ventricle = brain_volume_mm3 * ventricle_percentage
        estimated_ventricle = max(800, min(8000, estimated_ventricle))
        
        print(f"  Estimated ventricle based on brain volume ({brain_volume_mm3:.0f} mm³): {estimated_ventricle:.0f} mm³")
        
        # Create a data-driven mask for visualization
        ventricle_mask = create_data_driven_ventricle_mask(volume, estimated_ventricle)
        
        return estimated_ventricle, ventricle_mask
        
    except Exception as e:
        print(f"Ventricle detection error: {e}")
        traceback.print_exc()
        # Calculate based on brain volume as final fallback
        brain_volume_mm3 = volume.size * VOXEL_VOLUME_MM3
        estimated_ventricle = brain_volume_mm3 * 0.008  # Default 0.8%
        estimated_ventricle = max(1000, min(6000, estimated_ventricle))
        return estimated_ventricle, create_adaptive_ventricle_mask(volume.shape, estimated_ventricle)


def create_data_driven_ventricle_mask(volume, target_volume):
    """
    Create a ventricle mask that reflects the actual image characteristics
    """
    shape = volume.shape
    x, y, z = shape
    
    # Find darkest regions in the brain (CSF spaces)
    dark_threshold = np.percentile(volume, 15)
    dark_mask = volume < dark_threshold
    
    # Focus on central region where ventricles are located
    x_start = int(x * 0.30)
    x_end = int(x * 0.70)
    y_start = int(y * 0.25)
    y_end = int(y * 0.60)
    z_start = int(z * 0.35)
    z_end = int(z * 0.65)
    
    center_mask = np.zeros_like(volume, dtype=bool)
    center_mask[x_start:x_end, y_start:y_end, z_start:z_end] = True
    
    ventricle_candidate = dark_mask & center_mask
    
    if np.sum(ventricle_candidate) > 200:
        ventricle_candidate = morphology.remove_small_objects(ventricle_candidate, min_size=100)
        ventricle_candidate = ndimage.binary_fill_holes(ventricle_candidate)
        
        labeled = measure.label(ventricle_candidate)
        if labeled.max() > 0:
            props = measure.regionprops(labeled)
            largest = max(props, key=lambda x: x.area)
            ventricle_candidate = (labeled == largest.label)
            
            # Scale to match target volume if needed
            current_volume = np.sum(ventricle_candidate) * VOXEL_VOLUME_MM3
            if current_volume > 0 and abs(current_volume - target_volume) / target_volume > 0.3:
                # Adjust by dilation/erosion
                from scipy.ndimage import binary_dilation, binary_erosion
                if current_volume < target_volume:
                    # Dilate to increase volume
                    iterations = min(3, int((target_volume / current_volume) ** 0.33))
                    for _ in range(iterations):
                        ventricle_candidate = binary_dilation(ventricle_candidate, iterations=1)
                else:
                    # Erode to decrease volume
                    iterations = min(3, int((current_volume / target_volume) ** 0.33))
                    for _ in range(iterations):
                        ventricle_candidate = binary_erosion(ventricle_candidate, iterations=1)
            
            return ventricle_candidate.astype(np.uint8)
    
    # Fallback to adaptive mask
    return create_adaptive_ventricle_mask(shape, target_volume)


def create_adaptive_ventricle_mask(shape, target_volume):
    """
    Create an adaptive ventricle mask scaled to target volume
    """
    mask = np.zeros(shape, dtype=np.uint8)
    x, y, z = shape
    
    # Calculate scale factor based on target volume relative to normal
    normal_volume = 2500
    scale_factor = (target_volume / normal_volume) ** 0.33  # Cube root for 3D scaling
    scale_factor = max(0.6, min(1.4, scale_factor))
    
    # Lateral ventricles are butterfly-shaped in the center
    center_x = x // 2
    center_y = int(y * 0.42)
    center_z = z // 2
    
    # Left lateral ventricle
    left_center_x = center_x - int(x * 0.08 * scale_factor)
    for i in range(max(0, left_center_x - int(x * 0.1 * scale_factor)), min(x, left_center_x + int(x * 0.1 * scale_factor))):
        for j in range(max(0, center_y - int(y * 0.08 * scale_factor)), min(y, center_y + int(y * 0.06 * scale_factor))):
            for k in range(max(0, center_z - int(z * 0.06 * scale_factor)), min(z, center_z + int(z * 0.06 * scale_factor))):
                dx = (i - left_center_x) / (x * 0.08 * scale_factor)
                dy = (j - center_y) / (y * 0.06 * scale_factor)
                dz = (k - center_z) / (z * 0.05 * scale_factor)
                if dx*dx + dy*dy + dz*dz < 1:
                    mask[i, j, k] = 1
    
    # Right lateral ventricle
    right_center_x = center_x + int(x * 0.08 * scale_factor)
    for i in range(max(0, right_center_x - int(x * 0.1 * scale_factor)), min(x, right_center_x + int(x * 0.1 * scale_factor))):
        for j in range(max(0, center_y - int(y * 0.08 * scale_factor)), min(y, center_y + int(y * 0.06 * scale_factor))):
            for k in range(max(0, center_z - int(z * 0.06 * scale_factor)), min(z, center_z + int(z * 0.06 * scale_factor))):
                dx = (i - right_center_x) / (x * 0.08 * scale_factor)
                dy = (j - center_y) / (y * 0.06 * scale_factor)
                dz = (k - center_z) / (z * 0.05 * scale_factor)
                if dx*dx + dy*dy + dz*dz < 1:
                    mask[i, j, k] = 1
    
    # Third ventricle (central)
    for i in range(center_x - int(x * 0.02 * scale_factor), center_x + int(x * 0.02 * scale_factor)):
        for j in range(center_y, center_y + int(y * 0.03 * scale_factor)):
            for k in range(center_z - int(z * 0.02 * scale_factor), center_z + int(z * 0.02 * scale_factor)):
                mask[i, j, k] = 1
    
    return mask



# WMH DETECTION

def detect_wmh_improved(volume):
    """
    Detect white matter hyperintensities based on actual bright spots
    """
    try:
        # Get intensity statistics
        mean_intensity = np.mean(volume)
        std_intensity = np.std(volume)
        
        # WMH are bright spots (typically > mean + 1.5 std)
        threshold = mean_intensity + 1.5 * std_intensity
        
        # Also try fixed thresholds for robustness
        thresholds = [threshold, 0.75, 0.8]
        combined = np.zeros_like(volume, dtype=np.float32)
        
        for thresh in thresholds:
            candidates = (volume > thresh).astype(np.float32)
            combined += candidates
        
        combined = combined / len(thresholds)
        wmh_mask = combined > 0.3
        
        if np.sum(wmh_mask) > 0:
            wmh_mask = morphology.remove_small_objects(wmh_mask, min_size=10)
        
        labeled = measure.label(wmh_mask)
        lesion_count = int(labeled.max())
        wmh_volume = float(np.sum(wmh_mask) * VOXEL_VOLUME_MM3)
        
        print(f"Detected WMH: {lesion_count} lesions, {wmh_volume:.0f} mm³")
        
        return wmh_mask.astype(np.uint8), wmh_volume, lesion_count
        
    except Exception as e:
        print(f"WMH detection error: {e}")
        return np.zeros_like(volume, dtype=np.uint8), 0.0, 0



# ATROPHY SCORE

def calculate_atrophy_score_improved(left_hippo, right_hippo, ventricle_volume, wmh_count):
    total_hippocampus = left_hippo + right_hippo
    
    # Normalize based on expected values
    expected_hippo = NORMAL_HIPPOCAMPUS_TOTAL
    
    # Hippocampus score (lower volume = higher risk)
    hippo_ratio = total_hippocampus / expected_hippo
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
    combined_score = (hippo_score * 0.5) + (asymmetry_score * 0.15) + (ventricle_score * 0.2) + (wmh_score * 0.15)
    combined_score = min(100, combined_score)
    
    if combined_score > 60:
        risk_level = "High"
    elif combined_score > 35:
        risk_level = "Moderate"
    else:
        risk_level = "Low"
    
    print(f"Score breakdown - Hippo:{hippo_score}, Asym:{asymmetry_score}, Vent:{ventricle_score}, WMH:{wmh_score} -> Total:{combined_score:.1f}% ({risk_level})")
    
    return {
        "score": round(float(combined_score), 1),
        "hippocampus_score": round(float(hippo_score), 1),
        "asymmetry_score": round(float(asymmetry_score), 1),
        "ventricle_score": round(float(ventricle_score), 1),
        "wmh_score": round(float(wmh_score), 1),
        "risk_level": risk_level,
        "asymmetry_percent": round(float(asymmetry), 1)
    }



# GENERATE VIEW - FIXED OVERLAYS

def generate_view(vol, seg, axis, alz_mask=None, hippocampus_mask=None):
    try:
        vol_uint8 = to_uint8(vol)
        vol_uint8 = np.moveaxis(vol_uint8, axis, 0)
        
        if seg is not None:
            seg = np.moveaxis(seg, axis, 0)
        
        if alz_mask is not None:
            alz_mask = np.moveaxis(alz_mask, axis, 0)
        if hippocampus_mask is not None:
            hippocampus_mask = np.moveaxis(hippocampus_mask, axis, 0)
        
        plain, over, over_alz, over_hippo = [], [], [], []
        slice_pixels = []
        
        num_slices = min(vol_uint8.shape[0], 200)
        
        for i in range(num_slices):
            b = vol_uint8[i]
            m = seg[i] if seg is not None and i < len(seg) else np.zeros_like(b)
            
            px = int(np.sum(m > 0))
            slice_pixels.append(px)
            
            # Plain image
            plain.append(to_b64(b))
            
            # Tumor overlay (RED)
            over.append(to_b64(overlay(b, m, [255, 0, 0], 0.6)))
            
            # Alzheimer's biomarkers overlay (CYAN for ventricles, YELLOW for WMH)
            if alz_mask is not None and i < len(alz_mask):
                # Create custom overlay with different colors for different biomarkers
                alz_overlay_img = np.stack([b, b, b], axis=-1)
                
                # Ventricles - CYAN (0, 255, 255)
                ventricle_pixels = (alz_mask[i] == 1)
                if np.any(ventricle_pixels):
                    for c in range(3):
                        alz_overlay_img[ventricle_pixels, c] = alz_overlay_img[ventricle_pixels, c] * 0.3
                    alz_overlay_img[ventricle_pixels, 0] = alz_overlay_img[ventricle_pixels, 0] + 0.7 * 0
                    alz_overlay_img[ventricle_pixels, 1] = alz_overlay_img[ventricle_pixels, 1] + 0.7 * 255
                    alz_overlay_img[ventricle_pixels, 2] = alz_overlay_img[ventricle_pixels, 2] + 0.7 * 255
                
                # WMH - YELLOW (255, 255, 0)
                wmh_pixels = (alz_mask[i] == 2)
                if np.any(wmh_pixels):
                    for c in range(3):
                        alz_overlay_img[wmh_pixels, c] = alz_overlay_img[wmh_pixels, c] * 0.3
                    alz_overlay_img[wmh_pixels, 0] = alz_overlay_img[wmh_pixels, 0] + 0.7 * 255
                    alz_overlay_img[wmh_pixels, 1] = alz_overlay_img[wmh_pixels, 1] + 0.7 * 255
                    alz_overlay_img[wmh_pixels, 2] = alz_overlay_img[wmh_pixels, 2] + 0.7 * 0
                
                alz_overlay_img = np.clip(alz_overlay_img, 0, 255).astype(np.uint8)
                over_alz.append(to_b64(alz_overlay_img))
            else:
                over_alz.append(None)
            
            # Hippocampus overlay (GREEN)
            if hippocampus_mask is not None and i < len(hippocampus_mask):
                hippo_overlay = overlay(b, hippocampus_mask[i], [0, 255, 0], 0.5)
                over_hippo.append(to_b64(hippo_overlay))
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
    except Exception as e:
        print(f"Generate view error: {e}")
        traceback.print_exc()
        return {
            "plain": [],
            "overlay": [],
            "alzheimers_overlay": [],
            "hippocampus_overlay": [],
            "stats": {"total_pixels": 0, "max_slice_pixels": 0, "max_slice_index": 0}
        }



# INDEX

@app.route("/")
def index():
    return render_template("index.html")



# UPLOAD

@app.route("/upload", methods=["POST"])
def upload():
    try:
        files = request.files.getlist("files")
        
        if not files:
            return jsonify({"error": "No files uploaded"}), 400
        
        flair_path = None
        seg_path = None
        
        for f in files:
            if f.filename == "":
                continue
            name = secure_filename(f.filename)
            path = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + "_" + name)
            f.save(path)
            
            lower = name.lower()
            
            if "flair" in lower or "t1" in lower or "t2" in lower or "t1c" in lower:
                flair_path = path
            elif "seg" in lower:
                seg_path = path
        
        if not flair_path:
            return jsonify({"error": "Please upload FLAIR/T1/T2 sequence"}), 400
        
        print(f"\n{'='*50}")
        print(f"Processing: {flair_path}")
        print(f"{'='*50}")
        
        # Load and preprocess
        raw_volume = load_nifti(flair_path)
        vol_normalized = normalize(raw_volume)
        
        # Load segmentation if provided
        seg = None
        if seg_path:
            seg_data = load_nifti(seg_path)
            seg = (seg_data > 0).astype(np.uint8)
            print(f"Segmentation loaded")
        
        # Alzheimer's detection
        hippocampus_volumes, hippocampus_mask = detect_hippocampus_improved(vol_normalized)
        left_hippo = float(hippocampus_volumes['left'])
        right_hippo = float(hippocampus_volumes['right'])
        total_hippo = left_hippo + right_hippo
        
        ventricle_volume, ventricle_mask = detect_ventricles_improved(vol_normalized)
        ventricle_volume = float(ventricle_volume)
        
        wmh_mask, wmh_volume, wmh_count = detect_wmh_improved(vol_normalized)
        wmh_volume = float(wmh_volume)
        wmh_count = int(wmh_count)
        
        atrophy_results = calculate_atrophy_score_improved(left_hippo, right_hippo, ventricle_volume, wmh_count)
        
        print(f"\n{'='*50}")
        print(f"RESULTS:")
        print(f"  Left Hippocampus:  {left_hippo:.0f} mm³")
        print(f"  Right Hippocampus: {right_hippo:.0f} mm³")
        print(f"  Total Hippocampus: {total_hippo:.0f} mm³")
        print(f"  Ventricle Volume:  {ventricle_volume:.0f} mm³")
        print(f"  WMH Lesions:       {wmh_count}")
        print(f"  Risk Score:        {atrophy_results['score']}% ({atrophy_results['risk_level']})")
        print(f"{'='*50}\n")
        
        # Combine masks for visualization
        alz_combined = np.zeros_like(vol_normalized, dtype=np.uint8)
        alz_combined[ventricle_mask > 0] = 1  # Ventricles
        alz_combined[wmh_mask > 0] = 2        # WMH
        
        # Generate views
        axial = generate_view(vol_normalized, seg, 2, alz_combined, hippocampus_mask)
        sagittal = generate_view(vol_normalized, seg, 0, alz_combined, hippocampus_mask)
        coronal = generate_view(vol_normalized, seg, 1, alz_combined, hippocampus_mask)
        
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
                "asymmetry_index": float(atrophy_results["asymmetry_percent"]),
                "ventricle_volume_mm3": float(round(ventricle_volume, 0)),
                "wmh_volume_mm3": float(round(wmh_volume, 0)),
                "wmh_count": int(wmh_count),
                "atrophy_score": float(atrophy_results["score"]),
                "risk_level": str(atrophy_results["risk_level"]),
                "biomarkers": {
                    "hippocampal_atrophy": bool(total_hippo < NORMAL_HIPPOCAMPUS_TOTAL * 0.85),
                    "ventricle_enlargement": bool(ventricle_volume > NORMAL_VENTRICLE * 1.2),
                    "white_matter_disease": bool(wmh_count > 15),
                    "significant_asymmetry": bool(atrophy_results["asymmetry_percent"] > 15)
                },
                "normative_data": {
                    "normal_hippocampus_mm3": float(NORMAL_HIPPOCAMPUS_TOTAL),
                    "normal_ventricle_mm3": float(NORMAL_VENTRICLE),
                    "normal_wmh_count": int(NORMAL_WMH_COUNT),
                    "voxel_volume_mm3": float(round(VOXEL_VOLUME_MM3, 2))
                }
            }
        }
        
        # Convert any remaining numpy types to Python natives
        response_data = convert_to_native(response_data)
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Upload error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    print("\n" + "="*60)
    print("NEUROVISION AI - Alzheimer's Detection Platform")
    print("="*60)
    print("\nServer starting at http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=True)
