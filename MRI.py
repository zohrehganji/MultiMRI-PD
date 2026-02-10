#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/MRI.py — MultiMRI-PD

Multimodal Hybrid Deep Learning for Explainable Parkinson's Disease Diagnosis:
Integrating Nigrostriatal Neuromelanin and Perfusion MRI Dynamics

Ganji Z, Nikparast F, Shoeibi N, Shoeibi A, Shieali E, Hashemi SM, Zare H (2026)

This script faithfully implements the seven-stage framework described in the
paper, including:

  PREPROCESSING
    1. Visual / automated QC
    2. N4ITK bias-field correction (SimpleITK)
    3. Brain extraction — HD-BET (deep-learning) with Otsu morphological fallback
    4. Intra-subject rigid registration (NM → T1, ASL → T1) via ANTsPy / MI cost
    5. MNI152 non-linear registration (ANTs SyN) and Harvard–Oxford atlas propagation
    6. ASL CBF quantification via FSL BASIL (subprocess) with partial-volume correction
    7. Subject-wise z-score intensity normalisation (within brain mask only)
    8. NM-MRI biomarker extraction (SNR, CNR of substantia nigra)
    9. ROI-aware 5-slice axial selection centred on SNpc; 2.5D stack formation
       (3 modalities × 5 slices = 15 channels)
   10. Tabular feature vector assembly (NM metrics, regional CBF, age)

  MODEL TRAINING (paper Stages 1–5)
    Stage 1  Unimodal benchmarks (T1-only, NM-only, ASL-tabular-only)
    Stage 2  2.5D ResNet18 with modified input layer
    Stage 3  Hybrid fusion: ResNet18-CBAM image branch + MLP/Multi-head
             Attention tabular branch + late fusion classifier
    Stage 4  Ablation over modality combinations, fusion strategies (early /
             standard-late / attention-late), and backbones (ResNet18/34,
             DenseNet121, EfficientNet-B3)
    Stage 5  Isotonic probability calibration; Youden-optimal threshold

  EXPLAINABILITY (paper Stages 6–7)
    Stage 6  Image-level: multi-layer Grad-CAM with gamma correction,
             Integrated Gradients (Captum, 50 steps)
    Stage 7  Tabular-level: KernelSHAP; quantitative fidelity (ROI coverage)
             and reliability (stability index μ/σ) metrics

  TRAINING PROTOCOL (faithful to paper)
    • Stratified 5-fold subject-level cross-validation (no slice leakage)
    • AdamW optimiser (lr 1e-4, weight-decay 1e-5)
    • Batch size 64
    • Early stopping patience 10 on validation loss
    • Conservative augmentation: rotation ±10°, translation ≤3 px,
      mild intensity jitter; NO horizontal flip
    • Weighted random sampler for class imbalance

Usage:
  # Single-subject preprocessing
  python code/MRI.py --mode single --t1 T1.nii.gz --nm NM.nii.gz \\
      --asl ASL.nii.gz --asl-m0 M0.nii.gz --subject-id SUB01 \\
      --age 62 --output-root ./out

  # Batch preprocessing + training + explainability
  python code/MRI.py --mode batch --input-root /data/cohort \\
      --output-root /data/results --stages 1,2,3,4,5,6,7

External dependencies (optional but recommended):
  • HD-BET  (pip install hd-bet)
  • ANTsPy  (pip install antspyx)
  • FSL     (BASIL / oxford_asl for quantitative CBF)
  • Captum  (pip install captum)
  • SHAP    (pip install shap)

If any external tool is unavailable the script falls back to simpler
alternatives and logs a warning.

Author: adapted for Zohreh Ganji (2026)
License: MIT
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS, CONFIGURATION, LOGGING, REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

# ---- PyTorch core ----
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.models as tvmodels
import torchvision.transforms.functional as TF

# ---- scikit-learn ----
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

# ---- Optional: ANTsPy for registration ----
try:
    import ants
    ANTSPY_AVAILABLE = True
except ImportError:
    ANTSPY_AVAILABLE = False

# ---- Optional: Captum for Integrated Gradients ----
try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

# ---- Optional: SHAP for tabular explainability ----
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("MultiMRI-PD")

# ---- Default pipeline configuration (override via CLI or config file) ----
DEFAULT_CONFIG: Dict[str, Any] = {
    # Paths
    "INPUT_ROOT": "",
    "OUTPUT_ROOT": "./sMRI_pipeline_output",
    # Slice selection
    "NUM_SLICES": 5,                       # paper: 5 adjacent axial slices
    # Morphological cleanup
    "CLOSE_RADIUS_VOX": 3,
    "MIN_COMPONENT_VOX": 1000,
    # External tool names
    "HD_BET_CMD": "hd-bet",
    "BASIL_CMD": "oxford_asl",
    # N4 parameters
    "N4_SHRINKFACTOR": 2,
    # ASL acquisition parameters (pCASL, Philips Ingenia 1.5 T)
    "ASL_BOLUS": 1.65,                     # labelling duration (s)
    "ASL_PLD": 1.80,                       # post-label delay (s)
    "ASL_TI": 3.45,                        # TI = bolus + PLD
    # Training
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "BATCH_SIZE": 64,
    "NUM_WORKERS": 4,
    "MAX_EPOCHS": 200,
    "PATIENCE": 10,
    "LR": 1e-4,
    "WEIGHT_DECAY": 1e-5,
    "SEED": 42,
    "NUM_FOLDS": 5,
    # Augmentation
    "AUG_ROTATION": 10,                    # ±degrees
    "AUG_TRANSLATE_PX": 3,                 # max pixels
    "AUG_INTENSITY_SCALE": 0.05,           # ± relative
    "AUG_INTENSITY_SHIFT": 0.05,           # ± absolute (after z-norm)
    # Model
    "TAB_D_MODEL": 64,                     # tabular embedding dimension
    "TAB_NUM_HEADS": 4,
    "TAB_D_FF": 128,
    "DROPOUT_CLASSIFIER": 0.4,
    "DROPOUT_TAB": 0.3,
    # Grad-CAM gamma correction exponent
    "GRADCAM_GAMMA": 0.5,
    # Integrated Gradients steps
    "IG_STEPS": 50,
    # Harvard–Oxford atlas ROI label indices (FSL sub-cortical maxprob atlas)
    "ATLAS_ROIS": {
        "L_Thalamus": 4,
        "L_Caudate": 5,
        "L_Putamen": 6,
        "L_Pallidum": 7,
        "Brainstem": 8,
        "R_Thalamus": 15,
        "R_Caudate": 16,
        "R_Putamen": 17,
        "R_Pallidum": 18,
    },
    # Tabular feature names (must match order in feature vector)
    "TABULAR_FEATURE_NAMES": [
        "NM_SNR_SN_L", "NM_SNR_SN_R",
        "NM_CNR_SN_L", "NM_CNR_SN_R",
        "CBF_Putamen_L", "CBF_Putamen_R",
        "CBF_Caudate_L", "CBF_Caudate_R",
        "CBF_Thalamus_L", "CBF_Thalamus_R",
        "CBF_Pallidum_L", "CBF_Pallidum_R",
        "Age",
    ],
}

# ---- Reproducibility ----
def seed_everything(seed: int = 42) -> None:
    """Set random seeds for full reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(DEFAULT_CONFIG["SEED"])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — I/O UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_dir(p: str | Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_nifti(path: str | Path) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Load a NIfTI file and return (data_float32, affine, header)."""
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img.affine.copy(), img.header


def save_nifti(data: np.ndarray, affine: np.ndarray, path: str | Path) -> None:
    """Save a numpy array as a NIfTI-1 file."""
    if data.dtype == bool:
        data = data.astype(np.uint8)
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), str(path))


def locate_fsl_atlas(name: str = "HarvardOxford-sub-maxprob-thr25-2mm") -> Optional[Path]:
    """Try to find the Harvard–Oxford subcortical atlas shipped with FSL."""
    fsldir = os.environ.get("FSLDIR", "")
    candidate = Path(fsldir) / "data" / "atlases" / "HarvardOxford" / f"{name}.nii.gz"
    if candidate.exists():
        return candidate
    # fallback: search common locations
    for d in ["/usr/share/fsl", "/usr/local/fsl", "/opt/fsl"]:
        c2 = Path(d) / "data" / "atlases" / "HarvardOxford" / f"{name}.nii.gz"
        if c2.exists():
            return c2
    logger.warning("Harvard–Oxford atlas not found. ROI-based metrics will be skipped.")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PREPROCESSING PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# 3.1  N4 Bias-Field Correction (SimpleITK wrapper)
# ---------------------------------------------------------------------------
def run_n4_correction(
    in_path: str, out_path: str, shrink: int = 2
) -> None:
    """Apply N4ITK bias-field correction (Tustison et al. 2010) via SimpleITK."""
    logger.info(f"N4 correction: {Path(in_path).name}")
    img = sitk.ReadImage(str(in_path), sitk.sitkFloat32)
    # Otsu mask for N4 (foreground)
    try:
        mask = sitk.OtsuThreshold(img, 0, 1)
    except Exception:
        mask = sitk.Image(img.GetSize(), sitk.sitkUInt8) + 1
        mask.CopyInformation(img)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    corrector.SetConvergenceThreshold(1e-6)
    try:
        corrected = corrector.Execute(img, mask)
    except TypeError:
        # Older SimpleITK API
        corrected = sitk.N4BiasFieldCorrection(img, mask)
    sitk.WriteImage(corrected, str(out_path))


# ---------------------------------------------------------------------------
# 3.2  Brain Extraction — HD-BET (preferred) + Otsu fallback
# ---------------------------------------------------------------------------
def try_hdbet(in_nii: str, out_prefix: str, device: str = "cpu") -> Optional[str]:
    """
    Attempt HD-BET skull stripping (Isensee et al. 2019).
    Returns the path of the brain mask NIfTI if successful, else None.
    """
    hd_exe = shutil.which(DEFAULT_CONFIG["HD_BET_CMD"])
    if hd_exe is None:
        logger.info("HD-BET CLI not found; will use Otsu fallback.")
        return None
    cmd = [
        hd_exe, "-i", str(in_nii), "-o", str(out_prefix),
        "-device", device, "-mode", "fast", "-tta", "0",
    ]
    logger.info("Running HD-BET: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=600,
        )
        if proc.returncode != 0:
            logger.warning("HD-BET returned %d — stderr: %s", proc.returncode, proc.stderr[:500])
    except Exception as exc:
        logger.warning("HD-BET execution error: %s", exc)
        return None
    # HD-BET produces *_mask.nii.gz or *_bet_mask.nii.gz depending on version
    for suffix in ["_mask.nii.gz", "_bet_mask.nii.gz", ".nii.gz"]:
        candidate = Path(str(out_prefix) + suffix)
        if candidate.exists():
            return str(candidate)
    return None


def otsu_brain_mask(
    volume: np.ndarray, close_r: int = 3, min_vox: int = 1000
) -> np.ndarray:
    """
    Fallback brain mask via Otsu thresholding + morphological cleanup
    (binary closing with a ball structuring element, largest-component retention).
    """
    v = np.clip(volume, 0, None)
    fg = v[v > 0]
    thresh = filters.threshold_otsu(fg) if fg.size > 100 else np.percentile(v, 50)
    mask = (v > thresh).astype(np.uint8)
    mask = morphology.binary_closing(mask, morphology.ball(close_r)).astype(np.uint8)
    labels = measure.label(mask)
    if labels.max() == 0:
        return mask
    largest = max(measure.regionprops(labels), key=lambda p: p.area).label
    mask = (labels == largest).astype(np.uint8)
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_vox).astype(np.uint8)
    return mask


def brain_extraction(
    n4_path: str, out_dir: str, subj: str, cfg: dict
) -> Tuple[str, str]:
    """
    Produce a brain mask and skull-stripped volume.
    Returns (mask_path, brain_path).
    """
    mask_path = str(Path(out_dir) / f"{subj}_brain_mask.nii.gz")
    brain_path = str(Path(out_dir) / f"{subj}_brain.nii.gz")

    hdbet_out = try_hdbet(n4_path, str(Path(out_dir) / f"{subj}_hdbet"), device="cpu")
    if hdbet_out is not None:
        m, aff, _ = load_nifti(hdbet_out)
        mask = (m > 0.5).astype(np.uint8)
        method = "hd-bet"
    else:
        vol, aff, _ = load_nifti(n4_path)
        mask = otsu_brain_mask(vol, close_r=cfg["CLOSE_RADIUS_VOX"], min_vox=cfg["MIN_COMPONENT_VOX"])
        method = "otsu-fallback"

    save_nifti(mask, aff, mask_path)
    logger.info("Brain mask (%s): %s", method, Path(mask_path).name)

    vol, aff, _ = load_nifti(n4_path)
    brain = vol * mask
    save_nifti(brain, aff, brain_path)
    return mask_path, brain_path


# ---------------------------------------------------------------------------
# 3.3  Intra-Subject Rigid Registration (NM/ASL → T1)
# ---------------------------------------------------------------------------
def rigid_register_to_t1(
    moving_path: str, fixed_t1_path: str, out_path: str
) -> str:
    """
    Rigidly register *moving* volume to the T1 reference using mutual
    information cost function.  Uses ANTsPy if available, else SimpleITK fallback.
    """
    if ANTSPY_AVAILABLE:
        fixed = ants.image_read(str(fixed_t1_path))
        moving = ants.image_read(str(moving_path))
        result = ants.registration(
            fixed=fixed, moving=moving,
            type_of_transform="Rigid",
            aff_metric="MI",
            verbose=False,
        )
        ants.image_write(result["warpedmovout"], str(out_path))
        logger.info("Rigid registration (ANTsPy): %s → T1", Path(moving_path).name)
        return str(out_path)

    # ---------- SimpleITK fallback ----------
    logger.info("Rigid registration (SimpleITK fallback): %s → T1", Path(moving_path).name)
    fixed_sitk = sitk.ReadImage(str(fixed_t1_path), sitk.sitkFloat32)
    moving_sitk = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
    init_tx = sitk.CenteredTransformInitializer(
        fixed_sitk, moving_sitk,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=200,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    )
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(init_tx, inPlace=False)
    final_tx = reg.Execute(fixed_sitk, moving_sitk)
    resampled = sitk.Resample(
        moving_sitk, fixed_sitk, final_tx, sitk.sitkLinear, 0.0,
    )
    sitk.WriteImage(resampled, str(out_path))
    return str(out_path)


# ---------------------------------------------------------------------------
# 3.4  MNI152 Non-linear Registration (ANTs SyN) + Atlas Propagation
# ---------------------------------------------------------------------------
def register_to_mni_and_propagate_atlas(
    t1_brain_path: str,
    out_dir: str,
    subj: str,
    atlas_mni_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Non-linearly register T1 brain to MNI152 2 mm template using ANTs SyN.
    Warp Harvard–Oxford atlas back to native space for ROI extraction.
    Returns dict with paths of:
      - t1_in_mni, forward_warp, inverse_warp, atlas_native
    """
    result_paths: Dict[str, str] = {}

    if not ANTSPY_AVAILABLE:
        logger.warning(
            "ANTsPy not available — MNI registration and atlas propagation skipped. "
            "ROI-based features will be unavailable."
        )
        return result_paths

    # MNI template bundled with ANTsPy
    mni = ants.image_read(ants.get_ants_data("mni"))
    t1 = ants.image_read(str(t1_brain_path))

    logger.info("SyN registration to MNI152 for %s …", subj)
    reg = ants.registration(
        fixed=mni, moving=t1,
        type_of_transform="SyN",
        syn_metric="CC",
        syn_sampling=4,
        verbose=False,
    )
    t1_mni_path = str(Path(out_dir) / f"{subj}_T1_MNI.nii.gz")
    ants.image_write(reg["warpedmovout"], t1_mni_path)
    result_paths["t1_in_mni"] = t1_mni_path
    result_paths["fwd_transforms"] = reg["fwdtransforms"]
    result_paths["inv_transforms"] = reg["invtransforms"]

    # Propagate atlas to native space using inverse transforms
    if atlas_mni_path and Path(atlas_mni_path).exists():
        atlas_mni = ants.image_read(str(atlas_mni_path))
        atlas_native = ants.apply_transforms(
            fixed=t1,
            moving=atlas_mni,
            transformlist=reg["invtransforms"],
            interpolator="nearestNeighbor",
        )
        atlas_nat_path = str(Path(out_dir) / f"{subj}_atlas_native.nii.gz")
        ants.image_write(atlas_native, atlas_nat_path)
        result_paths["atlas_native"] = atlas_nat_path
        logger.info("Atlas propagated to native space: %s", Path(atlas_nat_path).name)

    return result_paths


# ---------------------------------------------------------------------------
# 3.5  ASL CBF Quantification — FSL BASIL (subprocess)
# ---------------------------------------------------------------------------
def run_basil_cbf(
    asl_path: str,
    m0_path: Optional[str],
    t1_anat_dir: Optional[str],
    out_dir: str,
    subj: str,
    cfg: dict,
) -> Optional[str]:
    """
    Call FSL's oxford_asl (BASIL) for quantitative CBF estimation with
    partial-volume correction.  Returns the path to the CBF map, or None
    if BASIL is unavailable.

    Recommended external call (for documentation / manual use):
        oxford_asl -i <asl> -o <outdir> --casl --bolus=1.65 --tis=3.45 \\
            -c <m0> --cmethod=single --pvcorr --fslanat=<t1_anat_dir>
    """
    basil_exe = shutil.which(cfg.get("BASIL_CMD", "oxford_asl"))
    if basil_exe is None:
        logger.warning(
            "FSL oxford_asl not found — CBF will be estimated with a simple "
            "pairwise-subtraction fallback (NOT recommended for publication)."
        )
        return None

    basil_out = str(Path(out_dir) / f"{subj}_basil")
    cmd = [
        basil_exe,
        "-i", str(asl_path),
        "-o", basil_out,
        "--casl",
        f"--bolus={cfg['ASL_BOLUS']}",
        f"--tis={cfg['ASL_TI']}",
        "--spatial",
    ]
    if m0_path:
        cmd += ["-c", str(m0_path), "--cmethod=single"]
    if t1_anat_dir:
        cmd += [f"--fslanat={t1_anat_dir}"]
        cmd += ["--pvcorr"]                 # partial-volume correction

    logger.info("Running BASIL: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1200)
    except Exception as exc:
        logger.warning("BASIL failed: %s", exc)
        return None

    # BASIL outputs CBF in native_space/perfusion_calib.nii.gz
    cbf_candidates = [
        Path(basil_out) / "native_space" / "perfusion_calib.nii.gz",
        Path(basil_out) / "native_space" / "perfusion.nii.gz",
    ]
    for c in cbf_candidates:
        if c.exists():
            final = str(Path(out_dir) / f"{subj}_CBF.nii.gz")
            shutil.copy(str(c), final)
            logger.info("BASIL CBF map: %s", Path(final).name)
            return final
    logger.warning("BASIL completed but expected CBF output not found.")
    return None


def simple_cbf_fallback(asl_path: str, out_dir: str, subj: str) -> str:
    """
    Minimal fallback when BASIL is not available:
    pairwise label–control subtraction → mean → Gaussian smooth → z-normalise.
    NOTE: this does NOT produce quantitative CBF in physiological units;
    use only for prototyping.
    """
    logger.info("Fallback CBF estimation (pairwise subtraction) for %s", subj)
    vol, aff, _ = load_nifti(asl_path)
    if vol.ndim == 4 and vol.shape[3] >= 2:
        controls = vol[:, :, :, 0::2]
        labels = vol[:, :, :, 1::2]
        n = min(controls.shape[3], labels.shape[3])
        diff = (controls[:, :, :, :n] - labels[:, :, :, :n]).mean(axis=3)
    else:
        diff = vol if vol.ndim == 3 else vol.mean(axis=3)

    diff = ndi.gaussian_filter(diff, sigma=1.5)
    fg = diff[diff > 0]
    if fg.size > 0:
        diff = (diff - fg.mean()) / (fg.std() + 1e-8)
    out_path = str(Path(out_dir) / f"{subj}_CBF_fallback.nii.gz")
    save_nifti(diff, aff, out_path)
    return out_path


# ---------------------------------------------------------------------------
# 3.6  Subject-wise Z-Score Normalisation
# ---------------------------------------------------------------------------
def zscore_normalise(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Subject-wise z-score normalisation within the brain mask.
    Paper §2.2 step 5: voxel intensities within brain mask normalised to
    zero mean and unit standard deviation computed from that subject's own
    brain voxels.  Avoids cross-subject leakage.
    """
    brain_voxels = vol[mask > 0]
    if brain_voxels.size == 0:
        return vol
    mu = float(brain_voxels.mean())
    sigma = float(brain_voxels.std())
    if sigma < 1e-8:
        sigma = 1.0
    normed = (vol - mu) / sigma
    normed[mask == 0] = 0.0
    return normed


# ---------------------------------------------------------------------------
# 3.7  NM-MRI Biomarker Extraction (SNR, CNR)
# ---------------------------------------------------------------------------
def compute_nm_metrics(
    nm_vol: np.ndarray,
    atlas_native: Optional[np.ndarray],
    roi_labels: Dict[str, int],
) -> Dict[str, float]:
    """
    Compute neuromelanin SNR and CNR for the substantia nigra (SN) in each
    hemisphere.

    SNR = μ_SN / σ_noise
    CNR = (μ_SN − μ_ref) / σ_noise

    where μ_ref is mean intensity in adjacent cerebral-peduncle white matter,
    and σ_noise is the standard deviation of a background (air) region.
    When atlas-based SN delineation is unavailable, we use an intensity-
    threshold region-growing heuristic within the midbrain bounding box.
    """
    metrics: Dict[str, float] = {}

    # Estimate noise σ from background (lowest 5 % of non-zero voxels or zeros)
    bg = nm_vol[nm_vol < np.percentile(nm_vol[nm_vol > 0], 5)] if (nm_vol > 0).any() else nm_vol.ravel()
    sigma_noise = float(bg.std()) if bg.size > 10 else 1.0
    if sigma_noise < 1e-8:
        sigma_noise = 1.0

    # If we have the atlas in native space, use brainstem ROI as approximate SN location
    if atlas_native is not None:
        brainstem_label = roi_labels.get("Brainstem", 8)
        brainstem_mask = (atlas_native == brainstem_label)
        # Heuristic: top 20 % intensity voxels within brainstem → likely neuromelanin-rich
        bs_vals = nm_vol[brainstem_mask]
        if bs_vals.size > 0:
            thresh = np.percentile(bs_vals, 80)
            sn_mask = brainstem_mask & (nm_vol >= thresh)
            ref_mask = brainstem_mask & (nm_vol < thresh)
            mu_sn = float(nm_vol[sn_mask].mean()) if sn_mask.any() else 0.0
            mu_ref = float(nm_vol[ref_mask].mean()) if ref_mask.any() else 0.0
            # Split left / right by midline (x-axis centre)
            cx = nm_vol.shape[0] // 2
            for side, slc in [("L", slice(0, cx)), ("R", slice(cx, None))]:
                side_sn = sn_mask.copy()
                side_sn[slc, :, :] = False          # keep opposite side False
                side_sn = sn_mask & ~side_sn         # isolate this side (inverted logic)
                # Simpler: directly mask by hemisphere
                hemi_mask = np.zeros_like(sn_mask)
                if side == "L":
                    hemi_mask[:cx, :, :] = True
                else:
                    hemi_mask[cx:, :, :] = True
                side_sn_vals = nm_vol[sn_mask & hemi_mask]
                side_ref_vals = nm_vol[ref_mask & hemi_mask]
                mu_s = float(side_sn_vals.mean()) if side_sn_vals.size > 0 else 0.0
                mu_r = float(side_ref_vals.mean()) if side_ref_vals.size > 0 else 0.0
                metrics[f"NM_SNR_SN_{side}"] = mu_s / sigma_noise
                metrics[f"NM_CNR_SN_{side}"] = (mu_s - mu_r) / sigma_noise
    else:
        # No atlas: use centre-of-volume heuristic for SN
        Z = nm_vol.shape[2]
        mid_slices = range(max(0, Z // 2 - 2), min(Z, Z // 2 + 3))
        sub = nm_vol[:, :, list(mid_slices)]
        fg = sub[sub > 0]
        if fg.size > 0:
            thresh = np.percentile(fg, 80)
            sn_approx = sub[sub >= thresh]
            ref_approx = sub[(sub > 0) & (sub < thresh)]
            mu_sn = float(sn_approx.mean()) if sn_approx.size else 0.0
            mu_ref = float(ref_approx.mean()) if ref_approx.size else 0.0
        else:
            mu_sn = mu_ref = 0.0
        metrics["NM_SNR_SN_L"] = mu_sn / sigma_noise
        metrics["NM_SNR_SN_R"] = mu_sn / sigma_noise
        metrics["NM_CNR_SN_L"] = (mu_sn - mu_ref) / sigma_noise
        metrics["NM_CNR_SN_R"] = (mu_sn - mu_ref) / sigma_noise

    return metrics


# ---------------------------------------------------------------------------
# 3.8  Regional CBF Extraction (atlas-based)
# ---------------------------------------------------------------------------
def extract_regional_cbf(
    cbf_vol: np.ndarray,
    atlas_native: Optional[np.ndarray],
    roi_labels: Dict[str, int],
) -> Dict[str, float]:
    """
    Extract mean CBF within Harvard–Oxford subcortical ROIs.
    Falls back to whole-brain mean if atlas unavailable.
    """
    metrics: Dict[str, float] = {}
    if atlas_native is not None:
        for name, idx in roi_labels.items():
            roi_mask = (atlas_native == idx)
            vals = cbf_vol[roi_mask]
            metrics[f"CBF_{name}"] = float(vals.mean()) if vals.size > 0 else 0.0
    else:
        # Fallback: global mean within positive values
        fg = cbf_vol[cbf_vol > 0]
        global_mean = float(fg.mean()) if fg.size > 0 else 0.0
        for side in ["L", "R"]:
            for region in ["Putamen", "Caudate", "Thalamus", "Pallidum"]:
                metrics[f"CBF_{side}_{region}"] = global_mean
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ROI-AWARE SLICE SELECTION & 2.5D INPUT FORMATION
# ═══════════════════════════════════════════════════════════════════════════════

def select_axial_slices(
    mask: np.ndarray,
    k: int = 5,
    atlas_native: Optional[np.ndarray] = None,
    roi_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Select *k* contiguous axial slices that maximise coverage of PD-relevant
    structures (SNpc, basal ganglia, thalamus).

    Paper §2.2 step 8: "five adjacent axial slices centred on the SNpc were
    chosen for the primary 2.5D input."

    Strategy:
    1. If an atlas in native space is available, compute per-slice ROI
       coverage and pick the k-slice window with highest total overlap.
    2. Otherwise, use the centre-of-mass of the brain mask along z.
    """
    Z = mask.shape[2]
    k = min(k, Z)  # safety

    if atlas_native is not None and roi_indices:
        roi_mask = np.isin(atlas_native, roi_indices).astype(np.uint8)
        # Per-slice ROI voxel count
        coverage = np.array([
            np.logical_and(roi_mask[:, :, z], mask[:, :, z]).sum()
            for z in range(Z)
        ], dtype=np.float64)
        # Sliding window of size k → max sum
        if Z >= k:
            sums = np.convolve(coverage, np.ones(k), mode="valid")
            best_start = int(np.argmax(sums))
        else:
            best_start = 0
        return np.arange(best_start, best_start + k)

    # Fallback: centre-of-mass
    if mask.sum() > 0:
        com_z = int(np.round(ndi.center_of_mass(mask)[2]))
    else:
        com_z = Z // 2
    half = k // 2
    start = max(0, min(com_z - half, Z - k))
    return np.arange(start, start + k)


def build_2p5d_stack(vol: np.ndarray, z_indices: np.ndarray) -> np.ndarray:
    """
    Extract *k* axial slices from a 3-D volume and return shape (k, H, W).
    """
    return np.stack([vol[:, :, int(z)] for z in z_indices], axis=0).astype(np.float32)


def form_multimodal_input(
    t1: np.ndarray,
    nm: Optional[np.ndarray],
    asl_cbf: Optional[np.ndarray],
    z_indices: np.ndarray,
) -> np.ndarray:
    """
    Stack slices from each available modality along the channel axis.
    Paper §2.2 step 8: "3 modalities × 5 slices = 15 channels".
    Returns shape (C, H, W).
    """
    channels = [build_2p5d_stack(t1, z_indices)]
    if nm is not None:
        channels.append(build_2p5d_stack(nm, z_indices))
    if asl_cbf is not None:
        channels.append(build_2p5d_stack(asl_cbf, z_indices))
    return np.concatenate(channels, axis=0)  # (C, H, W)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TABULAR FEATURE VECTOR ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def assemble_tabular_features(
    nm_metrics: Dict[str, float],
    cbf_metrics: Dict[str, float],
    age: float,
    feature_names: List[str],
) -> np.ndarray:
    """
    Build a fixed-length tabular feature vector in the canonical order
    defined in DEFAULT_CONFIG["TABULAR_FEATURE_NAMES"].
    Missing values default to 0.0.
    """
    merged = {**nm_metrics, **cbf_metrics, "Age": age}
    vec = np.array([merged.get(f, 0.0) for f in feature_names], dtype=np.float32)
    return vec


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DATA AUGMENTATION & PYTORCH DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class ConservativeAugmentation:
    """
    Paper §2.4 augmentation (applied identically across all channels for a
    given sample so that multimodal spatial correspondence is preserved):

      • Random in-plane rotation ±10° (no out-of-plane)
      • Random in-plane translation ≤3 px (x and y)
      • Mild intensity jitter (random scale [0.95, 1.05] + shift [-0.05, 0.05])
      • NO horizontal flip (preserve lateralisation patterns)
    """

    def __init__(self, max_rot: float = 10.0, max_trans: int = 3,
                 scale_range: float = 0.05, shift_range: float = 0.05):
        self.max_rot = max_rot
        self.max_trans = max_trans
        self.scale_range = scale_range
        self.shift_range = shift_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """tensor: (C, H, W) float."""
        # --- spatial transforms (identical across channels) ---
        angle = float(torch.empty(1).uniform_(-self.max_rot, self.max_rot))
        tx = int(torch.randint(-self.max_trans, self.max_trans + 1, (1,)))
        ty = int(torch.randint(-self.max_trans, self.max_trans + 1, (1,)))

        # Rotation
        tensor = TF.rotate(tensor, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        # Translation via affine (no scaling, no shearing)
        tensor = TF.affine(
            tensor, angle=0, translate=[tx, ty], scale=1.0, shear=0,
            interpolation=TF.InterpolationMode.BILINEAR, fill=0,
        )

        # --- intensity jitter (per-channel random scale + shift) ---
        C = tensor.shape[0]
        scale = 1.0 + (torch.rand(C, 1, 1) * 2 - 1) * self.scale_range
        shift = (torch.rand(C, 1, 1) * 2 - 1) * self.shift_range
        tensor = tensor * scale + shift

        return tensor


class MultimodalPDDataset(Dataset):
    """
    PyTorch Dataset for the hybrid model.

    Each manifest entry must contain:
      - subject_id : str
      - fused_npy  : str   (path to (C, H, W) numpy array)
      - tabular    : list[float] or np.ndarray  (length = num_tab_features)
      - label      : int   (0 = HC, 1 = PD)

    The dataset can optionally apply conservative spatial and intensity
    augmentation (training set only).
    """

    def __init__(
        self,
        manifest: List[Dict[str, Any]],
        augment: bool = False,
        cfg: Optional[dict] = None,
    ):
        self.manifest = manifest
        self.augment_fn = ConservativeAugmentation(
            max_rot=cfg.get("AUG_ROTATION", 10) if cfg else 10,
            max_trans=cfg.get("AUG_TRANSLATE_PX", 3) if cfg else 3,
            scale_range=cfg.get("AUG_INTENSITY_SCALE", 0.05) if cfg else 0.05,
            shift_range=cfg.get("AUG_INTENSITY_SHIFT", 0.05) if cfg else 0.05,
        ) if augment else None

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        entry = self.manifest[idx]
        # Load pre-computed fused 2.5D stack
        arr = np.load(entry["fused_npy"]).astype(np.float32)  # (C, H, W)
        img_tensor = torch.from_numpy(arr)

        if self.augment_fn is not None:
            img_tensor = self.augment_fn(img_tensor)

        # Tabular features
        tab = np.array(entry["tabular"], dtype=np.float32)
        tab_tensor = torch.from_numpy(tab)

        label = torch.tensor(float(entry["label"]), dtype=torch.float32)
        return img_tensor, tab_tensor, label, entry["subject_id"]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# 7.1  Convolutional Block Attention Module (CBAM)
#      Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
# ---------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    """Channel attention sub-module of CBAM."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        avg_pool = x.mean(dim=(2, 3))                       # (B, C)
        max_pool = x.amax(dim=(2, 3))                       # (B, C)
        attn = torch.sigmoid(self.shared_mlp(avg_pool) + self.shared_mlp(max_pool))
        return x * attn.view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention sub-module of CBAM."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)      # (B, 2, H, W)
        attn = torch.sigmoid(self.conv(combined))             # (B, 1, H, W)
        return x * attn


class CBAM(nn.Module):
    """Full CBAM block: channel attention → spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, spatial_ks: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(spatial_ks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x


# ---------------------------------------------------------------------------
# 7.2  ResNet18 Backbone with CBAM & Modified Input Layer
# ---------------------------------------------------------------------------
class ResNet18CBAM(nn.Module):
    """
    ImageNet-pretrained ResNet18 with:
          • First conv layer adapted from 3 → *in_channels* channels
            (via weight replication / averaging, following standard practice)
          • CBAM inserted after each residual stage (layer1–layer4)
          • Global average pool → 512-dim embedding
        """

    def __init__(self, in_channels: int = 15):
        super().__init__()
        base = tvmodels.resnet18(weights=tvmodels.ResNet18_Weights.IMAGENET1K_V1)

        # --- Adapt first conv from 3 → in_channels ---
        old_conv = base.conv1  # (64, 3, 7, 7)
        new_conv = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False,
        )
        with torch.no_grad():
            # Replicate pretrained weights across new channels
            repeat = math.ceil(in_channels / 3)
            w = old_conv.weight.data.repeat(1, repeat, 1, 1)[:, :in_channels, :, :]
            w /= (in_channels / 3.0)  # scale to preserve activation magnitude
            new_conv.weight.copy_(w)
        self.conv1 = new_conv
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        # Residual stages with CBAM after each
        self.layer1 = base.layer1  # 64 channels
        self.cbam1 = CBAM(64)
        self.layer2 = base.layer2  # 128 channels
        self.cbam2 = CBAM(128)
        self.layer3 = base.layer3  # 256 channels
        self.cbam3 = CBAM(256)
        self.layer4 = base.layer4  # 512 channels
        self.cbam4 = CBAM(512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns 512-d embedding (B, 512)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))

        x = self.avgpool(x)
        return x.flatten(1)  # (B, 512)

    def get_last_conv_layer(self) -> nn.Module:
        """Return the final convolutional stage for Grad-CAM hook attachment."""
        return self.layer4


# ---------------------------------------------------------------------------
# 7.3  Multi-Head Self-Attention Tabular Branch
# ---------------------------------------------------------------------------
class TabularAttentionBranch(nn.Module):
    """
    Paper §2.3 Stage 3, tabular branch:
      MLP projection → Multi-head Self-Attention → learned embedding.

    Input : (B, num_features)
    Output: (B, d_model) attended embedding  AND  (B, num_features) raw copy
    """

    def __init__(
            self,
            num_features: int,
            d_model: int = 64,
            num_heads: int = 4,
            d_ff: int = 128,
            dropout: float = 0.3,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Project each scalar feature to d_model-dim token
        self.feature_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Learnable positional embedding for each feature slot
        self.pos_embed = nn.Parameter(torch.randn(1, num_features, d_model) * 0.02)

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward block
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Pool tokens → single embedding
        self.pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : (B, num_features) raw tabular vector
        Returns:
            attended : (B, d_model)  attention-pooled embedding
            raw      : (B, num_features)  passthrough copy for late fusion
        """
        B = x.shape[0]
        # Each feature becomes a token: (B, F) → (B, F, 1) → (B, F, d_model)
        tokens = self.feature_proj(x.unsqueeze(-1))  # (B, F, d_model)
        tokens = tokens + self.pos_embed  # add positional encoding

        # Self-attention
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)

        # Feed-forward
        ff_out = self.ff(tokens)
        tokens = self.norm2(tokens + ff_out)

        # Attention-weighted pool across feature tokens → single embedding
        weights = torch.softmax(self.pool(tokens).mean(dim=-1), dim=1)  # (B, F)
        attended = (tokens * weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)

        return attended, x  # also return raw features for concatenation


# ---------------------------------------------------------------------------
# 7.4  Hybrid Fusion Classifier (ResNet18-CBAM + Tabular Attention + Late Fusion)
# ---------------------------------------------------------------------------
class HybridFusionModel(nn.Module):
    """
    Paper §2.3 Stage 3 — full architecture:

      Image branch  :  ResNet18-CBAM → 512-dim embedding
      Tabular branch:  MLP + Multi-head Attention → d_model-dim embedding
      Fusion        :  concat(image_embed, attended_tab, raw_tab) → MLP → logit

    The fusion is "attention-enhanced late fusion" as described in the paper.
    """

    def __init__(
            self,
            img_channels: int = 15,
            num_tab_features: int = 13,
            d_model: int = 64,
            num_heads: int = 4,
            d_ff: int = 128,
            dropout_cls: float = 0.4,
            dropout_tab: float = 0.3,
    ):
        super().__init__()
        self.image_branch = ResNet18CBAM(in_channels=img_channels)
        self.tabular_branch = TabularAttentionBranch(
            num_features=num_tab_features,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout_tab,
        )

        # Fusion dimension = 512 (image) + d_model (attended) + num_tab_features (raw)
        fused_dim = 512 + d_model + num_tab_features

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_cls),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_cls),
            nn.Linear(64, 1),
        )

    def forward(
            self, img: torch.Tensor, tab: torch.Tensor
    ) -> torch.Tensor:
        """
        img : (B, C, H, W)
        tab : (B, num_tab_features)
        Returns logit (B,) — apply sigmoid for probability.
        """
        img_embed = self.image_branch(img)  # (B, 512)
        tab_attend, tab_raw = self.tabular_branch(tab)  # (B, d_model), (B, F)
        fused = torch.cat([img_embed, tab_attend, tab_raw], dim=1)
        logit = self.classifier(fused).squeeze(-1)  # (B,)
        return logit

    def get_image_branch(self) -> ResNet18CBAM:
        """Accessor for Grad-CAM hooks."""
        return self.image_branch


# ---------------------------------------------------------------------------
# 7.5  Alternative Backbones (for Stage 4 ablation)
# ---------------------------------------------------------------------------
def _make_backbone(name: str, in_channels: int) -> Tuple[nn.Module, int]:
    """
    Create an ImageNet-pretrained backbone with modified input layer.
    Returns (backbone_module, embedding_dim).
    Supported: resnet18, resnet34, densenet121, efficientnet_b3
    """
    if name == "resnet18":
        return ResNet18CBAM(in_channels), 512

    elif name == "resnet34":
        base = tvmodels.resnet34(weights=tvmodels.ResNet34_Weights.IMAGENET1K_V1)
        old_w = base.conv1.weight.data
        new_conv = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            r = math.ceil(in_channels / 3)
            w = old_w.repeat(1, r, 1, 1)[:, :in_channels, :, :]
            w /= (in_channels / 3.0)
            new_conv.weight.copy_(w)
        base.conv1 = new_conv
        base.fc = nn.Identity()
        return base, 512

    elif name == "densenet121":
        base = tvmodels.densenet121(weights=tvmodels.DenseNet121_Weights.IMAGENET1K_V1)
        old_w = base.features.conv0.weight.data
        new_conv = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            r = math.ceil(in_channels / 3)
            w = old_w.repeat(1, r, 1, 1)[:, :in_channels, :, :]
            w /= (in_channels / 3.0)
            new_conv.weight.copy_(w)
        base.features.conv0 = new_conv
        base.classifier = nn.Identity()
        return base, 1024

    elif name == "efficientnet_b3":
        base = tvmodels.efficientnet_b3(weights=tvmodels.EfficientNet_B3_Weights.IMAGENET1K_V1)
        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=False,
        )
        with torch.no_grad():
            old_w = old_conv.weight.data
            r = math.ceil(in_channels / 3)
            w = old_w.repeat(1, r, 1, 1)[:, :in_channels, :, :]
            w /= (in_channels / 3.0)
            new_conv.weight.copy_(w)
        base.features[0][0] = new_conv
        base.classifier = nn.Identity()
        return base, 1536

    else:
        raise ValueError(f"Unsupported backbone: {name}")


class FlexibleHybridModel(nn.Module):
    """
    Same architecture as HybridFusionModel but accepts an arbitrary backbone.
    Used for Stage 4 backbone comparison ablation.
    """

    def __init__(
            self,
            backbone: nn.Module,
            img_embed_dim: int,
            num_tab_features: int = 13,
            d_model: int = 64,
            num_heads: int = 4,
            d_ff: int = 128,
            dropout_cls: float = 0.4,
            dropout_tab: float = 0.3,
    ):
        super().__init__()
        self.image_branch = backbone
        self.tabular_branch = TabularAttentionBranch(
            num_features=num_tab_features,
            d_model=d_model, num_heads=num_heads,
            d_ff=d_ff, dropout=dropout_tab,
        )
        fused_dim = img_embed_dim + d_model + num_tab_features
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout_cls),
            nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout_cls),
            nn.Linear(64, 1),
        )

    def forward(self, img: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        img_embed = self.image_branch(img)
        if img_embed.dim() > 2:
            img_embed = F.adaptive_avg_pool2d(img_embed, 1).flatten(1)
        tab_attend, tab_raw = self.tabular_branch(tab)
        fused = torch.cat([img_embed, tab_attend, tab_raw], dim=1)
        return self.classifier(fused).squeeze(-1)


# ---------------------------------------------------------------------------
# 7.6  Unimodal Baselines (Stage 1)
# ---------------------------------------------------------------------------
class UnimodalImageModel(nn.Module):
    """Unimodal image-only model (T1, NM, or ASL slices)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.backbone = ResNet18CBAM(in_channels)
        self.head = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(128, 1),
        )

    def forward(self, img: torch.Tensor, tab: torch.Tensor = None) -> torch.Tensor:
        return self.head(self.backbone(img)).squeeze(-1)


class UnimodalTabularModel(nn.Module):
    """Unimodal tabular-only model (ASL CBF + demographics)."""

    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, img: torch.Tensor = None, tab: torch.Tensor = None) -> torch.Tensor:
        return self.net(tab).squeeze(-1)


# ---------------------------------------------------------------------------
# 7.7  Fusion Strategy Variants (Stage 4)
# ---------------------------------------------------------------------------
class EarlyFusionModel(nn.Module):
    """
    Early fusion: concatenate tabular features into the image tensor
    by broadcasting across spatial dimensions before the CNN.
    """

    def __init__(self, img_channels: int, num_tab_features: int):
        super().__init__()
        self.backbone = ResNet18CBAM(img_channels + num_tab_features)
        self.head = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(128, 1),
        )

    def forward(self, img: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        B, _, H, W = img.shape
        tab_map = tab.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)
        x = torch.cat([img, tab_map], dim=1)
        return self.head(self.backbone(x)).squeeze(-1)


class StandardLateFusionModel(nn.Module):
    """
    Standard late fusion: concatenate image embed + raw tabular (no attention).
    """

    def __init__(self, img_channels: int, num_tab_features: int):
        super().__init__()
        self.image_branch = ResNet18CBAM(img_channels)
        self.classifier = nn.Sequential(
            nn.Linear(512 + num_tab_features, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward(self, img: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        img_embed = self.image_branch(img)
        fused = torch.cat([img_embed, tab], dim=1)
        return self.classifier(fused).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TRAINING LOOP WITH EARLY STOPPING AND WEIGHTED SAMPLER
# ═══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """Standard early stopping monitor on validation loss."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: Optional[dict] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to handle class imbalance (HC / PD).
    Each sample is weighted by the inverse of its class frequency.
    """
    class_counts = np.bincount(labels.astype(int))
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = class_weights[labels.astype(int)]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        device: str,
) -> float:
    """Train for one epoch, return mean loss."""
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n = 0
    for img, tab, label, _ in loader:
        img = img.to(device)
        tab = tab.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        logit = model(img, tab)
        loss = criterion(logit, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * img.size(0)
        n += img.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
        model: nn.Module,
        loader: DataLoader,
        device: str,
) -> Dict[str, Any]:
    """
    Evaluate model, returning dict with loss, probabilities, labels, and
    subject IDs (for downstream calibration and metrics).
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    all_probs, all_labels, all_ids = [], [], []
    for img, tab, label, sids in loader:
        img = img.to(device);
        tab = tab.to(device);
        label = label.to(device)
        logit = model(img, tab)
        total_loss += criterion(logit, label).item()
        all_probs.append(torch.sigmoid(logit).cpu().numpy())
        all_labels.append(label.cpu().numpy())
        all_ids.extend(sids)
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    loss = total_loss / max(len(labels), 1)

    # Metrics
    try:
        auc_val = roc_auc_score(labels, probs)
    except ValueError:
        auc_val = float("nan")
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    sens = recall_score(labels, preds, zero_division=0)
    spec = recall_score(1 - labels, 1 - preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    brier = brier_score_loss(labels, probs)

    return {
        "loss": loss,
        "auc": auc_val,
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "brier": brier,
        "probs": probs,
        "labels": labels,
        "subject_ids": all_ids,
    }


def run_training(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        fold_label: str = "",
) -> Dict[str, Any]:
    """
    Full training loop with AdamW, early stopping, and best-model restoration.
    Returns final validation metrics + training history.
    """
    device = cfg["DEVICE"]
    model = model.to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"],
    )
    stopper = EarlyStopping(patience=cfg["PATIENCE"])
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, cfg["MAX_EPOCHS"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "[%s] Epoch %3d | TrainLoss %.4f | ValLoss %.4f | AUC %.4f",
                fold_label, epoch, train_loss, val_metrics["loss"], val_metrics["auc"],
            )

        if stopper.step(val_metrics["loss"], model):
            logger.info("[%s] Early stopping at epoch %d", fold_label, epoch)
            break

    stopper.restore_best(model)
    final_val = evaluate(model, val_loader, device)
    final_val["history"] = history
    return final_val


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — STRATIFIED 5-FOLD CROSS-VALIDATION DRIVER
# ═══════════════════════════════════════════════════════════════════════════════

def run_cross_validation(
        manifest: List[Dict[str, Any]],
        model_factory,
        cfg: dict,
        experiment_name: str = "experiment",
        out_dir: str = "./cv_results",
) -> Dict[str, Any]:
    """
    Stratified 5-fold subject-level cross-validation.

    Paper §2.4: "Subjects, not slices, were the unit of splitting: in each
    fold, approximately 80 % of subjects were used for training and 20 % for
    validation, preserving the PD/HC ratio."

    Parameters
    ----------
    manifest : list of dicts (one per subject) with keys: fused_npy, tabular,
               label, subject_id.
    model_factory : callable() → nn.Module  (creates a fresh model per fold)
    cfg : pipeline configuration dict.
    experiment_name : descriptive string for logging / output filenames.
    out_dir : directory for fold-level results.

    Returns
    -------
    dict with pooled metrics, per-fold metrics, and calibration results.
    """
    ensure_dir(out_dir)
    labels = np.array([e["label"] for e in manifest])
    skf = StratifiedKFold(n_splits=cfg["NUM_FOLDS"], shuffle=True, random_state=cfg["SEED"])

    fold_metrics = []
    pooled_probs = []
    pooled_labels = []
    pooled_ids = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(manifest, labels)):
        fold_tag = f"{experiment_name}/fold{fold_idx}"
        logger.info("===== %s =====", fold_tag)

        train_manifest = [manifest[i] for i in train_idx]
        val_manifest = [manifest[i] for i in val_idx]

        # Datasets
        train_ds = MultimodalPDDataset(train_manifest, augment=True, cfg=cfg)
        val_ds = MultimodalPDDataset(val_manifest, augment=False, cfg=cfg)

        # Weighted sampler for class imbalance
        train_labels = np.array([e["label"] for e in train_manifest])
        sampler = make_weighted_sampler(train_labels)

        train_loader = DataLoader(
            train_ds, batch_size=cfg["BATCH_SIZE"], sampler=sampler,
            num_workers=cfg["NUM_WORKERS"], pin_memory=True, drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg["BATCH_SIZE"], shuffle=False,
            num_workers=cfg["NUM_WORKERS"], pin_memory=True,
        )

        model = model_factory()
        result = run_training(model, train_loader, val_loader, cfg, fold_label=fold_tag)

        fold_metrics.append({
            "fold": fold_idx,
            "auc": result["auc"],
            "accuracy": result["accuracy"],
            "sensitivity": result["sensitivity"],
            "specificity": result["specificity"],
            "precision": result["precision"],
            "brier": result["brier"],
        })
        pooled_probs.append(result["probs"])
        pooled_labels.append(result["labels"])
        pooled_ids.extend(result["subject_ids"])

        # Save fold model (small checkpoints only — large ones go to Zenodo)
        ckpt_path = Path(out_dir) / f"{experiment_name}_fold{fold_idx}.pt"
        torch.save(model.state_dict(), str(ckpt_path))

    # --- Pooled metrics across all folds ---
    pooled_probs = np.concatenate(pooled_probs)
    pooled_labels = np.concatenate(pooled_labels)

    try:
        pooled_auc = roc_auc_score(pooled_labels, pooled_probs)
    except ValueError:
        pooled_auc = float("nan")

    # --- Isotonic probability calibration (Stage 5) ---
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(pooled_probs, pooled_labels)
    calibrated_probs = iso.predict(pooled_probs)
    brier_before = brier_score_loss(pooled_labels, pooled_probs)
    brier_after = brier_score_loss(pooled_labels, calibrated_probs)

    # Youden-optimal threshold (on calibrated probabilities)
    fpr, tpr, thresholds = roc_curve(pooled_labels, calibrated_probs)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    optimal_threshold = float(thresholds[best_idx])

    # Summary
    fold_aucs = [fm["auc"] for fm in fold_metrics]
    ci_lo = np.percentile(fold_aucs, 2.5)
    ci_hi = np.percentile(fold_aucs, 97.5)

    summary = {
        "experiment": experiment_name,
        "pooled_auc": pooled_auc,
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "std_fold_auc": float(np.std(fold_aucs)),
        "auc_95ci": (ci_lo, ci_hi),
        "brier_before_calibration": brier_before,
        "brier_after_calibration": brier_after,
        "youden_threshold": optimal_threshold,
        "fold_metrics": fold_metrics,
        "pooled_probs": pooled_probs,
        "pooled_labels": pooled_labels,
        "pooled_ids": pooled_ids,
        "calibrated_probs": calibrated_probs,
    }

    # Save summary JSON (exclude numpy arrays)
    json_safe = {k: v for k, v in summary.items()
                 if not isinstance(v, np.ndarray)}
    json_safe["auc_95ci"] = list(json_safe["auc_95ci"])
    with open(Path(out_dir) / f"{experiment_name}_summary.json", "w") as f:
        json.dump(json_safe, f, indent=2)

    logger.info(
        "[%s] Pooled AUC=%.4f  Mean±SD=%.4f±%.4f  95%%CI=(%.4f,%.4f)  "
        "Brier %.4f→%.4f  Threshold %.3f",
        experiment_name, pooled_auc, np.mean(fold_aucs), np.std(fold_aucs),
        ci_lo, ci_hi, brier_before, brier_after, optimal_threshold,
    )
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — ABLATION EXPERIMENTS (STAGES 1–4)
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_ablation(
        manifest: List[Dict[str, Any]],
        cfg: dict,
        out_dir: str,
) -> pd.DataFrame:
    """
    Paper Stages 1–4: systematic ablation over modality combinations,
    fusion strategies, and CNN backbones.

    Modality combinations (7):
      T1-only, NM-only, ASL-tab-only, T1+NM, T1+ASL, NM+ASL, T1+NM+ASL

    Fusion strategies (3, on full T1+NM+ASL):
      Early, Standard-Late, Attention-Late

    Backbones (4, on Attention-Late T1+NM+ASL):
      ResNet18, ResNet34, DenseNet121, EfficientNet-B3
    """
    ensure_dir(out_dir)
    k = cfg["NUM_SLICES"]
    num_tab = len(cfg["TABULAR_FEATURE_NAMES"])
    all_results = []

    # ── Helper: create a modality-reduced manifest ──
    def reduce_manifest(combo: List[str]) -> List[Dict[str, Any]]:
        """Slice fused_npy to include only selected modalities' channels."""
        reduced = []
        for entry in manifest:
            arr = np.load(entry["fused_npy"])  # (C, H, W) — full 15 channels
            total_c = arr.shape[0]
            modalities_present = total_c // k  # how many modalities were stacked

            ch_sel = []
            if "T1" in combo:
                ch_sel.extend(range(0, k))
            if "NM" in combo and modalities_present >= 2:
                ch_sel.extend(range(k, 2 * k))
            if "ASL" in combo and modalities_present >= 3:
                ch_sel.extend(range(2 * k, 3 * k))

            if not ch_sel:
                ch_sel = list(range(k))  # safety fallback

            reduced_arr = arr[ch_sel, :, :]
            tmp_path = str(
                Path(out_dir) / f"{entry['subject_id']}_{'_'.join(combo)}.npy"
            )
            np.save(tmp_path, reduced_arr.astype(np.float32))
            new_entry = {**entry, "fused_npy": tmp_path}
            reduced.append(new_entry)
        return reduced

    # ------------------------------------------------------------------
    # A) Modality ablation (all use attention-late fusion where applicable)
    # ------------------------------------------------------------------
    modality_combos = [
        (["T1"], "T1_only"),
        (["NM"], "NM_only"),
        (["ASL"], "ASL_tab_only"),  # tabular MLP only
        (["T1", "NM"], "T1_NM"),
        (["T1", "ASL"], "T1_ASL"),
        (["NM", "ASL"], "NM_ASL"),
        (["T1", "NM", "ASL"], "T1_NM_ASL"),
    ]

    for combo, exp_name in modality_combos:
        logger.info("=" * 60)
        logger.info("Modality ablation: %s", exp_name)

        if exp_name == "ASL_tab_only":
            # Tabular-only model (MLP on CBF features + age)
            factory = lambda: UnimodalTabularModel(num_tab=num_tab)
        else:
            img_ch = len(combo) * k
            if len(combo) == 1:
                factory = lambda _c=img_ch: UnimodalImageModel(in_channels=_c)
            else:
                factory = lambda _c=img_ch: HybridFusionModel(
                    img_channels=_c, num_tab_features=num_tab,
                    d_model=cfg["TAB_D_MODEL"], num_heads=cfg["TAB_NUM_HEADS"],
                    d_ff=cfg["TAB_D_FF"],
                    dropout_cls=cfg["DROPOUT_CLASSIFIER"],
                    dropout_tab=cfg["DROPOUT_TAB"],
                )

        reduced_m = reduce_manifest(combo) if exp_name != "ASL_tab_only" else manifest
        result = run_cross_validation(
            reduced_m, factory, cfg,
            experiment_name=exp_name, out_dir=out_dir,
        )
        row = {
            "experiment": exp_name,
            "modalities": "+".join(combo),
            "fusion": "attention-late" if len(combo) > 1 else "unimodal",
            "backbone": "ResNet18-CBAM",
            "pooled_auc": result["pooled_auc"],
            "mean_auc": result["mean_fold_auc"],
            "std_auc": result["std_fold_auc"],
            "brier_calib": result["brier_after_calibration"],
            "threshold": result["youden_threshold"],
        }
        all_results.append(row)

    # ------------------------------------------------------------------
    # B) Fusion strategy comparison (on full T1+NM+ASL)
    # ------------------------------------------------------------------
    full_manifest = reduce_manifest(["T1", "NM", "ASL"])
    img_ch_full = 3 * k

    fusion_variants = [
        ("early_fusion", lambda: EarlyFusionModel(img_ch_full, num_tab)),
        ("standard_late", lambda: StandardLateFusionModel(img_ch_full, num_tab)),
        # attention-late already run above as T1_NM_ASL
    ]
    for fname, factory in fusion_variants:
        logger.info("Fusion ablation: %s", fname)
        result = run_cross_validation(
            full_manifest, factory, cfg,
            experiment_name=fname, out_dir=out_dir,
        )
        all_results.append({
            "experiment": fname,
            "modalities": "T1+NM+ASL",
            "fusion": fname,
            "backbone": "ResNet18-CBAM",
            "pooled_auc": result["pooled_auc"],
            "mean_auc": result["mean_fold_auc"],
            "std_auc": result["std_fold_auc"],
            "brier_calib": result["brier_after_calibration"],
            "threshold": result["youden_threshold"],
        })

    # ------------------------------------------------------------------
    # C) Backbone comparison (attention-late, full T1+NM+ASL)
    # ------------------------------------------------------------------
    for bname in ["resnet34", "densenet121", "efficientnet_b3"]:
        logger.info("Backbone ablation: %s", bname)

        def factory(_bn=bname):
            backbone, edim = _make_backbone(_bn, img_ch_full)
            return FlexibleHybridModel(
                backbone, edim, num_tab,
                d_model=cfg["TAB_D_MODEL"], num_heads=cfg["TAB_NUM_HEADS"],
                d_ff=cfg["TAB_D_FF"],
            )

        result = run_cross_validation(
            full_manifest, factory, cfg,
            experiment_name=f"backbone_{bname}", out_dir=out_dir,
        )
        all_results.append({
            "experiment": f"backbone_{bname}",
            "modalities": "T1+NM+ASL",
            "fusion": "attention-late",
            "backbone": bname,
            "pooled_auc": result["pooled_auc"],
            "mean_auc": result["mean_fold_auc"],
            "std_auc": result["std_fold_auc"],
            "brier_calib": result["brier_after_calibration"],
            "threshold": result["youden_threshold"],
        })

    # ------------------------------------------------------------------
    # Save aggregated ablation table
    # ------------------------------------------------------------------
    df = pd.DataFrame(all_results)
    csv_path = Path(out_dir) / "ablation_results.csv"
    df.to_csv(str(csv_path), index=False)
    logger.info("Ablation results saved to %s", csv_path)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — EXPLAINABILITY (STAGES 6–7)
# ═══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# 11.1  Multi-Layer Grad-CAM with Gamma Correction
# ---------------------------------------------------------------------------
class MultiLayerGradCAM:
    """
    Grad-CAM with multi-layer aggregation and gamma correction.

    Paper §2.3 Stage 6: "Grad-CAM was computed from the last convolutional
    block, with multi-layer aggregation and mild gamma correction."

    Usage:
        cam = MultiLayerGradCAM(model.image_branch, target_layers=[...])
        heatmap = cam(input_img, class_idx=0)
    """

    def __init__(
            self,
            backbone: nn.Module,
            target_layers: Optional[List[nn.Module]] = None,
            gamma: float = 0.5,
    ):
        self.backbone = backbone
        self.gamma = gamma

        # Default: use last 2 stages of ResNet (layer3, layer4)
        if target_layers is None:
            if hasattr(backbone, "layer3") and hasattr(backbone, "layer4"):
                target_layers = [backbone.layer3, backbone.layer4]
            elif hasattr(backbone, "layer4"):
                target_layers = [backbone.layer4]
            else:
                raise ValueError("Cannot auto-detect target layers. Pass them explicitly.")

        self.target_layers = target_layers
        self._activations: Dict[int, torch.Tensor] = {}
        self._gradients: Dict[int, torch.Tensor] = {}
        self._hooks: List = []
        self._register_hooks()

    def _register_hooks(self):
        for idx, layer in enumerate(self.target_layers):
            def fwd_hook(module, inp, out, _idx=idx):
                self._activations[_idx] = out.detach()

            def bwd_hook(module, grad_in, grad_out, _idx=idx):
                self._gradients[_idx] = grad_out[0].detach()

            self._hooks.append(layer.register_forward_hook(fwd_hook))
            self._hooks.append(layer.register_full_backward_hook(bwd_hook))

    def __call__(
            self, input_tensor: torch.Tensor, class_idx: int = 0,
    ) -> np.ndarray:
        """
        input_tensor : (1, C, H, W) on model device
        Returns numpy heatmap (H, W) in [0, 1].
        """
        self.backbone.zero_grad()
        out = self.backbone(input_tensor)  # (1, 512) for ResNet18CBAM
        # For binary: just use the single logit
        if out.dim() == 1:
            score = out[0]
        elif out.shape[1] == 1:
            score = out[0, 0]
        else:
            score = out[0, class_idx]
        score.backward(retain_graph=True)

        cams = []
        for idx in range(len(self.target_layers)):
            act = self._activations[idx]  # (1, C_f, H_f, W_f)
            grad = self._gradients[idx]  # same shape
            weights = grad.mean(dim=(2, 3), keepdim=True)  # GAP of gradients
            cam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, H_f, W_f)
            cam = F.relu(cam)
            # Upsample to input spatial size
            cam = F.interpolate(
                cam, size=input_tensor.shape[-2:],
                mode="bilinear", align_corners=False,
            )
            cams.append(cam)

        # Multi-layer aggregation: element-wise mean
        agg_cam = torch.stack(cams).mean(dim=0)  # (1, 1, H, W)
        agg_cam = agg_cam.squeeze().cpu().numpy()

        # Gamma correction (paper: mild gamma ≈ 0.5)
        agg_cam = (agg_cam - agg_cam.min()) / (agg_cam.max() - agg_cam.min() + 1e-8)
        agg_cam = np.power(agg_cam, self.gamma)

        return agg_cam

    def close(self):
        for h in self._hooks:
            h.remove()


# ---------------------------------------------------------------------------
# 11.2  Integrated Gradients (via Captum)
# ---------------------------------------------------------------------------
def compute_integrated_gradients(
        model: nn.Module,
        img_tensor: torch.Tensor,
        tab_tensor: torch.Tensor,
        device: str,
        n_steps: int = 50,
) -> Optional[np.ndarray]:
    """
    Compute pixel-wise Integrated Gradients attribution (Sundararajan et al. 2017).
    Paper §2.3 Stage 6: "Integrated Gradients (50 steps) were computed from a
    zero-baseline using Captum."
    Returns attribution map (H, W) or None if Captum unavailable.
    """
    if not CAPTUM_AVAILABLE:
        logger.warning("Captum not installed — Integrated Gradients skipped.")
        return None

    model.eval()
    img_tensor = img_tensor.to(device).requires_grad_(True)
    tab_tensor = tab_tensor.to(device)

    # Wrap model to accept only the image (fix tabular)
    def forward_img(img):
        return model(img, tab_tensor.expand(img.shape[0], -1)).unsqueeze(-1)

    ig = IntegratedGradients(forward_img)
    baseline = torch.zeros_like(img_tensor)
    attr = ig.attribute(img_tensor, baselines=baseline, n_steps=n_steps)
    # Aggregate across channels → (H, W)
    attr_map = attr.squeeze(0).abs().mean(dim=0).cpu().numpy()
    attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
    return attr_map


# ---------------------------------------------------------------------------
# 11.3  KernelSHAP for Tabular Features
# ---------------------------------------------------------------------------
def compute_tabular_shap(
        model: nn.Module,
        manifest: List[Dict[str, Any]],
        device: str,
        feature_names: List[str],
        out_dir: str,
        n_background: int = 20,
) -> Optional[np.ndarray]:
    """
    KernelSHAP analysis on the tabular branch / fused decision.
    Paper §2.3 Stage 7: "KernelSHAP with a background set of randomly
    sampled HC and PD subjects was used to estimate feature attributions."
    Returns SHAP values matrix (N_subjects, N_features) or None.
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not installed — tabular SHAP analysis skipped.")
        return None

    model.eval()
    ensure_dir(out_dir)

    # Collect all tabular vectors and a dummy image for each subject
    all_tab = np.stack([np.array(e["tabular"], dtype=np.float32) for e in manifest])
    # For SHAP we need a prediction function on tabular inputs;
    # use a mean image as the image input (frozen) to isolate tabular contribution
    sample_img = np.load(manifest[0]["fused_npy"]).astype(np.float32)
    mean_img = torch.from_numpy(sample_img).unsqueeze(0).to(device)

    def predict_tab(tab_np: np.ndarray) -> np.ndarray:
        tab_t = torch.from_numpy(tab_np.astype(np.float32)).to(device)
        img_batch = mean_img.expand(tab_t.shape[0], -1, -1, -1)
        with torch.no_grad():
            logits = model(img_batch, tab_t)
        return torch.sigmoid(logits).cpu().numpy()

    # Background set
    bg_idx = np.random.choice(len(all_tab), size=min(n_background, len(all_tab)), replace=False)
    background = all_tab[bg_idx]

    explainer = shap.KernelExplainer(predict_tab, background)
    shap_values = explainer.shap_values(all_tab, nsamples=200)

    # Save beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, all_tab, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / "shap_beeswarm.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Save bar plot (mean |SHAP|)
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, all_tab, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / "shap_bar.png"), dpi=200, bbox_inches="tight")
    plt.close()

    logger.info("SHAP plots saved to %s", out_dir)
    return shap_values


# ---------------------------------------------------------------------------
# 11.4  Quantitative XAI Validation: Fidelity & Reliability
# ---------------------------------------------------------------------------
def compute_xai_fidelity_reliability(
        saliency_maps: Dict[str, np.ndarray],
        atlas_native_paths: Dict[str, str],
        labels: Dict[str, int],
        roi_labels: Dict[str, int],
        out_dir: str,
        top_quantile: float = 0.90,
) -> pd.DataFrame:
    """
    Paper §2.3 Quantitative XAI Validation:

    Fidelity (ROI coverage):
      Threshold saliency map at top quantile → compute fraction of
      thresholded voxels within each ROI.  Compare PD vs HC.

    Reliability (Stability Index):
      RI = μ / σ  of saliency coverage across subjects per group per ROI.
      RI > 2 → relatively stable.

    Returns DataFrame with per-subject per-ROI coverage and group summaries.
    """
    ensure_dir(out_dir)
    records = []

    for subj_id, sal in saliency_maps.items():
        atlas_path = atlas_native_paths.get(subj_id)
        if atlas_path is None or not Path(atlas_path).exists():
            continue
        atlas, _, _ = load_nifti(atlas_path)

        # Threshold saliency at top quantile
        thresh = np.quantile(sal[sal > 0], top_quantile) if (sal > 0).any() else 0.0
        binary_sal = (sal >= thresh).astype(np.uint8)

        for roi_name, roi_idx in roi_labels.items():
            roi_mask = (atlas == roi_idx)
            roi_size = roi_mask.sum()
            if roi_size == 0:
                continue
            overlap = np.logical_and(binary_sal, roi_mask).sum()
            coverage = float(overlap) / float(roi_size)
            records.append({
                "subject_id": subj_id,
                "label": labels[subj_id],
                "group": "PD" if labels[subj_id] == 1 else "HC",
                "ROI": roi_name,
                "coverage": coverage,
            })

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("No XAI fidelity data produced (atlas unavailable?).")
        return df

    # Group summary: mean, std, reliability index
    summary = df.groupby(["group", "ROI"]).agg(
        mean_coverage=("coverage", "mean"),
        std_coverage=("coverage", "std"),
        n=("coverage", "count"),
    ).reset_index()
    summary["reliability_index"] = summary["mean_coverage"] / (summary["std_coverage"] + 1e-8)

    summary.to_csv(str(Path(out_dir) / "xai_fidelity_reliability.csv"), index=False)
    logger.info("XAI fidelity/reliability table saved.")

    # Simple bar chart: PD vs HC coverage per ROI
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    rois = sorted(df["ROI"].unique())
    x = np.arange(len(rois))
    width = 0.35
    for gi, grp in enumerate(["HC", "PD"]):
        means = [
            summary.loc[(summary.group == grp) & (summary.ROI == r), "mean_coverage"].values
            for r in rois
        ]
        means = [float(m[0]) if len(m) > 0 else 0.0 for m in means]
        ax.bar(x + gi * width, means, width, label=grp, alpha=0.8)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(rois, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean saliency coverage")
    ax.set_title("Fidelity: ROI coverage by group")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / "xai_fidelity_barplot.png"), dpi=200)
    plt.close()

    return df


# ---------------------------------------------------------------------------
# 11.5  Per-Subject XAI Dashboard (Figure 7 in paper)
# ---------------------------------------------------------------------------
def save_subject_xai_panel(
        subj_id: str,
        img_2d: np.ndarray,
        gradcam: np.ndarray,
        ig_map: Optional[np.ndarray],
        shap_vals: Optional[np.ndarray],
        feature_names: List[str],
        pred_prob: float,
        true_label: int,
        out_dir: str,
) -> None:
    """
    Create a composite XAI dashboard for one subject showing:
      • Original slice • Grad-CAM overlay • IG overlay • SHAP bar chart
    """
    ensure_dir(out_dir)
    n_cols = 3 + (1 if shap_vals is not None else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    # Normalise image for display
    disp = img_2d.copy()
    disp = (disp - np.percentile(disp, 2)) / (np.percentile(disp, 98) - np.percentile(disp, 2) + 1e-8)
    disp = np.clip(disp, 0, 1)

    # Panel 1: original
    axes[0].imshow(disp.T, cmap="gray", origin="lower")
    axes[0].set_title(f"{subj_id}\nLabel={'PD' if true_label else 'HC'}  P(PD)={pred_prob:.2f}")
    axes[0].axis("off")

    # Panel 2: Grad-CAM overlay
    axes[1].imshow(disp.T, cmap="gray", origin="lower")
    axes[1].imshow(gradcam.T, cmap="jet", alpha=0.4, origin="lower")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    # Panel 3: Integrated Gradients
    ig_disp = ig_map if ig_map is not None else np.zeros_like(gradcam)
    axes[2].imshow(disp.T, cmap="gray", origin="lower")
    axes[2].imshow(ig_disp.T, cmap="hot", alpha=0.4, origin="lower")
    axes[2].set_title("Integrated Gradients")
    axes[2].axis("off")

    # Panel 4: SHAP feature attributions (bar)
    if shap_vals is not None and n_cols > 3:
        ax_shap = axes[3]
        y_pos = np.arange(len(feature_names))
        ax_shap.barh(y_pos, shap_vals, color=["#d62728" if v > 0 else "#1f77b4" for v in shap_vals])
        ax_shap.set_yticks(y_pos)
        ax_shap.set_yticklabels(feature_names, fontsize=7)
        ax_shap.set_xlabel("SHAP value")
        ax_shap.set_title("Feature Attribution")

    plt.suptitle(f"XAI Dashboard — {subj_id}", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / f"{subj_id}_xai_panel.png"), dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — VISUALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(
        histories: List[Dict[str, list]],
        out_path: str,
) -> None:
    """
    Plot training vs validation loss and accuracy across folds (Figure 2).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, h in enumerate(histories):
        epochs = range(1, len(h["train_loss"]) + 1)
        ax1.plot(epochs, h["train_loss"], "--", alpha=0.5, label=f"Fold {i} train")
        ax1.plot(epochs, h["val_loss"], "-", alpha=0.8, label=f"Fold {i} val")
    ax1.set_xlabel("Epoch");
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs Validation Loss");
    ax1.legend(fontsize=7)

    for i, h in enumerate(histories):
        epochs = range(1, len(h["val_auc"]) + 1)
        ax2.plot(epochs, h["val_auc"], "-", alpha=0.8, label=f"Fold {i}")
    ax2.set_xlabel("Epoch");
    ax2.set_ylabel("Validation AUC")
    ax2.set_title("Validation AUC per Fold");
    ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()


def plot_roc_and_pr(
        labels: np.ndarray,
        probs: np.ndarray,
        calibrated_probs: np.ndarray,
        threshold: float,
        out_path: str,
) -> None:
    """
    Plot ROC curve, precision–recall curve, calibrated probability histogram,
    and confusion matrix (Figure 5).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ROC
    fpr, tpr, _ = roc_curve(labels, calibrated_probs)
    roc_auc = roc_auc_score(labels, calibrated_probs)
    axes[0, 0].plot(fpr, tpr, "b-", lw=2, label=f"AUC = {roc_auc:.3f}")
    # Mark Youden point
    best_idx = int(np.argmax(tpr - fpr))
    axes[0, 0].plot(fpr[best_idx], tpr[best_idx], "ro", markersize=10, label=f"Youden (t={threshold:.3f})")
    axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0, 0].set_xlabel("FPR");
    axes[0, 0].set_ylabel("TPR")
    axes[0, 0].set_title("ROC Curve");
    axes[0, 0].legend()

    # Precision-Recall
    prec_arr, rec_arr, _ = precision_recall_curve(labels, calibrated_probs)
    ap = auc(rec_arr, prec_arr)
    axes[0, 1].plot(rec_arr, prec_arr, "g-", lw=2, label=f"AP = {ap:.3f}")
    axes[0, 1].set_xlabel("Recall");
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title("Precision–Recall Curve");
    axes[0, 1].legend()

    # Confusion matrix
    pred_labels = (calibrated_probs >= threshold).astype(int)
    tn = int(((pred_labels == 0) & (labels == 0)).sum())
    fp = int(((pred_labels == 1) & (labels == 0)).sum())
    fn = int(((pred_labels == 0) & (labels == 1)).sum())
    tp = int(((pred_labels == 1) & (labels == 1)).sum())
    cm = np.array([[tn, fp], [fn, tp]])
    axes[1, 0].imshow(cm, cmap="Blues", vmin=0)
    for (i, j), val in np.ndenumerate(cm):
        axes[1, 0].text(j, i, str(val), ha="center", va="center", fontsize=16,
                        color="white" if val > cm.max() / 2 else "black")
    axes[1, 0].set_xticks([0, 1]);
    axes[1, 0].set_xticklabels(["HC", "PD"])
    axes[1, 0].set_yticks([0, 1]);
    axes[1, 0].set_yticklabels(["HC", "PD"])
    axes[1, 0].set_xlabel("Predicted");
    axes[1, 0].set_ylabel("True")
    axes[1, 0].set_title(f"Confusion Matrix (t={threshold:.3f})")

    # Calibrated probability distribution
    hc_probs = calibrated_probs[labels == 0]
    pd_probs = calibrated_probs[labels == 1]
    axes[1, 1].hist(hc_probs, bins=15, alpha=0.6, label="HC", color="blue")
    axes[1, 1].hist(pd_probs, bins=15, alpha=0.6, label="PD", color="red")
    axes[1, 1].axvline(threshold, color="k", ls="--", label=f"Threshold={threshold:.3f}")
    axes[1, 1].set_xlabel("Calibrated P(PD)");
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Calibrated Probability Distribution");
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — SINGLE-SUBJECT PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_single_subject(
        subj_id: str,
        t1_path: Optional[str],
        nm_path: Optional[str],
        asl_path: Optional[str],
        m0_path: Optional[str],
        age: float,
        out_root: str,
        cfg: dict,
) -> Dict[str, Any]:
    """
    Run the full per-subject preprocessing pipeline (paper §2.2 steps 1–10).
    Returns a manifest-compatible dict with all paths and features.
    """
    subdir = ensure_dir(Path(out_root) / subj_id)
    result: Dict[str, Any] = {"subject_id": subj_id, "age": age}
    atlas_mni_path = locate_fsl_atlas()

    # ------------------------------------------------------------------
    # Step 1: QC (log header info; visual QC is manual)
    # ------------------------------------------------------------------
    if t1_path:
        t1_data, t1_aff, t1_hdr = load_nifti(t1_path)
        logger.info(
            "[QC] %s T1: shape=%s voxsize=%s",
            subj_id, t1_data.shape, t1_hdr.get_zooms()[:3],
        )
    if nm_path:
        nm_data, _, nm_hdr = load_nifti(nm_path)
        logger.info("[QC] %s NM: shape=%s", subj_id, nm_data.shape)
    if asl_path:
        asl_data, _, asl_hdr = load_nifti(asl_path)
        logger.info("[QC] %s ASL: shape=%s", subj_id, asl_data.shape)

    # ------------------------------------------------------------------
    # Step 2: N4 Bias-Field Correction (all modalities)
    # ------------------------------------------------------------------
    t1_n4 = nm_n4 = asl_n4 = None

    if t1_path:
        t1_n4 = str(subdir / f"{subj_id}_T1_N4.nii.gz")
        try:
            run_n4_correction(t1_path, t1_n4, shrink=cfg["N4_SHRINKFACTOR"])
        except Exception as e:
            logger.warning("T1 N4 failed: %s — using original", e)
            shutil.copy(t1_path, t1_n4)

    if nm_path:
        nm_n4 = str(subdir / f"{subj_id}_NM_N4.nii.gz")
        try:
            run_n4_correction(nm_path, nm_n4, shrink=cfg["N4_SHRINKFACTOR"])
        except Exception as e:
            logger.warning("NM N4 failed: %s", e)
            shutil.copy(nm_path, nm_n4)

    if asl_path:
        asl_n4 = str(subdir / f"{subj_id}_ASL_N4.nii.gz")
        try:
            run_n4_correction(asl_path, asl_n4, shrink=cfg["N4_SHRINKFACTOR"])
        except Exception as e:
            logger.warning("ASL N4 failed: %s", e)
            shutil.copy(asl_path, asl_n4)

    # ------------------------------------------------------------------
    # Step 3: Brain Extraction (HD-BET or Otsu fallback on T1)
    # ------------------------------------------------------------------
    mask_path = brain_path = None
    if t1_n4:
        mask_path, brain_path = brain_extraction(t1_n4, str(subdir), subj_id, cfg)
    result["mask_path"] = mask_path
    result["brain_path"] = brain_path

    # ------------------------------------------------------------------
    # Step 4: Intra-Subject Rigid Registration (NM → T1, ASL → T1)
    # ------------------------------------------------------------------
    nm_reg = asl_reg = None
    if nm_n4 and t1_n4:
        nm_reg = str(subdir / f"{subj_id}_NM_reg.nii.gz")
        rigid_register_to_t1(nm_n4, t1_n4, nm_reg)

    if asl_n4 and t1_n4:
        asl_reg = str(subdir / f"{subj_id}_ASL_reg.nii.gz")
        rigid_register_to_t1(asl_n4, t1_n4, asl_reg)

    # ------------------------------------------------------------------
    # Step 5: MNI Registration + Atlas Propagation
    # ------------------------------------------------------------------
    mni_result = {}
    if brain_path:
        mni_result = register_to_mni_and_propagate_atlas(
            brain_path, str(subdir), subj_id,
            atlas_mni_path=str(atlas_mni_path) if atlas_mni_path else None,
        )
    result["mni"] = mni_result

    # Load atlas in native space (if available)
    atlas_native = None
    if "atlas_native" in mni_result and Path(mni_result["atlas_native"]).exists():
        atlas_native, _, _ = load_nifti(mni_result["atlas_native"])

    # ------------------------------------------------------------------
    # Step 6: ASL CBF Quantification (BASIL or fallback)
    # ------------------------------------------------------------------
    cbf_path = None
    if asl_reg or asl_n4:
        asl_input = asl_reg if asl_reg else asl_n4
        cbf_path = run_basil_cbf(
            asl_input, m0_path, None, str(subdir), subj_id, cfg,
        )
        if cbf_path is None:
            cbf_path = simple_cbf_fallback(asl_input, str(subdir), subj_id)
    result["cbf_path"] = cbf_path

    # ------------------------------------------------------------------
    # Step 7: Subject-wise Z-Score Normalisation
    # ------------------------------------------------------------------
    mask_data = None
    if mask_path:
        mask_data, _, _ = load_nifti(mask_path)

    t1_z = nm_z = cbf_z = None

    if brain_path and mask_data is not None:
        t1_vol, t1_aff, _ = load_nifti(brain_path)
        t1_vol = zscore_normalise(t1_vol, mask_data)
        t1_z = str(subdir / f"{subj_id}_T1_z.nii.gz")
        save_nifti(t1_vol, t1_aff, t1_z)

    if nm_reg and mask_data is not None:
        nm_vol, nm_aff, _ = load_nifti(nm_reg)
        nm_vol = zscore_normalise(nm_vol, mask_data)
        nm_z = str(subdir / f"{subj_id}_NM_z.nii.gz")
        save_nifti(nm_vol, nm_aff, nm_z)
    elif nm_n4 and mask_data is not None:
        nm_vol, nm_aff, _ = load_nifti(nm_n4)
        nm_vol = zscore_normalise(nm_vol, mask_data)
        nm_z = str(subdir / f"{subj_id}_NM_z.nii.gz")
        save_nifti(nm_vol, nm_aff, nm_z)

    if cbf_path and mask_data is not None:
        cbf_vol, cbf_aff, _ = load_nifti(cbf_path)
        cbf_vol = zscore_normalise(cbf_vol, mask_data)
        cbf_z = str(subdir / f"{subj_id}_CBF_z.nii.gz")
        save_nifti(cbf_vol, cbf_aff, cbf_z)

    result["t1_z"] = t1_z
    result["nm_z"] = nm_z
    result["cbf_z"] = cbf_z

    # ------------------------------------------------------------------
    # Step 8: NM-MRI Biomarker Extraction (SNR, CNR)
    # ------------------------------------------------------------------
    nm_vol_for_metrics = None
    if nm_z:
        nm_vol_for_metrics, _, _ = load_nifti(nm_z)
    elif nm_n4:
        nm_vol_for_metrics, _, _ = load_nifti(nm_n4)

    nm_metrics = {}
    if nm_vol_for_metrics is not None:
        nm_metrics = compute_nm_metrics(
            nm_vol_for_metrics, atlas_native, cfg["ATLAS_ROIS"],
        )
    result["nm_metrics"] = nm_metrics

    # ------------------------------------------------------------------
    # Step 9: Regional CBF Extraction
    # ------------------------------------------------------------------
    cbf_vol_for_metrics = None
    if cbf_z:
        cbf_vol_for_metrics, _, _ = load_nifti(cbf_z)
    elif cbf_path:
        cbf_vol_for_metrics, _, _ = load_nifti(cbf_path)

    cbf_metrics = {}
    if cbf_vol_for_metrics is not None:
        cbf_metrics = extract_regional_cbf(
            cbf_vol_for_metrics, atlas_native, cfg["ATLAS_ROIS"],
        )
    result["cbf_metrics"] = cbf_metrics

    # ------------------------------------------------------------------
    # Step 10: ROI-Aware Slice Selection & 2.5D Formation
    # ------------------------------------------------------------------
    t1_vol_final = np.zeros((1, 1, 1), dtype=np.float32)
    if t1_z:
        t1_vol_final, _, _ = load_nifti(t1_z)
    elif brain_path:
        t1_vol_final, _, _ = load_nifti(brain_path)

    nm_vol_final = None
    if nm_z:
        nm_vol_final, _, _ = load_nifti(nm_z)

    cbf_vol_final = None
    if cbf_z:
        cbf_vol_final, _, _ = load_nifti(cbf_z)

    roi_indices = list(cfg["ATLAS_ROIS"].values())
    z_indices = select_axial_slices(
        mask_data if mask_data is not None else np.ones_like(t1_vol_final),
        k=cfg["NUM_SLICES"],
        atlas_native=atlas_native,
        roi_indices=roi_indices,
    )
    result["slice_indices"] = z_indices.tolist()

    fused = form_multimodal_input(t1_vol_final, nm_vol_final, cbf_vol_final, z_indices)
    fused_path = str(subdir / f"{subj_id}_fused.npy")
    np.save(fused_path, fused.astype(np.float32))
    result["fused_npy"] = fused_path

    # ------------------------------------------------------------------
    # Step 11: Tabular Feature Vector Assembly
    # ------------------------------------------------------------------
    tabular = assemble_tabular_features(
        nm_metrics, cbf_metrics, age, cfg["TABULAR_FEATURE_NAMES"],
    )
    result["tabular"] = tabular.tolist()

    # ------------------------------------------------------------------
    # Save per-subject summary
    # ------------------------------------------------------------------
    summary_json = {
        k: v for k, v in result.items()
        if not isinstance(v, (np.ndarray,))
    }
    with open(subdir / f"{subj_id}_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2, default=str)

    logger.info("Subject %s preprocessing complete → %s", subj_id, subdir)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — BATCH DISCOVERY AND ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def discover_subjects(input_root: str) -> List[Dict[str, Any]]:
    """
    Walk input_root for subject folders under Healthy/ and Parkinson/ categories.
    Expected layout:
        input_root/
          Healthy/
            SUBJ01/  ← contains T1*.nii*, NM*.nii*, ASL*.nii*, optionally M0*
          Parkinson/
            SUBJ02/ ...
    Returns list of dicts with subject_id, label, t1, nm, asl, m0, age.
    """
    root = Path(input_root)
    subjects = []
    for category, label in [("Healthy", 0), ("Parkinson", 1)]:
        cat_dir = root / category
        if not cat_dir.exists():
            logger.warning("Category dir not found: %s", cat_dir)
            continue
        for subj_dir in sorted(cat_dir.iterdir()):
            if not subj_dir.is_dir():
                continue

            def _find(patterns):
                for p in patterns:
                    hits = list(subj_dir.rglob(p))
                    if hits:
                        return str(hits[0])
                return None

            t1 = _find(["*T1*.[nN]*", "*t1*.[nN]*"])
            nm = _find(["*NM*.[nN]*", "*nm*.[nN]*", "*neuromelanin*.[nN]*"])
            asl_f = _find(["*ASL*.[nN]*", "*asl*.[nN]*", "*pcasl*.[nN]*"])
            m0 = _find(["*M0*.[nN]*", "*m0*.[nN]*"])

            # Try to read age from a demographics JSON/CSV if present
            age = 0.0
            demo_files = list(subj_dir.rglob("*demo*")) + list(subj_dir.rglob("*meta*"))
            for df_path in demo_files:
                try:
                    if str(df_path).endswith(".json"):
                        with open(df_path) as jf:
                            d = json.load(jf)
                            age = float(d.get("age", d.get("Age", 0)))
                    elif str(df_path).endswith(".csv"):
                        ddf = pd.read_csv(str(df_path))
                        if "age" in ddf.columns or "Age" in ddf.columns:
                            col = "age" if "age" in ddf.columns else "Age"
                            age = float(ddf[col].iloc[0])
                except Exception:
                    pass

            subjects.append({
                "subject_id": subj_dir.name,
                "label": label,
                "t1": t1, "nm": nm, "asl": asl_f, "m0": m0,
                "age": age,
            })
    logger.info("Discovered %d subjects (%d HC, %d PD) in %s",
                len(subjects),
                sum(1 for s in subjects if s["label"] == 0),
                sum(1 for s in subjects if s["label"] == 1),
                input_root)
    return subjects


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — MAIN CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "MultiMRI-PD: 7-stage multimodal MRI pipeline for explainable "
            "Parkinson's disease diagnosis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["single", "batch"], default="single",
                        help="'single' for one subject, 'batch' for full cohort.")
    # Single-mode paths
    parser.add_argument("--t1", type=str, default=None)
    parser.add_argument("--nm", type=str, default=None)
    parser.add_argument("--asl", type=str, default=None)
    parser.add_argument("--asl-m0", type=str, default=None, help="M0 calibration image for ASL")
    parser.add_argument("--subject-id", type=str, default="SUBJ")
    parser.add_argument("--age", type=float, default=0.0)
    # Batch-mode
    parser.add_argument("--input-root", type=str, default="")
    # Output
    parser.add_argument("--output-root", type=str, default=DEFAULT_CONFIG["OUTPUT_ROOT"])
    # Stage control
    parser.add_argument(
        "--stages", type=str, default="1,2,3,4,5,6,7",
        help="Comma-separated stage numbers to run (1-5=preproc, 6-7=XAI).",
    )
    # Overrides
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["MAX_EPOCHS"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["BATCH_SIZE"])
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["DEVICE"])
    parser.add_argument("--num-folds", type=int, default=DEFAULT_CONFIG["NUM_FOLDS"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["LR"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["PATIENCE"])

    args = parser.parse_args()

    # Merge CLI into config
    cfg = DEFAULT_CONFIG.copy()
    cfg["MAX_EPOCHS"] = args.epochs
    cfg["BATCH_SIZE"] = args.batch_size
    cfg["DEVICE"] = args.device
    cfg["NUM_FOLDS"] = args.num_folds
    cfg["LR"] = args.lr
    cfg["PATIENCE"] = args.patience

    stages = {int(s.strip()) for s in args.stages.split(",") if s.strip().isdigit()}
    ensure_dir(args.output_root)

    # ==================================================================
    # SINGLE-SUBJECT MODE
    # ==================================================================
    if args.mode == "single":
        entry = preprocess_single_subject(
            subj_id=args.subject_id,
            t1_path=args.t1,
            nm_path=args.nm,
            asl_path=args.asl,
            m0_path=args.asl_m0,
            age=args.age,
            out_root=args.output_root,
            cfg=cfg,
        )
        logger.info("Single-subject preprocessing complete.")
        if 6 in stages or 7 in stages:
            logger.info(
                "Stages 6-7 (training/explainability) require batch mode with labels. "
                "Run with --mode batch for full pipeline."
            )
        return

    # ==================================================================
    # BATCH MODE
    # ==================================================================
    # ---- Preprocessing (Stages 1–5) ----
    subjects = discover_subjects(args.input_root)
    if not subjects:
        logger.error("No subjects found under %s. Check directory layout.", args.input_root)
        sys.exit(1)

    manifest: List[Dict[str, Any]] = []
    preproc_stages = stages.intersection({1, 2, 3, 4, 5})
    if preproc_stages:
        logger.info("Running preprocessing stages %s for %d subjects…",
                    sorted(preproc_stages), len(subjects))
        for s in subjects:
            entry = preprocess_single_subject(
                subj_id=s["subject_id"],
                t1_path=s["t1"], nm_path=s["nm"],
                asl_path=s["asl"], m0_path=s.get("m0"),
                age=s["age"],
                out_root=args.output_root, cfg=cfg,
            )
            entry["label"] = s["label"]
            manifest.append(entry)

        # Save manifest
        manifest_df = pd.DataFrame([
            {k: v for k, v in m.items() if
             not isinstance(v, (np.ndarray, dict, list)) or k in ("tabular", "slice_indices")}
            for m in manifest
        ])
        manifest_path = Path(args.output_root) / "manifest.csv"
        manifest_df.to_csv(str(manifest_path), index=False)
        logger.info("Manifest saved: %s", manifest_path)
    else:
        # Load existing manifest
        manifest_path = Path(args.output_root) / "manifest.csv"
        if manifest_path.exists():
            df = pd.read_csv(str(manifest_path))
            manifest = df.to_dict("records")
            # Parse tabular back from string if needed
            for m in manifest:
                if isinstance(m.get("tabular"), str):
                    m["tabular"] = json.loads(m["tabular"])
            logger.info("Loaded existing manifest with %d subjects.", len(manifest))
        else:
            logger.error("No manifest found. Run preprocessing stages first.")
            sys.exit(1)

    # ---- Training & Ablation (Stage 6 = paper Stages 1–5) ----
    if 6 in stages:
        logger.info("=" * 70)
        logger.info("STAGE 6: Ablation experiments and model training")
        logger.info("=" * 70)

        num_tab = len(cfg["TABULAR_FEATURE_NAMES"])
        img_ch = 3 * cfg["NUM_SLICES"]  # 15 for full 3-modality

        ablation_df = run_full_ablation(manifest, cfg, args.output_root)
        print("\n--- Ablation Results ---")
        print(ablation_df.to_string(index=False))

    # ---- Explainability (Stage 7 = paper Stages 6–7) ----
    if 7 in stages:
        logger.info("=" * 70)
        logger.info("STAGE 7: Explainability (Grad-CAM, IG, SHAP, Fidelity/Reliability)")
        logger.info("=" * 70)

        num_tab = len(cfg["TABULAR_FEATURE_NAMES"])
        img_ch = 3 * cfg["NUM_SLICES"]

        # Load best model from the T1+NM+ASL attention-late experiment
        best_ckpt = Path(args.output_root) / "T1_NM_ASL_fold0.pt"
        if not best_ckpt.exists():
            logger.warning(
                "No trained model checkpoint found (%s). "
                "Run Stage 6 first. Skipping explainability.", best_ckpt,
            )
        else:
            model = HybridFusionModel(
                img_channels=img_ch, num_tab_features=num_tab,
                d_model=cfg["TAB_D_MODEL"], num_heads=cfg["TAB_NUM_HEADS"],
                d_ff=cfg["TAB_D_FF"],
            )
            model.load_state_dict(torch.load(str(best_ckpt), map_location=cfg["DEVICE"]))
            model.to(cfg["DEVICE"])
            model.eval()

            xai_dir = ensure_dir(Path(args.output_root) / "explainability")

            # -- Grad-CAM & IG for each subject --
            cam_gen = MultiLayerGradCAM(
                model.get_image_branch(),
                gamma=cfg["GRADCAM_GAMMA"],
            )

            saliency_maps: Dict[str, np.ndarray] = {}
            atlas_native_paths: Dict[str, str] = {}
            subj_labels: Dict[str, int] = {}

            for entry in manifest:
                sid = entry["subject_id"]
                arr = np.load(entry["fused_npy"])  # (C, H, W)
                tab = np.array(entry["tabular"], dtype=np.float32)
                img_t = torch.from_numpy(arr).unsqueeze(0).to(cfg["DEVICE"])
                tab_t = torch.from_numpy(tab).unsqueeze(0).to(cfg["DEVICE"])

                # Grad-CAM
                gc_map = cam_gen(img_t)  # (H, W)
                saliency_maps[sid] = gc_map

                # Integrated Gradients
                ig_map = compute_integrated_gradients(
                    model, img_t, tab_t, cfg["DEVICE"], n_steps=cfg["IG_STEPS"],
                )

                # Get prediction probability
                with torch.no_grad():
                    prob = float(torch.sigmoid(model(img_t, tab_t)).cpu())

                # Per-subject XAI panel
                mid_slice = arr[arr.shape[0] // 2]  # middle channel as 2D display
                shap_entry = None  # per-subject SHAP computed globally below
                save_subject_xai_panel(
                    sid, mid_slice, gc_map, ig_map, shap_entry,
                    cfg["TABULAR_FEATURE_NAMES"], prob, int(entry["label"]),
                    str(xai_dir),
                )

                # Track for fidelity analysis
                subj_labels[sid] = int(entry["label"])
                mni_info = entry.get("mni", {})
                if isinstance(mni_info, dict) and "atlas_native" in mni_info:
                    atlas_native_paths[sid] = mni_info["atlas_native"]

            cam_gen.close()

            # -- Tabular SHAP --
            compute_tabular_shap(
                model, manifest, cfg["DEVICE"],
                cfg["TABULAR_FEATURE_NAMES"], str(xai_dir),
            )

            # -- Fidelity & Reliability --
            if atlas_native_paths:
                compute_xai_fidelity_reliability(
                    saliency_maps, atlas_native_paths, subj_labels,
                    cfg["ATLAS_ROIS"], str(xai_dir),
                )
            else:
                logger.warning(
                    "No atlas in native space available — fidelity/reliability metrics skipped."
                )

            # -- Performance summary plots --
            # Load pooled CV results from the best experiment
            summary_json = Path(args.output_root) / "T1_NM_ASL_summary.json"
            if summary_json.exists():
                with open(summary_json) as f:
                    cv_summary = json.load(f)
                # Reconstruct numpy arrays from saved npy if available
                npy_probs = Path(args.output_root) / "T1_NM_ASL_pooled_probs.npy"
                npy_labels = Path(args.output_root) / "T1_NM_ASL_pooled_labels.npy"
                if npy_probs.exists() and npy_labels.exists():
                    pooled_p = np.load(str(npy_probs))
                    pooled_l = np.load(str(npy_labels))
                    # Calibrate
                    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
                    iso.fit(pooled_p, pooled_l)
                    cal_p = iso.predict(pooled_p)
                    plot_roc_and_pr(
                        pooled_l, pooled_p, cal_p,
                        cv_summary.get("youden_threshold", 0.5),
                        str(xai_dir / "performance_summary.png"),
                    )

            logger.info("Explainability analysis complete → %s", xai_dir)

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()