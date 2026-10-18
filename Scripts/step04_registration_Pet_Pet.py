#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_registration_pet_pet.py
==========================
Recalage PET sujet -> PET de référence (MINE 3D), APPROCHE 1 : une matrice par sujet.

    T1, T2, ..., T80 : une matrice affine 4x4 propre à chaque sujet, apprise par MINE
    (Mutual Information Neural Estimation) puis appliquée au PET ORIGINAL (valeurs d'origine).

ENTRÉES
    data/Reference/PET_Ref_recale.nii.gz   image FIXE = PET de référence recalé (étape 02),
                                           grille MNI 193x229x193, valeurs d'origine
    data/RAW/*.nii(.gz)                    les 80 PET des sujets (400x400x110, 1.018x1.018x1.5 mm)
    data/interim/M_affine_4x4.npy          (optionnel) matrice de l'étape 02, sert de point de départ
    00_inspection.csv                      (optionnel) contrôle de cohérence de la géométrie

SORTIES
    data/registered/pet_pet/<id>_reg.nii.gz      PET du sujet recalé, valeurs d'origine (float32)
    data/interim/pet_pet/<id>_T.npy / _T.txt     matrice T_i (4x4)
    data/interim/pet_pet/<id>_result.json        paramètres, scores, métriques
    data/interim/pet_pet/T_all.npy               les N matrices (N,4,4), ordre = index 1..N
    data/interim/pet_pet/registration_summary.csv
    data/qc/figures/pet_pet/                     figures (référence, 80 sujets, matrices, bilan)
    data/qc/figures/pet_pet/subjects/            une figure "pipeline" par sujet (étapes 1 à 5)

CONVENTION DE T_i
    T_i envoie les coordonnées normalisées [-1,1] de la grille de la RÉFÉRENCE vers les
    coordonnées normalisées du PET du sujet (mis à la grille de la référence, comme à l'étape 02).
    Le PET recalé s'obtient en lisant le sujet en T_i·x pour chaque voxel x de la référence
    (c'est le sens attendu par grid_sample). T_i est donc comparable à la matrice M de l'étape 02.

TRAITEMENT D'UN SUJET (étapes illustrées dans la figure "pipeline")
    1. PET du sujet + PET de référence (même grille, même normalisation)
    2. Pyramide gaussienne multi-résolution (celle de la référence est calculée UNE fois)
    3. Optimisation de la similarité par MINE (grossier -> fin)
    4. Estimation de T_i
    5. Application de T_i au PET ORIGINAL

GAIN DE TEMPS
    Pour l'image fixe, calculés UNE SEULE FOIS : normalisation, pyramide, masque, voxels échantillonnés
    (contours + aléatoires) et leurs intensités. Seul le mobile change à chaque sujet.

ROBUSTESSE (positions et orientations initiales variables)
    - Départ multiple : M de l'étape 02, M de l'étape 02 + alignement des centres de masse,
      identité + centres de masse. On garde le candidat de meilleure NMI (échantillon, niveau grossier).
    - Régularisation autour du point de départ (et non autour de l'identité).
    - Paramètres bornés autour du point de départ.
    - Garde-fou : si le résultat est moins bon que le point de départ, on conserve le point de départ.
    - Reprise : un sujet déjà traité est sauté (sauf --force). Un sujet en échec n'arrête pas le lot.

USAGE
    python 04_registration_pet_pet.py --limit 3          # pilote : 3 sujets, à regarder AVANT le lot complet
    python 04_registration_pet_pet.py                    # les 80 sujets (reprend où il s'est arrêté)
    python 04_registration_pet_pet.py --subjects 200011432 976017 --force
    python 04_registration_pet_pet.py --group-only       # refait seulement les figures/CSV de groupe

LIMITES CONNUES
    - Modèle affine à 9 DOF (translations, rotations, échelles) : pas de déformation locale.
    - Pensé pour des rotations initiales modérées (de l'ordre de 20 degrés autour de chaque axe).
    - Le prétraitement des sujets (flip éventuel + étirement vers la grille de la référence) DOIT être
      identique à celui de la référence à l'étape 02 : voir CFG.flip_subject_y.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("Agg")                       # aucune fenêtre : les figures sont écrites sur disque
import matplotlib.pyplot as plt             # noqa: E402

import torch                                # noqa: E402
import torch.nn as nn                       # noqa: E402
import torch.nn.functional as F             # noqa: E402
import torch.optim as optim                 # noqa: E402

from scipy import ndimage as ndi            # noqa: E402
from scipy.ndimage import gaussian_filter, sobel  # noqa: E402
from scipy.ndimage import zoom as scipy_zoom      # noqa: E402
from scipy.linalg import expm, logm         # noqa: E402

# =====================================================================================
# 1. CHEMINS ET RÉGLAGES
# =====================================================================================
try:
    _START = Path(__file__).resolve().parent
except NameError:                           # exécuté sans __file__ (cellule interactive)
    _START = Path.cwd()


def find_project_root() -> Path:
    """Remonte l'arborescence jusqu'à trouver un dossier 'data/'."""
    for start in (_START, Path.cwd()):
        for p in [start, *start.parents]:
            if (p / "data").is_dir():
                return p
    raise FileNotFoundError("Dossier 'data/' introuvable : place ce script dans le projet.")


ROOT = find_project_root()

# --- Entrées ---
REF_PATH = ROOT / "data/Reference/PET_Ref_recale.nii.gz"      # image FIXE
RAW_DIR = ROOT / "data/RAW"                                    # les 80 sujets
M_REF_PATH = ROOT / "data/interim/M_affine_4x4.npy"           # matrice de l'étape 02 (optionnelle)

# --- Sorties ---
OUT_REG_DIR = ROOT / "data/registered/pet_pet"
OUT_MAT_DIR = ROOT / "data/interim/pet_pet"
FIG_DIR = ROOT / "data/qc/figures/pet_pet"
FIG_SUBJ_DIR = FIG_DIR / "subjects"

PARAM_KEYS = ["tx", "ty", "tz", "rx_deg", "ry_deg", "rz_deg", "sx", "sy", "sz"]

# Aspect des figures (comme l'étape 02)
BG, FG_TITLE, FG_AXIS = "#0a0e1a", "#89b4fa", "#cdd6f4"


@dataclass
class Cfg:
    # --- Prétraitement (doit rester identique à l'étape 02 pour la référence) ---
    flip_subject_y: bool = True     # = FLIP_PET_Y de l'étape 02. À mettre à False si la référence n'a pas été flippée.
    zoom_order: int = 1             # interpolation de l'étirement vers la grille de la référence
    sigma_pre: float = 1.0          # lissage avant la pyramide
    # --- Pyramide ---
    downscale: float = 2.0
    max_levels: int = 4
    levels_used: int = 5
    # --- Échantillonnage des voxels de l'image fixe (une seule fois) ---
    max_indices: int = 90000
    edge_ratio: float = 0.7         # 70 % contours, 30 % aléatoires (dans le masque de la tête)
    fg_frac: float = 0.10           # seuil (fraction du max normalisé) du masque d'échantillonnage
    # --- Optimisation ---
    n_iters: int = 1000
    max_samples: int = 32768
    lambda_reg: float = 1e-4
    lr_mine: float = 5e-4
    lr_vL: float = 2e-3
    lr_v1: float = 5e-4
    n_neurons: int = 256
    dropout: float = 0.1
    log_every: int = 100
    early_stop: bool = True
    es_window: int = 100
    es_patience: int = 3
    es_min_delta: float = 1e-3
    # --- Bornes autour du point de départ (en unités des paramètres de l'algèbre de Lie) ---
    bound_t: float = 0.5            # translations (unités normalisées : 1.0 = demi-champ de vue)
    bound_r: float = 0.40           # rotations (rad, soit ~23 degrés)
    bound_s: float = 0.35           # log des échelles (x0.70 à x1.42)
    # --- Point de départ ---
    use_ref_init: bool = True       # utiliser M_affine_4x4.npy (étape 02) comme candidat
    use_com_init: bool = True       # candidats avec alignement des centres de masse
    eval_level: int = 1             # niveau de pyramide pour comparer les candidats
    fallback_margin: float = 0.003  # tolérance du garde-fou (perte de NMI acceptée)
    warn_init_nmi: float = 1.10     # NMI de départ <= ce seuil (1.0 = indépendants) : alerte orientation/flip
    # --- Contrôle qualité ---
    flag_dice_min: float = 0.85
    flag_mad_k: float = 3.0
    seed: int = 0


CFG = Cfg()

_LOG_FILE: Optional[Path] = None


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FILE is not None:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# =====================================================================================
# 2. UTILITAIRES : fichiers, grille, normalisation, pyramide, masques, métriques
# =====================================================================================
def strip_nii_ext(name: str) -> str:
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def load_canonical(path):
    """Charge un NIfTI en orientation canonique (RAS) ; garde le 1er volume si 4D."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    vol = img.get_fdata(dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    return img, vol


def to_fixed_grid(vol, shape, order=1):
    """
    Étire `vol` à la taille `shape` (scipy zoom). IDENTIQUE à to_mri_grid de l'étape 02 :
    c'est ce prétraitement qui a été appliqué à la référence avant sa propre matrice.
    """
    shape = tuple(int(s) for s in shape)
    if vol.shape == shape:
        return vol.astype(np.float32)
    factors = tuple(s / r for s, r in zip(shape, vol.shape))
    out = scipy_zoom(vol, factors, order=order)
    res = np.zeros(shape, np.float32)
    sl = tuple(slice(0, min(a, b)) for a, b in zip(out.shape, shape))
    res[sl] = out[sl]
    return res


def norm01(x, q=99.5):
    """Normalisation [0,1] : x / percentile q des voxels > 0 (zéro conservé). Pour l'apprentissage et l'affichage."""
    x = np.asarray(x, np.float32)
    pos = x[x > 0]
    if pos.size == 0:
        return np.zeros_like(x)
    hi = float(np.percentile(pos, q))
    if hi <= 0:
        return np.zeros_like(x)
    return np.clip(x / hi, 0.0, 1.0).astype(np.float32)


def gradient_magnitude(vol):
    gx, gy, gz = sobel(vol, axis=0), sobel(vol, axis=1), sobel(vol, axis=2)
    return np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)


def pyramid_gaussian_3d(vol, downscale, max_levels):
    """Pyramide gaussienne 3D : [niveau 0 = pleine résolution, 1, 2, ...] (comme l'étape 02)."""
    levels = [vol.astype(np.float32)]
    current = vol.astype(np.float32)
    for _ in range(max_levels):
        blurred = gaussian_filter(current, sigma=downscale / 2.0)
        down = scipy_zoom(blurred, 1.0 / downscale, order=1).astype(np.float32)
        levels.append(down)
        current = down
        if min(current.shape) < 4:
            break
    return levels


def make_grid_3d(D, H, W, device):
    """Grille normalisée [-1,1] : [D,H,W,3] avec (x=W, y=H, z=D)."""
    z_ = torch.linspace(-1, 1, D, device=device, dtype=torch.float32)
    y_ = torch.linspace(-1, 1, H, device=device, dtype=torch.float32)
    x_ = torch.linspace(-1, 1, W, device=device, dtype=torch.float32)
    gz, gy, gx = torch.meshgrid(z_, y_, x_, indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1)


def voxel_coords_normalized(idx, shape_dhw):
    """Indices plats d'un volume [D,H,W] -> coordonnées normalisées [N,3] (x=W, y=H, z=D)."""
    D, H, W = shape_dhw
    d, h, w = np.unravel_index(idx, shape_dhw)
    f = lambda i, n: -1.0 + 2.0 * i / max(n - 1, 1)
    return np.stack([f(w, W), f(h, H), f(d, D)], axis=1).astype(np.float32)


def to_dhw(vol_xyz):
    """[X,Y,Z] -> [D,H,W] = [Z,Y,X], contigu."""
    return np.ascontiguousarray(np.transpose(vol_xyz, (2, 1, 0)))


def foreground_mask(vol01, frac, dilate=2):
    """Masque d'échantillonnage (tête entière) : seuil, plus grande composante, trous remplis, dilatation."""
    m = vol01 > frac
    m = ndi.binary_opening(m, iterations=1) if m.sum() > 500 else m
    if not m.any():
        return np.ones(vol01.shape, bool)
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    m = ndi.binary_fill_holes(m)
    if dilate:
        m = ndi.binary_dilation(m, iterations=dilate)
    return m


def down2(v):
    return v[::2, ::2, ::2]


def qc_mask_half(vol_half01):
    """
    Masque de contrôle qualité à DEMI-résolution (Dice, NMI, NCC), même méthode que l'étape 02
    (seuil relatif, érosion pour détacher le scalp, plus grande composante, dilatation).
    """
    v = gaussian_filter(np.asarray(vol_half01, np.float32), 0.7)
    pos = v[v > 0]
    if pos.size == 0:
        return np.zeros(v.shape, bool)
    m = v > 0.25 * float(np.percentile(pos, 99.5))
    m = ndi.binary_opening(m, iterations=1)
    m = ndi.binary_erosion(m, iterations=2)
    lab, n = ndi.label(m)
    if n == 0:
        return np.zeros(v.shape, bool)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    m = ndi.binary_dilation(m, iterations=2)
    return ndi.binary_fill_holes(m)


def upsample_mask(mask_half, full_shape):
    """Masque demi-résolution -> résolution complète (répétition), pour les contours des figures."""
    m = np.repeat(np.repeat(np.repeat(mask_half, 2, 0), 2, 1), 2, 2)
    out = np.zeros(full_shape, bool)
    sl = tuple(slice(0, min(a, b)) for a, b in zip(m.shape, full_shape))
    out[sl] = m[sl]
    return out


def com_normalized(weights_half, full_shape):
    """Centre de masse d'un volume demi-résolution, en coordonnées normalisées (x,y,z) de la grille complète."""
    w = np.asarray(weights_half, np.float64)
    if w.sum() <= 0:
        return np.zeros(3)
    c = np.array(ndi.center_of_mass(w)) * 2.0                      # indices en pleine résolution (x,y,z)
    return -1.0 + 2.0 * c / (np.array(full_shape, np.float64) - 1.0)


def nmi_hist(a, b, bins=32):
    """Information mutuelle normalisée (H(a)+H(b))/H(a,b) pour a,b dans [0,1] : 1 = indépendants, 2 = identiques."""
    h, _, _ = np.histogram2d(a, b, bins=bins, range=[[0, 1], [0, 1]])
    s = h.sum()
    if s <= 0:
        return 0.0
    p = h / s
    H = lambda q: -(q[q > 0] * np.log(q[q > 0])).sum()
    hxy = H(p)
    return float((H(p.sum(1)) + H(p.sum(0))) / hxy) if hxy > 0 else 0.0


def ncc(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return float(2 * (a & b).sum() / (a.sum() + b.sum() + 1e-12))


# =====================================================================================
# 3. TRANSFORMATIONS : paramètres <-> matrice 4x4, décomposition lisible
# =====================================================================================
def build_C(v):
    """Générateur de l'algèbre de Lie (4x4) à partir de 9 paramètres [tx,ty,tz, rz,ry,rx, sx,sy,sz]
    (même base que AffineNet3D)."""
    v = np.asarray(v, np.float64).ravel()
    C = np.zeros((4, 4))
    C[0, 3], C[1, 3], C[2, 3] = v[0], v[1], v[2]
    C[0, 1], C[1, 0] = v[3], -v[3]
    C[0, 2], C[2, 0] = v[4], -v[4]
    C[1, 2], C[2, 1] = v[5], -v[5]
    C[0, 0], C[1, 1], C[2, 2] = v[6], v[7], v[8]
    return C


def matrix_to_params(M):
    """Logarithme matriciel de M projeté sur la base à 9 paramètres (inverse de expm(build_C(.)))."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        C = np.real(logm(np.asarray(M, np.float64)))
    return np.array([C[0, 3], C[1, 3], C[2, 3], C[0, 1], C[0, 2], C[1, 2],
                     C[0, 0], C[1, 1], C[2, 2]], dtype=np.float64)


def roundtrip_error(M):
    """Écart max entre M et expm(build_C(matrix_to_params(M))) : vérifie que M est représentable."""
    try:
        return float(np.abs(expm(build_C(matrix_to_params(M))) - np.asarray(M, np.float64)).max())
    except Exception:
        return float("inf")


def decompose_affine(M):
    """
    Paramètres LISIBLES d'une matrice 4x4 : translation, angles (degrés), échelles par axe.
    Échelles = normes des colonnes de A ; rotation = facteur polaire de A (SVD).
    """
    M = np.asarray(M, np.float64)
    A, t = M[:3, :3], M[:3, 3]
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    rx = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    ry = math.degrees(-math.asin(float(np.clip(R[2, 0], -1, 1))))
    rz = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    sx, sy, sz = np.linalg.norm(A, axis=0)
    return {"tx": float(t[0]), "ty": float(t[1]), "tz": float(t[2]),
            "rx_deg": rx, "ry_deg": ry, "rz_deg": rz,
            "sx": float(sx), "sy": float(sy), "sz": float(sz)}


# =====================================================================================
# 4. MODÈLES ET WARP
# =====================================================================================
class AffineNet3D(nn.Module):
    """
    Matrice affine 4x4 par exponentielle de matrice (9 DOF) :
      0-2 translations | 3-5 rotations | 6-8 échelles (logarithmes)
    Démarre à `init_params` (point de départ) ; vL0 mémorise ce point pour la régularisation et les bornes.
    """

    def __init__(self, device, init_params=None):
        super().__init__()
        B = torch.zeros(9, 4, 4, dtype=torch.float32, device=device)
        B[0, 0, 3] = 1.0
        B[1, 1, 3] = 1.0
        B[2, 2, 3] = 1.0
        B[3, 0, 1] = 1.0; B[3, 1, 0] = -1.0     # rz
        B[4, 0, 2] = 1.0; B[4, 2, 0] = -1.0     # ry
        B[5, 1, 2] = 1.0; B[5, 2, 1] = -1.0     # rx
        B[6, 0, 0] = 1.0                        # sx
        B[7, 1, 1] = 1.0                        # sy
        B[8, 2, 2] = 1.0                        # sz
        self.register_buffer("B", B)
        v0 = torch.zeros(9, 1, 1, dtype=torch.float32)
        if init_params is not None:
            v0 = torch.as_tensor(np.asarray(init_params), dtype=torch.float32).view(9, 1, 1).clone()
        v0 = v0.to(device)
        self.vL = nn.Parameter(v0.clone())                              # partie commune à tous les niveaux
        self.v1 = nn.Parameter(torch.zeros(9, 1, 1, device=device))     # correction fine (niveau 0)
        self.register_buffer("vL0", v0.clone())

    def forward(self, s: int) -> torch.Tensor:
        C = torch.sum(self.B * self.vL, dim=0)
        if s == 0:
            C = C + torch.sum(self.B * self.v1, dim=0)
        return torch.linalg.matrix_exp(C)

    def regularization(self):
        """Pénalise l'écart au point de départ (et non à l'identité)."""
        return (self.vL - self.vL0).pow(2).mean() + self.v1.pow(2).mean()

    @torch.no_grad()
    def clamp_(self, bounds):
        """Borne vL autour du point de départ et v1 autour de 0 ; bounds : tensor [9,1,1]."""
        lo, hi = self.vL0 - bounds, self.vL0 + bounds
        self.vL.copy_(torch.minimum(torch.maximum(self.vL, lo), hi))
        half = 0.5 * bounds
        self.v1.copy_(torch.minimum(torch.maximum(self.v1, -half), half))


class MINE3D(nn.Module):
    """Estimateur d'information mutuelle sur des paires d'intensités (fixe, mobile)."""

    def __init__(self, n_neurons=256, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(2, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def _net(self, z):
        z = F.leaky_relu(self.fc1(z), 0.1)
        z = self.dropout(F.leaky_relu(self.fc2(z), 0.1))
        return self.fc3(z).squeeze(1)

    def forward(self, fixed_vals, moving_vals):
        n = fixed_vals.numel()
        perm = torch.randperm(n, device=fixed_vals.device)
        joint = torch.stack([fixed_vals, moving_vals], dim=1)
        marginal = torch.stack([fixed_vals, moving_vals[perm]], dim=1)
        z = self._net(torch.cat([joint, marginal], dim=0))          # une seule passe réseau
        z_joint, z_marg = z[:n], z[n:]
        return z_joint.mean() - (torch.logsumexp(z_marg, dim=0) - math.log(n))


def warp_points(moving, M, pts):
    """moving [D,H,W] | M [4,4] | pts [N,3] normalisés -> intensités du mobile en M·pts, [N]."""
    ones = torch.ones(pts.shape[0], 1, device=pts.device, dtype=pts.dtype)
    p = torch.cat([pts, ones], dim=1) @ M.T
    p = p[:, :3] / p[:, 3:4].clamp_min(1e-6)
    out = F.grid_sample(moving[None, None], p.view(1, -1, 1, 1, 3), mode="bilinear",
                        padding_mode="zeros", align_corners=True)
    return out.view(-1)


@torch.no_grad()
def apply_affine_to_volume(vol_xyz, M_np, device, chunk=2_000_000):
    """Applique une matrice 4x4 à un volume [X,Y,Z] déjà sur la grille de la référence (par blocs)."""
    dhw = to_dhw(vol_xyz)
    D, H, W = dhw.shape
    moving = torch.from_numpy(dhw).float().to(device)
    M = torch.from_numpy(np.asarray(M_np, dtype=np.float32)).to(device)
    grid = make_grid_3d(D, H, W, device).reshape(-1, 3)
    out = torch.empty(grid.shape[0], device=device)
    for i in range(0, grid.shape[0], chunk):
        out[i:i + chunk] = warp_points(moving, M, grid[i:i + chunk])
    out = np.transpose(out.view(D, H, W).cpu().numpy(), (2, 1, 0))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# =====================================================================================
# 5. IMAGE FIXE : tout est calculé UNE SEULE FOIS
# =====================================================================================
@dataclass
class FixedContext:
    img: object                     # image nibabel de la référence (affine, zooms)
    affine: np.ndarray
    shape: tuple                    # (X, Y, Z)
    raw: np.ndarray                 # valeurs d'origine
    norm: np.ndarray                # normalisée [0,1]
    pyramid: list                   # pyramide (lissée), niveaux 0..L-1
    L: int
    pts: list = field(default_factory=list)       # par niveau : coordonnées [N,3] (tensor)
    vals: list = field(default_factory=list)      # par niveau : intensités fixes [N] (tensor)
    mask_half: Optional[np.ndarray] = None        # masque QC demi-résolution
    mask_full: Optional[np.ndarray] = None        # idem, résolution complète (contours)
    norm_half: Optional[np.ndarray] = None
    com: Optional[np.ndarray] = None              # centre de masse normalisé (x,y,z)
    mid: tuple = (0, 0, 0)                        # coupes centrales (x,y,z)


def build_fixed_context(device) -> FixedContext:
    log(f"Image fixe : {REF_PATH}")
    img, raw = load_canonical(REF_PATH)
    shape = tuple(int(s) for s in raw.shape)
    norm = norm01(raw)
    pyr = pyramid_gaussian_3d(gaussian_filter(norm, CFG.sigma_pre), CFG.downscale, CFG.max_levels)
    L = min(CFG.levels_used, len(pyr))
    ctx = FixedContext(img=img, affine=img.affine.copy(), shape=shape, raw=raw, norm=norm,
                       pyramid=pyr[:L], L=L, mid=tuple(s // 2 for s in shape))

    rng = np.random.default_rng(CFG.seed)
    for s in range(L):
        lvl = pyr[s]
        dhw = to_dhw(lvl)
        D, H, W = dhw.shape
        fg = to_dhw(foreground_mask(lvl, CFG.fg_frac))                  # tête entière (évite le fond bruité)
        mag = gradient_magnitude(dhw)
        thr = np.percentile(mag[fg], 50) if fg.any() else 0.0
        edge_idx = np.flatnonzero((mag > thr) & fg)
        fg_idx = np.flatnonzero(fg)
        if fg_idx.size == 0:
            fg_idx = np.arange(D * H * W)
        n_edge = int(CFG.max_indices * CFG.edge_ratio)
        n_rand = CFG.max_indices - n_edge
        edge_pick = (rng.choice(edge_idx, size=n_edge, replace=False)
                     if edge_idx.size > n_edge else edge_idx)
        rand_pick = rng.choice(fg_idx, size=min(n_rand, fg_idx.size), replace=False)
        idx = np.unique(np.concatenate([edge_pick, rand_pick]))
        ctx.pts.append(torch.from_numpy(voxel_coords_normalized(idx, (D, H, W))).to(device))
        ctx.vals.append(torch.from_numpy(dhw.ravel()[idx]).to(device))
        log(f"  niveau {s}: {D}x{H}x{W} | contours={edge_pick.size} | échantillon={idx.size}")

    ctx.norm_half = down2(norm)
    ctx.mask_half = qc_mask_half(ctx.norm_half)
    ctx.mask_full = upsample_mask(ctx.mask_half, shape)
    ctx.com = com_normalized(ctx.norm_half * ctx.mask_half, shape)
    log(f"  masque QC : {ctx.mask_half.mean():.3f} du volume (demi-résolution) | "
        f"centre de masse normalisé = {np.round(ctx.com, 3)}")
    return ctx


# =====================================================================================
# 6. UN SUJET : préparation, point de départ, MINE, application, métriques, sauvegardes
# =====================================================================================
def prepare_moving(path, ctx):
    """PET du sujet -> grille de la référence (mêmes étapes que la référence à l'étape 02)."""
    _, vol = load_canonical(path)
    if CFG.flip_subject_y:
        vol = np.ascontiguousarray(np.flip(vol, axis=1))
    raw = to_fixed_grid(vol, ctx.shape, CFG.zoom_order)                  # valeurs d'origine
    norm = norm01(raw)
    pyr = pyramid_gaussian_3d(gaussian_filter(norm, CFG.sigma_pre), CFG.downscale, CFG.max_levels)
    return raw, norm, pyr[:ctx.L]


def candidate_inits(ctx, mov_norm, M_ref):
    """Candidats de départ {nom: matrice 4x4}. Le meilleur (NMI) sera retenu."""
    cands = {}
    com_m = com_normalized(down2(mov_norm) * qc_mask_half(down2(mov_norm)), ctx.shape)

    def with_com(A):
        M = np.eye(4)
        M[:3, :3] = A
        M[:3, 3] = com_m - A @ ctx.com          # envoie le centre de masse fixe sur celui du mobile
        return M

    if M_ref is not None and CFG.use_ref_init:
        cands["ref"] = np.asarray(M_ref, np.float64)
        if CFG.use_com_init:
            cands["ref+com"] = with_com(np.asarray(M_ref, np.float64)[:3, :3])
    if CFG.use_com_init:
        cands["identité+com"] = with_com(np.eye(3))
    cands["identité"] = np.eye(4)
    return cands


@torch.no_grad()
def score_matrix(ctx, J_lst, M_np, level, device, n=32768):
    """NMI entre l'image fixe et le mobile déformé par M, sur un échantillon de voxels d'un niveau."""
    pts, vals = ctx.pts[level], ctx.vals[level]
    step = max(1, int(math.ceil(pts.shape[0] / n)))
    pts, vals = pts[::step], vals[::step]
    M = torch.from_numpy(np.asarray(M_np, np.float32)).to(device)
    mv = warp_points(J_lst[level], M, pts)
    return nmi_hist(vals.cpu().numpy(), mv.cpu().numpy(), bins=24)


def optimise_transform(ctx, J_lst, M_init, device, seed):
    """Optimise T par MINE multi-résolution, à partir de M_init. Retourne (T, courbe MI, nb d'itérations)."""
    torch.manual_seed(seed)
    affine_net = AffineNet3D(device, init_params=matrix_to_params(M_init)).to(device)
    mine_net = MINE3D(CFG.n_neurons, CFG.dropout).to(device)
    optimizer = optim.Adam(
        [{"params": mine_net.parameters(), "lr": CFG.lr_mine},
         {"params": affine_net.vL, "lr": CFG.lr_vL},
         {"params": affine_net.v1, "lr": CFG.lr_v1}],
        amsgrad=True)
    bounds = torch.tensor([CFG.bound_t] * 3 + [CFG.bound_r] * 3 + [CFG.bound_s] * 3,
                          dtype=torch.float32, device=device).view(9, 1, 1)
    L = ctx.L
    mi_curve, best_mean, bad = [], -np.inf, 0
    t0 = time.time()

    for itr in range(CFG.n_iters):
        optimizer.zero_grad(set_to_none=True)
        loss = 0.0
        for s in range(L - 1, -1, -1):                          # grossier -> fin
            pts, fv = ctx.pts[s], ctx.vals[s]
            if pts.shape[0] > CFG.max_samples:
                sel = torch.randperm(pts.shape[0], device=pts.device)[:CFG.max_samples]
                pts, fv = pts[sel], fv[sel]
            mv = warp_points(J_lst[s], affine_net(s), pts)
            loss = loss - (1.0 / L) * mine_net(fv, mv)
        loss = loss + CFG.lambda_reg * affine_net.regularization()
        loss.backward()
        optimizer.step()
        affine_net.clamp_(bounds)
        mi_curve.append(float(-loss.detach()))

        if (itr + 1) % CFG.log_every == 0:
            log(f"    iter {itr + 1}/{CFG.n_iters} | MI proxy {mi_curve[-1]:.4f} | {time.time() - t0:.0f}s")
        if CFG.early_stop and (itr + 1) % CFG.es_window == 0:
            cur = float(np.mean(mi_curve[-CFG.es_window:]))
            if cur > best_mean + CFG.es_min_delta:
                best_mean, bad = cur, 0
            else:
                bad += 1
                if bad >= CFG.es_patience:
                    log(f"    arrêt anticipé à l'itération {itr + 1} (MI stable)")
                    break

    with torch.no_grad():
        T = affine_net(0).cpu().numpy().astype(np.float64)
    del affine_net, mine_net, optimizer
    return T, mi_curve, len(mi_curve)


def compute_metrics(ctx, before_n, after_n):
    """NMI / NCC dans le masque de la référence, Dice des masques de tête, avant et après (demi-résolution)."""
    bh, ah = down2(before_n), down2(after_n)
    sel = ctx.mask_half
    out = {}
    for tag, vh in (("before", bh), ("after", ah)):
        out[f"nmi_{tag}"] = nmi_hist(ctx.norm_half[sel], vh[sel])
        out[f"ncc_{tag}"] = ncc(ctx.norm_half[sel], vh[sel])
        out[f"dice_{tag}"] = dice(ctx.mask_half, qc_mask_half(vh))
    return out


def central_planes(vol, mid):
    x, y, z = mid
    return {"ax": vol[:, :, z], "cor": vol[:, y, :], "sag": vol[x]}


def process_subject(ctx, path, index, n_total, M_ref, device):
    sid = strip_nii_ext(path.name)
    t0 = time.time()
    seed = CFG.seed + index
    random.seed(seed)
    np.random.seed(seed)
    log(f"[{index}/{n_total}] sujet {sid}")

    raw, norm, pyr = prepare_moving(path, ctx)
    J_lst = [torch.from_numpy(to_dhw(lvl)).to(device) for lvl in pyr]

    # --- Point de départ : on garde le candidat de meilleure NMI ---
    level = min(CFG.eval_level, ctx.L - 1)
    cands = candidate_inits(ctx, norm, M_ref)
    scores = {}
    for name, M0 in cands.items():
        err = roundtrip_error(M0)
        if err > 1e-3:
            log(f"    candidat '{name}' ignoré (non représentable, erreur {err:.1e})")
            continue
        scores[name] = score_matrix(ctx, J_lst, M0, level, device)
    if not scores:
        cands, scores = {"identité": np.eye(4)}, {"identité": score_matrix(ctx, J_lst, np.eye(4), level, device)}
    init_name = max(scores, key=scores.get)
    M_init = cands[init_name]
    log("    départ : " + " | ".join(f"{k}={v:.3f}" for k, v in scores.items()) + f"  -> {init_name}")
    if scores[init_name] <= CFG.warn_init_nmi:
        log(f"    ALERTE : NMI de départ {scores[init_name]:.3f} proche de l'indépendance (1.0) : "
            "vérifier l'orientation du sujet (CFG.flip_subject_y) avant de se fier à ce résultat")

    # --- MINE ---
    T, mi_curve, n_done = optimise_transform(ctx, J_lst, M_init, device, seed)
    s_final = score_matrix(ctx, J_lst, T, level, device)
    fell_back = bool(s_final < scores[init_name] - CFG.fallback_margin)
    if fell_back:
        log(f"    garde-fou : NMI {s_final:.3f} < départ {scores[init_name]:.3f} -> point de départ conservé")
        T = M_init
        s_final = scores[init_name]

    # --- Application au PET ORIGINAL (valeurs d'origine) ---
    warped_raw = apply_affine_to_volume(raw, T, device)
    before_n, after_n = norm, norm01(warped_raw)
    met = compute_metrics(ctx, before_n, after_n)

    # --- Sauvegardes ---
    nib.save(nib.Nifti1Image(warped_raw.astype(np.float32), ctx.affine), str(OUT_REG_DIR / f"{sid}_reg.nii.gz"))
    np.save(OUT_MAT_DIR / f"{sid}_T.npy", T)
    np.savetxt(OUT_MAT_DIR / f"{sid}_T.txt", T, fmt="%.6f")
    np.save(OUT_MAT_DIR / f"{sid}_mi.npy", np.array(mi_curve, np.float32))
    pb, pa = central_planes(before_n, ctx.mid), central_planes(after_n, ctx.mid)
    np.savez_compressed(OUT_MAT_DIR / f"{sid}_thumbs.npz",
                        **{f"b_{k}": v.astype(np.float16) for k, v in pb.items()},
                        **{f"a_{k}": v.astype(np.float16) for k, v in pa.items()})

    res = {"id": sid, "index": index, "file": path.name, "init": init_name,
           "init_scores": {k: float(v) for k, v in scores.items()}, "nmi_points_final": float(s_final),
           "fell_back": fell_back, "n_iters": n_done, "n_iters_max": CFG.n_iters,
           "mi_final": float(np.mean(mi_curve[-50:])) if mi_curve else 0.0,
           "flip_y": CFG.flip_subject_y, "T": T.tolist(), "params": decompose_affine(T),
           "time_s": round(time.time() - t0, 1), **{k: float(v) for k, v in met.items()}}
    with open(OUT_MAT_DIR / f"{sid}_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)

    fig_subject_pipeline(ctx, res, norm, pyr, after_n, mi_curve, T)
    log(f"    NMI {met['nmi_before']:.3f} -> {met['nmi_after']:.3f} | Dice {met['dice_before']:.3f} -> "
        f"{met['dice_after']:.3f} | {n_done} it | {res['time_s']:.0f}s")
    del J_lst, raw, warped_raw
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


# =====================================================================================
# 7. FIGURES
# =====================================================================================
def save_fig(fig, path, dpi=130):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _style_axes(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG_AXIS, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#45475a")


def rgb_composite(fix2d, mov2d):
    """Vert = référence, magenta = sujet : parfaitement aligné -> gris/blanc."""
    f, m = np.clip(fix2d.astype(np.float32), 0, 1), np.clip(mov2d.astype(np.float32), 0, 1)
    return np.dstack([m, f, m])


def fig_reference(ctx):
    x, y, z = ctx.mid
    sx, sy, sz = ctx.img.header.get_zooms()[:3]
    pos = ctx.raw[ctx.raw > 0]
    lo, hi = np.percentile(pos, [1, 99.5])
    planes = [("Sagittal", ctx.raw[x], ctx.mask_full[x], sz / sy, x),
              ("Coronal", ctx.raw[:, y], ctx.mask_full[:, y], sz / sx, y),
              ("Axial", ctx.raw[:, :, z], ctx.mask_full[:, :, z], sy / sx, z)]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor(BG)
    for ax, (name, sl, mk, asp, i) in zip(axs, planes):
        im = ax.imshow(sl.T, cmap="gray", origin="lower", aspect=asp, vmin=lo, vmax=hi)
        if mk.min() != mk.max():
            ax.contour(mk.T.astype(float), levels=[0.5], colors="red", linewidths=1.0, origin="lower")
        ax.set_title(f"{name} ({i})", color=FG_TITLE, fontsize=10)
        ax.axis("off")
    cb = fig.colorbar(im, ax=axs, shrink=0.8)
    cb.set_label("PET référence (valeurs d'origine)", color=FG_AXIS)
    cb.ax.tick_params(colors=FG_AXIS)
    fig.suptitle("PET de référence (image FIXE) — contour rouge : masque tête/cerveau", color=FG_AXIS,
                 fontsize=12, fontweight="bold")
    return save_fig(fig, FIG_DIR / "fig_01_reference.png", dpi=150)


def fig_montage(tiles, ids, ctx, title, path, contour=True, ncols=10):
    """tiles : liste de tableaux 2D [X,Y] dans [0,1] ; contour rouge = masque de la référence (coupe axiale)."""
    n = len(tiles)
    nrows = int(math.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 1.55, nrows * 1.75 + 0.7))
    fig.patch.set_facecolor(BG)
    axs = np.atleast_2d(axs)
    for ax in axs.ravel():
        ax.axis("off")
    ref_c = ctx.mask_full[:, :, ctx.mid[2]].T.astype(float) if contour else None
    for ax, t, sid in zip(axs.ravel(), tiles, ids):
        ax.imshow(np.asarray(t, np.float32).T, cmap="gray", origin="lower", vmin=0, vmax=1)
        if contour and ref_c.min() != ref_c.max() and t.shape == ctx.mask_full.shape[:2]:
            ax.contour(ref_c, levels=[0.5], colors="red", linewidths=0.5, origin="lower")
        ax.set_title(sid, fontsize=5.5, color=FG_AXIS, pad=1.5)
    fig.suptitle(title, color=FG_AXIS, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_fig(fig, path, dpi=140)


def fig_subject_pipeline(ctx, res, mov_norm, mov_pyr, after_n, mi_curve, T):
    """Une figure par sujet qui illustre les 5 étapes du traitement individuel."""
    x, y, z = ctx.mid
    sid, idx = res["id"], res["index"]
    fig = plt.figure(figsize=(19, 10.5))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(3, 6, hspace=0.30, wspace=0.10)

    def show(ax, im2d, title):
        ax.imshow(np.asarray(im2d, np.float32).T, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.set_title(title, color=FG_TITLE, fontsize=9)
        ax.axis("off")

    # Ligne 1 — étape 1 : PET du sujet + PET de référence
    show(fig.add_subplot(gs[0, 0]), ctx.norm[:, :, z], "1 · Référence (fixe), axial")
    show(fig.add_subplot(gs[0, 1]), mov_norm[:, :, z], f"1 · Sujet {sid} (avant), axial")
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(rgb_composite(ctx.norm[:, :, z].T, mov_norm[:, :, z].T), origin="lower")
    ax.set_title("1 · Avant : vert = réf., magenta = sujet", color=FG_TITLE, fontsize=9)
    ax.axis("off")
    axt = fig.add_subplot(gs[0, 3:])
    axt.axis("off")
    p = res["params"]
    txt = (f"T{idx}  —  sujet {sid}\n"
           f"départ : {res['init']}   (garde-fou : {'oui' if res['fell_back'] else 'non'})\n"
           f"itérations : {res['n_iters']}/{res['n_iters_max']}   MI proxy finale : {res['mi_final']:.3f}\n"
           f"NMI    {res['nmi_before']:.3f} -> {res['nmi_after']:.3f}\n"
           f"NCC    {res['ncc_before']:.3f} -> {res['ncc_after']:.3f}\n"
           f"Dice   {res['dice_before']:.3f} -> {res['dice_after']:.3f}\n"
           f"translation  ({p['tx']:+.3f}, {p['ty']:+.3f}, {p['tz']:+.3f})\n"
           f"rotation °   ({p['rx_deg']:+.1f}, {p['ry_deg']:+.1f}, {p['rz_deg']:+.1f})\n"
           f"échelles     ({p['sx']:.3f}, {p['sy']:.3f}, {p['sz']:.3f})")
    axt.text(0.0, 1.0, txt, va="top", ha="left", family="monospace", color=FG_AXIS, fontsize=10.5)

    # Ligne 2 — étape 2 : pyramide ; étape 3 : MINE
    for s in range(min(ctx.L, 5)):
        lvl = mov_pyr[s]
        show(fig.add_subplot(gs[1, s]), lvl[:, :, lvl.shape[2] // 2],
             f"2 · Pyramide niv. {s} {lvl.shape[0]}x{lvl.shape[1]}x{lvl.shape[2]}")
    axc = fig.add_subplot(gs[1, 5])
    _style_axes(axc)
    axc.plot(mi_curve, color="#6366f1", lw=0.8)
    axc.set_title("3 · MINE : MI proxy", color=FG_TITLE, fontsize=9)
    axc.set_xlabel("itération", color=FG_AXIS, fontsize=7)
    axc.grid(alpha=0.25)

    # Ligne 3 — étape 4 : T_i ; étape 5 : application au PET original
    axm = fig.add_subplot(gs[2, 0:2])
    axm.imshow(T, cmap="coolwarm", vmin=-1.2, vmax=1.2)
    for i in range(4):
        for j in range(4):
            axm.text(j, i, f"{T[i, j]:.3f}", ha="center", va="center", fontsize=10,
                     color="white" if abs(T[i, j]) > 0.7 else "black")
    axm.set_title(f"4 · Matrice T{idx} (grille réf. -> sujet)", color=FG_TITLE, fontsize=9)
    axm.set_xticks([])
    axm.set_yticks([])
    show(fig.add_subplot(gs[2, 2]), after_n[:, :, z], "5 · PET original recalé, axial")
    for k, (name, a_sl, r_sl) in enumerate([("axial", after_n[:, :, z], ctx.norm[:, :, z]),
                                            ("coronal", after_n[:, y, :], ctx.norm[:, y, :]),
                                            ("sagittal", after_n[x], ctx.norm[x])]):
        ax = fig.add_subplot(gs[2, 3 + k])
        ax.imshow(rgb_composite(r_sl.T, a_sl.T), origin="lower")
        ax.set_title(f"5 · Après ({name})", color=FG_TITLE, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Recalage PET -> PET de référence — T{idx} ({sid})", color=FG_AXIS, fontsize=13,
                 fontweight="bold", y=0.96)
    return save_fig(fig, FIG_SUBJ_DIR / f"{idx:02d}_{sid}_pipeline.png", dpi=90)


def robust_z(v):
    v = np.asarray(v, np.float64)
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    return (v - med) / (mad if mad > 1e-9 else 1.0)


def fig_transforms(ids, indices, P):
    """Les N matrices T_i : chaque sujet a sa propre transformation."""
    n = len(ids)
    Z = np.clip(np.apply_along_axis(robust_z, 0, P), -3, 3)
    fig = plt.figure(figsize=(17, max(6.0, 0.17 * n + 2.5)))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(3, 5, width_ratios=[1.7, 1, 1, 1, 0.05], wspace=0.35, hspace=0.55)
    axh = fig.add_subplot(gs[:, 0])
    im = axh.imshow(Z, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    axh.set_xticks(range(9))
    axh.set_xticklabels(["t_x", "t_y", "t_z", "rot_x", "rot_y", "rot_z", "éch_x", "éch_y", "éch_z"],
                        color=FG_AXIS, fontsize=8, rotation=45)
    axh.set_yticks(range(n))
    axh.set_yticklabels([f"T{i}  {s}" for i, s in zip(indices, ids)], color=FG_AXIS, fontsize=5)
    axh.set_title("Paramètres de chaque T_i (écart à la médiane, en écarts robustes)", color=FG_TITLE, fontsize=10)
    cax = fig.add_subplot(gs[:, 4])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(colors=FG_AXIS)
    labels = ["translation x", "translation y", "translation z", "rotation x (°)", "rotation y (°)",
              "rotation z (°)", "échelle x", "échelle y", "échelle z"]
    for k in range(9):
        ax = fig.add_subplot(gs[k // 3, 1 + k % 3])
        _style_axes(ax)
        ax.hist(P[:, k], bins=min(20, max(5, n // 3)), color="#6366f1", edgecolor=BG)
        ax.axvline(np.median(P[:, k]), color="#f59e0b", lw=1)
        ax.set_title(labels[k], color=FG_TITLE, fontsize=8)
    fig.suptitle(f"Une matrice par sujet : T1 ... T{n}", color=FG_AXIS, fontsize=13, fontweight="bold")
    return save_fig(fig, FIG_DIR / "fig_03_transforms.png", dpi=120)


def fig_mean_before_after(ctx, thumbs):
    """Image moyenne des sujets avant / après : plus nette et centrée sur la référence après recalage."""
    keys = [("ax", "Axial"), ("cor", "Coronal"), ("sag", "Sagittal")]
    fig, axs = plt.subplots(3, 4, figsize=(15, 11))
    fig.patch.set_facecolor(BG)
    ref = central_planes(ctx.norm, ctx.mid)
    for r, (k, name) in enumerate(keys):
        mb = np.mean([t[f"b_{k}"].astype(np.float32) for t in thumbs], axis=0)
        ma = np.mean([t[f"a_{k}"].astype(np.float32) for t in thumbs], axis=0)
        sa = np.std([t[f"a_{k}"].astype(np.float32) for t in thumbs], axis=0)
        mk = {"ax": ctx.mask_full[:, :, ctx.mid[2]], "cor": ctx.mask_full[:, ctx.mid[1], :],
              "sag": ctx.mask_full[ctx.mid[0]]}[k]
        for c, (im2d, ttl, vmax) in enumerate([(ref[k], "Référence", 1), (mb, f"Moyenne AVANT (N={len(thumbs)})", 1),
                                               (ma, "Moyenne APRÈS", 1), (sa, "Écart-type APRÈS", None)]):
            ax = axs[r, c]
            ax.imshow(im2d.T, cmap="gray" if c < 3 else "magma", origin="lower", vmin=0, vmax=vmax)
            if mk.min() != mk.max():
                ax.contour(mk.T.astype(float), levels=[0.5], colors="red", linewidths=0.6, origin="lower")
            ax.set_title(f"{name} — {ttl}", color=FG_TITLE, fontsize=8)
            ax.axis("off")
    fig.suptitle("Moyenne des sujets avant / après recalage (contour rouge : masque de la référence)",
                 color=FG_AXIS, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_fig(fig, FIG_DIR / "fig_06_mean_before_after.png", dpi=120)


def fig_metrics(rows, flagged):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.patch.set_facecolor(BG)
    for ax, (key, name) in zip(axs, [("nmi", "NMI"), ("ncc", "NCC"), ("dice", "Dice (masques de tête)")]):
        _style_axes(ax)
        b = np.array([r[f"{key}_before"] for r in rows])
        a = np.array([r[f"{key}_after"] for r in rows])
        lo, hi = min(b.min(), a.min()), max(b.max(), a.max())
        pad = 0.05 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#6c7086", lw=1, ls="--")
        ok = np.array([r["id"] not in flagged for r in rows])
        ax.scatter(b[ok], a[ok], s=18, color="#6366f1", label="OK")
        if (~ok).any():
            ax.scatter(b[~ok], a[~ok], s=28, color="#f87171", label="à vérifier")
            for r, bb, aa in zip(rows, b, a):
                if r["id"] in flagged:
                    ax.annotate(r["id"], (bb, aa), fontsize=6, color="#f87171", xytext=(3, 3),
                                textcoords="offset points")
        ax.set_xlabel("avant recalage", color=FG_AXIS, fontsize=9)
        ax.set_ylabel("après recalage", color=FG_AXIS, fontsize=9)
        ax.set_title(f"{name} — médiane {np.median(b):.3f} -> {np.median(a):.3f}", color=FG_TITLE, fontsize=10)
        ax.grid(alpha=0.2)
        ax.legend(facecolor=BG, labelcolor=FG_AXIS, fontsize=8)
    fig.suptitle("Qualité du recalage : au-dessus de la diagonale = amélioration", color=FG_AXIS,
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_fig(fig, FIG_DIR / "fig_07_metrics.png", dpi=120)


def fig_native_montage(subjects, ctx):
    """Les sujets tels qu'ils entrent dans le pipeline (flip éventuel appliqué), coupe axiale centrale native."""
    tiles, ids = [], []
    for p in subjects:
        try:
            _, vol = load_canonical(p)
            if CFG.flip_subject_y:
                vol = np.flip(vol, axis=1)
            tiles.append(norm01(vol[:, :, vol.shape[2] // 2]))
            ids.append(strip_nii_ext(p.name))
        except Exception as e:
            log(f"  montage natif : {p.name} ignoré ({e})")
    return fig_montage(tiles, ids, ctx, f"{len(tiles)} PET sujets : positions et orientations initiales (coupe axiale native)",
                       FIG_DIR / "fig_02_subjects_native.png", contour=False)


# =====================================================================================
# 8. DÉCOUVERTE, CONTRÔLES PRÉALABLES, AGRÉGATION
# =====================================================================================
def discover_subjects(raw_dir: Path):
    files = [p for p in raw_dir.iterdir()
             if p.is_file() and not p.name.startswith(".") and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))]
    return sorted(files, key=lambda p: p.name)


def find_inspection_csv():
    for c in (ROOT / "data/interim/00_inspection.csv", ROOT / "data/qc/00_inspection.csv",
              ROOT / "data/00_inspection.csv", ROOT / "00_inspection.csv"):
        if c.exists():
            return c
    return next(ROOT.glob("data/**/00_inspection.csv"), None)


def read_inspection(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def expected_geometry(rows):
    """(shape, voxel_size) de la ligne 'reference' de 00_inspection.csv, ou None."""
    try:
        ref = next(r for r in rows if r.get("role") == "reference")
        return (tuple(int(v) for v in ast.literal_eval(ref["shape"])),
                tuple(float(v) for v in ast.literal_eval(ref["voxel_size_mm"])))
    except Exception:
        return None


def preflight(subjects, rows):
    """Vérifie la géométrie de chaque sujet (en-têtes seulement) ; retourne (sujets_ok, problèmes)."""
    exp = expected_geometry(rows) if rows else None
    ok, bad = [], []
    for p in subjects:
        try:
            img = nib.load(str(p))
            shape = tuple(int(s) for s in img.shape[:3])
            zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        except Exception as e:
            bad.append((p.name, f"illisible : {e}"))
            continue
        if exp is None:
            exp = (shape, zooms)                        # à défaut de CSV : le premier sujet fait référence
        why = []
        if shape != exp[0]:
            why.append(f"shape {shape} != {exp[0]}")
        if not np.allclose(zooms, exp[1], atol=1e-3):
            why.append(f"voxels {tuple(round(z, 3) for z in zooms)} != {tuple(round(z, 3) for z in exp[1])}")
        (bad if why else ok).append(p if not why else (p.name, " ; ".join(why)))
    return ok, bad


def load_results(ok_subjects):
    rows = []
    for p in ok_subjects:
        sid = strip_nii_ext(p.name)
        jp = OUT_MAT_DIR / f"{sid}_result.json"
        if all(q.exists() for q in (jp, OUT_MAT_DIR / f"{sid}_T.npy", OUT_MAT_DIR / f"{sid}_thumbs.npz")):
            with open(jp, encoding="utf-8") as f:
                rows.append(json.load(f))
    return rows


def flag_subjects(rows):
    """Sujets à vérifier : Dice faible, NMI qui n'augmente pas, ou NMI finale très en dessous de la cohorte."""
    flagged = {}
    if not rows:
        return flagged
    z_nmi = robust_z([r["nmi_after"] for r in rows]) if len(rows) >= 5 else np.zeros(len(rows))
    for r, z in zip(rows, z_nmi):
        why = []
        if r["dice_after"] < CFG.flag_dice_min:
            why.append(f"Dice {r['dice_after']:.2f} < {CFG.flag_dice_min}")
        if r["nmi_after"] < r["nmi_before"]:
            why.append("NMI en baisse")
        if z < -CFG.flag_mad_k:
            why.append(f"NMI faible vs cohorte (z={z:.1f})")
        if r.get("fell_back"):
            why.append("garde-fou activé")
        if why:
            flagged[r["id"]] = "; ".join(why)
    return flagged


def aggregate(ctx, ok_subjects):
    rows = load_results(ok_subjects)
    if not rows:
        log("Aucun résultat à agréger.")
        return
    ids = [r["id"] for r in rows]
    idxs = [r["index"] for r in rows]
    T_all = np.stack([np.load(OUT_MAT_DIR / f"{i}_T.npy") for i in ids])
    P = np.array([[r["params"][k] for k in PARAM_KEYS] for r in rows])
    flagged = flag_subjects(rows)

    np.save(OUT_MAT_DIR / "T_all.npy", T_all)
    with open(OUT_MAT_DIR / "T_all.txt", "w", encoding="utf-8") as f:
        for r, T in zip(rows, T_all):
            f.write(f"T{r['index']}  ({r['id']})\n")
            np.savetxt(f, T, fmt="%.6f")
            f.write("\n")

    cols = ["index", "id", "init", "fell_back", "n_iters", "mi_final", "nmi_before", "nmi_after", "ncc_before",
            "ncc_after", "dice_before", "dice_after", "time_s"] + PARAM_KEYS + ["a_verifier"]
    with open(OUT_MAT_DIR / "registration_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["index"], r["id"], r["init"], r["fell_back"], r["n_iters"], round(r["mi_final"], 4),
                        round(r["nmi_before"], 4), round(r["nmi_after"], 4), round(r["ncc_before"], 4),
                        round(r["ncc_after"], 4), round(r["dice_before"], 4), round(r["dice_after"], 4),
                        r["time_s"]] + [round(r["params"][k], 5) for k in PARAM_KEYS] +
                       [flagged.get(r["id"], "")])

    thumbs = [dict(np.load(OUT_MAT_DIR / f"{i}_thumbs.npz")) for i in ids]
    z = ctx.mid[2]
    fig_montage([t["b_ax"].astype(np.float32) for t in thumbs], ids, ctx,
                f"{len(ids)} PET sujets sur la grille de la référence, AVANT recalage (contour rouge : masque référence)",
                FIG_DIR / "fig_04_subjects_before_vs_ref.png")
    fig_montage([t["a_ax"].astype(np.float32) for t in thumbs], ids, ctx,
                f"{len(ids)} PET sujets APRÈS recalage dans l'espace du PET de référence",
                FIG_DIR / "fig_05_subjects_after_registered.png")
    fig_transforms(ids, idxs, P)
    fig_mean_before_after(ctx, thumbs)
    fig_metrics(rows, flagged)

    med = lambda k: float(np.median([r[k] for r in rows]))
    log(f"Bilan sur {len(rows)} sujets — médianes : NMI {med('nmi_before'):.3f} -> {med('nmi_after'):.3f} | "
        f"NCC {med('ncc_before'):.3f} -> {med('ncc_after'):.3f} | Dice {med('dice_before'):.3f} -> {med('dice_after'):.3f}")
    inits = {}
    for r in rows:
        inits[r["init"]] = inits.get(r["init"], 0) + 1
    log("Points de départ retenus : " + ", ".join(f"{k}={v}" for k, v in inits.items()))
    if flagged:
        log(f"{len(flagged)} sujet(s) à vérifier visuellement :")
        for sid, why in flagged.items():
            log(f"   {sid} : {why}")
    log(f"Matrices : {OUT_MAT_DIR / 'T_all.npy'} | résumé : {OUT_MAT_DIR / 'registration_summary.csv'}")
    log(f"Figures  : {FIG_DIR}")


# =====================================================================================
# 9. PROGRAMME PRINCIPAL
# =====================================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Recalage PET sujet -> PET de référence (une matrice par sujet).")
    ap.add_argument("--limit", type=int, default=None, help="traiter seulement les N premiers sujets (pilote)")
    ap.add_argument("--subjects", nargs="+", default=None, help="identifiants à traiter (ex. 200011432 976017)")
    ap.add_argument("--force", action="store_true", help="refaire même si les résultats existent")
    ap.add_argument("--group-only", action="store_true", help="ne refaire que les figures et le CSV de groupe")
    ap.add_argument("--no-group", action="store_true", help="ne pas produire les figures de groupe")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--iters", type=int, default=None, help="nombre maximal d'itérations MINE par sujet")
    ap.add_argument("--no-flip", action="store_true", help="désactive le flip Y (si la référence n'a pas été flippée)")
    args, _ = ap.parse_known_args(argv)                  # tolère les arguments du noyau Jupyter
    return args


def main(argv=None):
    global _LOG_FILE
    args = parse_args(argv)
    if args.iters:
        CFG.n_iters = args.iters
    if args.no_flip:
        CFG.flip_subject_y = False

    for d in (OUT_REG_DIR, OUT_MAT_DIR, FIG_DIR, FIG_SUBJ_DIR):
        d.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = OUT_MAT_DIR / "registration.log"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA demandé mais indisponible.")
    if device.type == "cpu":
        log("Pas de GPU : le recalage 3D sera lent (plusieurs minutes par sujet).")
    else:
        log(f"GPU : {torch.cuda.get_device_name(0)}")
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)

    # --- Contrôles préalables ---
    for p, what in ((REF_PATH, "image fixe (référence)"), (RAW_DIR, "dossier des sujets")):
        if not p.exists():
            raise FileNotFoundError(f"{what} introuvable : {p}")
    subjects = discover_subjects(RAW_DIR)
    csv_path = find_inspection_csv()
    rows = read_inspection(csv_path) if csv_path else []
    log(f"{len(subjects)} fichiers dans {RAW_DIR}" + (f" | inspection : {csv_path.name}" if csv_path else " | pas de 00_inspection.csv"))
    if rows:
        want = {strip_nii_ext(Path(r["name"]).name) for r in rows if r.get("role") == "subject"}
        have = {strip_nii_ext(p.name) for p in subjects}
        if want - have:
            log(f"  ATTENTION : {len(want - have)} sujet(s) de l'inspection absents de RAW : {sorted(want - have)[:5]}...")
        if have - want:
            log(f"  ATTENTION : {len(have - want)} fichier(s) de RAW absents de l'inspection : {sorted(have - want)[:5]}...")
    ok_subjects, bad = preflight(subjects, rows)
    for name, why in bad:
        log(f"  IGNORÉ {name} : {why}")
    if not ok_subjects:
        raise RuntimeError("Aucun sujet exploitable.")
    index_of = {strip_nii_ext(p.name): i + 1 for i, p in enumerate(ok_subjects)}   # T1..TN, stable d'un lancement à l'autre
    if len(ok_subjects) != 80:
        log(f"  Remarque : {len(ok_subjects)} sujets exploitables (80 attendus).")

    todo = ok_subjects
    if args.subjects:
        wanted = set(args.subjects)
        todo = [p for p in ok_subjects if strip_nii_ext(p.name) in wanted]
        missing = wanted - {strip_nii_ext(p.name) for p in todo}
        if missing:
            log(f"  Identifiants introuvables : {sorted(missing)}")
    if args.limit:
        todo = todo[:args.limit]

    ctx = build_fixed_context(device)
    fig_reference(ctx)

    M_ref = None
    if CFG.use_ref_init and M_REF_PATH.exists():
        M_ref = np.load(M_REF_PATH).astype(np.float64)
        err = roundtrip_error(M_ref)
        log(f"Matrice de l'étape 02 : {M_REF_PATH.name} (erreur de représentation {err:.1e})")
        if err > 1e-3:
            log("  Elle n'est pas représentable par le modèle à 9 DOF : ignorée comme point de départ.")
            M_ref = None
    else:
        log("Pas de matrice d'étape 02 : départ par centres de masse / identité.")
    if not args.group_only:
        native_fig = FIG_DIR / "fig_02_subjects_native.png"
        if args.force or not native_fig.exists():
            fig_native_montage(ok_subjects, ctx)

        t_start, done_now, failures = time.time(), 0, []
        for k, path in enumerate(todo, 1):
            sid = strip_nii_ext(path.name)
            already = all((p).exists() for p in (OUT_MAT_DIR / f"{sid}_result.json", OUT_MAT_DIR / f"{sid}_T.npy",
                                                 OUT_REG_DIR / f"{sid}_reg.nii.gz", OUT_MAT_DIR / f"{sid}_thumbs.npz"))
            if already and not args.force:
                log(f"[{index_of[sid]}] {sid} : déjà traité, ignoré (--force pour refaire)")
                continue
            try:
                process_subject(ctx, path, index_of[sid], len(ok_subjects), M_ref, device)
                done_now += 1
                eta = (time.time() - t_start) / done_now * (len(todo) - k)
                log(f"    avancement {k}/{len(todo)} | reste ~{eta / 60:.0f} min")
            except Exception as e:                                   # un échec n'arrête pas le lot
                failures.append((sid, repr(e)))
                log(f"  ECHEC {sid} : {e!r}")
                log(traceback.format_exc())
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if failures:
            with open(OUT_MAT_DIR / "failures.txt", "w", encoding="utf-8") as f:
                for sid, err in failures:
                    f.write(f"{sid}\t{err}\n")
            log(f"{len(failures)} échec(s), voir {OUT_MAT_DIR / 'failures.txt'}")

    if not args.no_group:
        aggregate(ctx, ok_subjects)


if __name__ == "__main__":
    main()