# %% [markdown]
# # Recalage PET -> IRM (MINE 3D) : version VS Code (rapide, sans pop-up)
# Fixe = IRM (MNI152) | Mobile = PET (971882)
#
# Changements par rapport à la version précédente :
#   - warp des SEULS voxels échantillonnés (au lieu de tout le volume à chaque itération/niveau)
#   - positions + intensités de l'IRM pré-calculées une fois
#   - une seule passe réseau MINE (paires jointes + marginales)
#   - arrêt anticipé optionnel quand la MI se stabilise
#   - AMP retiré (réseau minuscule, et exp(matrice) en fp16 perdait de la précision)
#   - AUCUNE fenêtre : backend matplotlib "Agg", toutes les figures vont dans data/qc/figures
#
# Structure attendue :
#   data/mni_template/MNI152.nii
#   data/Reference/971882.nii
#   data/interim/            (créé automatiquement)

# %% 0 - Imports
import os
import math
import time
import random
from pathlib import Path

import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("Agg")                      # aucun pop-up : les figures sont écrites sur disque
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, sobel
from scipy.ndimage import zoom as scipy_zoom

# %% 1 - Configuration : chemins et réglages
try:
    _START = Path(__file__).resolve().parent
except NameError:                       # exécuté sans __file__ (notebook)
    _START = Path.cwd()


def find_project_root() -> Path:
    """Remonte l'arborescence jusqu'à trouver un dossier 'data/'."""
    for start in (_START, Path.cwd()):
        for p in [start, *start.parents]:
            if (p / "data").is_dir():
                return p
    raise FileNotFoundError(
        "Dossier 'data/' introuvable. Place ce fichier dans le projet "
        "ou fais un os.chdir(<racine du projet>) avant.")


ROOT = find_project_root()

# --- Entrées ---
MRI_PATH     = ROOT / "data/mni_template/MNI152.nii"      # fixe
PET_SRC_PATH = ROOT / "data/Reference/971882.nii"         # mobile (original)

# --- Fichiers produits ---
PET_FLIP_PATH  = ROOT / "data/Reference/971882.nii.gz"                # PET après flip Y
PET_REG_PATH   = ROOT / "data/Reference/PET_Ref_recale.nii.gz"        # PET recalé, valeurs d'origine
PET_CLEAN_PATH = ROOT / "data/Reference/PET_Ref_recale_clean.nii.gz"  # lissé + masqué (affichage)
OUT_DIR        = ROOT / "data/interim"
OUT_DIR.mkdir(parents=True, exist_ok=True)
M_PATH     = OUT_DIR / "M_affine_4x4.npy"
M_TXT_PATH = OUT_DIR / "M_affine_4x4.txt"
QC_FIG_DIR = ROOT / "data/qc/figures"
QC_FIG_DIR.mkdir(parents=True, exist_ok=True)

for _p in (MRI_PATH, PET_SRC_PATH):
    if not _p.exists():
        raise FileNotFoundError(f"Fichier introuvable : {_p}")

# --- Réglages ---
FLIP_PET_Y = True      # ton flip A-P (voir remarque dans la cellule 4)
ZOOM_ORDER = 1         # même interpolation à l'entraînement ET à l'application
DOWNSCALE  = 2.0
MAX_LEVELS = 4         # niveaux générés
L_USED     = 5         # niveaux utilisés
MAX_INDICES = 90000    # voxels candidats par niveau
EDGE_RATIO  = 0.7      # 70 % contours, 30 % aléatoires
N_ITERS     = 1000
LOG_EVERY   = 50
MAX_SAMPLES = 32768    # voxels par passage MINE
LAMBDA_REG  = 1e-4
SEED        = 0

# Arrêt anticipé : stoppe si la moyenne de la MI sur une fenêtre ne progresse plus
EARLY_STOP   = True
ES_WINDOW    = 100
ES_PATIENCE  = 3
ES_MIN_DELTA = 1e-3

# --- Aspect des figures ---
BG, FG_TITLE, FG_AXIS = "#0a0e1a", "#89b4fa", "#cdd6f4"

print("Racine du projet :", ROOT)

# %% 2 - Device et graine
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    torch.set_num_threads(os.cpu_count() or 1)
print("device :", device)
if device.type == "cuda":
    print("GPU    :", torch.cuda.get_device_name(0))
else:
    print("Pas de GPU détecté : vérifie l'installation de PyTorch avec CUDA (pytorch.org).")

# %% 3 - Fonctions utilitaires


def load_canonical(path):
    """Charge un NIfTI en orientation canonique (RAS) ; garde le 1er volume si 4D."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    vol = img.get_fdata(dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    return img, vol


def to_mri_grid(vol, shape, order=ZOOM_ORDER):
    """
    Met `vol` à la taille `shape` par étirement (scipy zoom).
    ATTENTION : la matrice M n'est valable que pour CE prétraitement. Il doit être
    identique à l'entraînement et à l'application (une seule fonction pour les deux).
    """
    shape = tuple(int(s) for s in shape)
    if vol.shape == shape:
        return vol.astype(np.float32)
    factors = tuple(s / r for s, r in zip(shape, vol.shape))
    out = scipy_zoom(vol, factors, order=order)
    res = np.zeros(shape, np.float32)                     # sécurité sur les arrondis
    sl = tuple(slice(0, min(a, b)) for a, b in zip(out.shape, shape))
    res[sl] = out[sl]
    return res


def robust_norm01(x, p_low=1, p_high=99):
    """Normalisation par percentiles vers [0, 1] (pour l'apprentissage uniquement)."""
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, p_low), np.percentile(x, p_high)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def gradient_magnitude_3d(vol):
    """Équivalent 3D de Canny : voxels dont le gradient de Sobel dépasse la médiane."""
    gx, gy, gz = sobel(vol, axis=0), sobel(vol, axis=1), sobel(vol, axis=2)
    mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    return (mag > np.percentile(mag, 50)).astype(np.float32)


def pyramid_gaussian_3d(vol, downscale, max_levels):
    """Pyramide gaussienne 3D : [niveau 0 = pleine résolution, 1, 2, ...]."""
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


def voxel_coords_normalized(idx, shape):
    """Indices plats d'un volume [D,H,W] -> coordonnées normalisées [N,3] (x=W, y=H, z=D).
    Identique à make_grid_3d(...).reshape(-1,3)[idx], sans construire toute la grille."""
    D, H, W = shape
    d, h, w = np.unravel_index(idx, shape)
    f = lambda i, n: -1.0 + 2.0 * i / max(n - 1, 1)       # == linspace(-1, 1, n)
    return np.stack([f(w, W), f(h, H), f(d, D)], axis=1).astype(np.float32)


def save_fig(fig, name, dpi=150):
    """Enregistre la figure dans data/qc/figures et la ferme (aucun pop-up)."""
    path = QC_FIG_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("Figure enregistrée :", path)
    return path


def quick_preview(vol, title, name, cmap="gray"):
    x, y, z = (s // 2 for s in vol.shape)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor(BG)
    for ax, pl, lbl in zip(axes, [vol[x], vol[:, y], vol[:, :, z]],
                           [f"Sagittal (x={x})", f"Coronal (y={y})", f"Axial (z={z})"]):
        ax.imshow(pl.T, cmap=cmap, origin="lower", aspect="auto")
        ax.set_title(lbl, color=FG_TITLE, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, color=FG_AXIS, fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, name)


print("Utilitaires prêts")

# %% 4 - Chargement des volumes
# REMARQUE sur le flip : np.flip(axis=1) sans modifier l'affine est un MIROIR (det = -1),
# pas une rotation. Une matrice affine apprise ensuite ne peut pas le défaire.
# Si le PET n'avait pas besoin de ce flip, les hémisphères sont inversés par rapport
# au template. Vérifie la latéralité avec le DICOM d'origine (ImageOrientationPatient).
if FLIP_PET_Y:
    _img, _vol = load_canonical(PET_SRC_PATH)
    _vol = np.ascontiguousarray(np.flip(_vol, axis=1))
    _out = nib.Nifti1Image(_vol, _img.affine)
    _out.set_data_dtype(np.float32)
    nib.save(_out, str(PET_FLIP_PATH))
    PET_PATH = PET_FLIP_PATH
    print("PET flippé (Y) sauvegardé :", PET_FLIP_PATH)
else:
    PET_PATH = PET_SRC_PATH

I_nib, I_raw = load_canonical(MRI_PATH)       # fixe
J_nib, J_src = load_canonical(PET_PATH)       # mobile, sur sa propre grille

zooms = lambda n: tuple(round(float(v), 3) for v in n.header.get_zooms()[:3])
print("I (fixe)   :", I_raw.shape, zooms(I_nib), "mm", nib.aff2axcodes(I_nib.affine))
print("J (mobile) :", J_src.shape, zooms(J_nib), "mm", nib.aff2axcodes(J_nib.affine))

# PET (valeurs d'origine) sur la grille de l'IRM : utilisé pour l'entraînement ET l'application
J_raw = to_mri_grid(J_src, I_raw.shape)
print("J après mise à la grille de l'IRM :", J_raw.shape)

quick_preview(I_raw, "I - Fixe (3 plans)", "preview_fixed.png")
quick_preview(J_raw, "J - Mobile (3 plans)", "preview_moving.png")

# %% 5 - Normalisation et pyramides
L = L_USED
I0 = robust_norm01(I_raw)
J0 = robust_norm01(J_raw)
I0_smooth = gaussian_filter(I0, sigma=1.0)
J0_smooth = gaussian_filter(J0, sigma=1.0)

pyramid_I = pyramid_gaussian_3d(I0_smooth, DOWNSCALE, MAX_LEVELS)
pyramid_J = pyramid_gaussian_3d(J0_smooth, DOWNSCALE, MAX_LEVELS)
L = min(L, len(pyramid_I), len(pyramid_J))
print(f"Niveaux disponibles : {len(pyramid_I)} | utilisés : L={L}")
for k in range(L):
    print(f"  Niveau {k}: I={pyramid_I[k].shape}  J={pyramid_J[k].shape}")

# %% 6 - Tenseurs et échantillonnage des voxels (tout est pré-calculé ici, une seule fois)
# Pour chaque niveau on garde :
#   J_lst[s]    : volume mobile [D,H,W]
#   pts_lst[s]  : coordonnées normalisées des voxels échantillonnés [N,3]
#   fix_lst[s]  : intensités de l'IRM en ces voxels [N]
J_lst, pts_lst, fix_lst = [], [], []
rng = np.random.default_rng(SEED)

for s in range(L):
    I_dhw = np.ascontiguousarray(np.transpose(pyramid_I[s].astype(np.float32), (2, 1, 0)))  # [X,Y,Z]->[D,H,W]
    J_dhw = np.ascontiguousarray(np.transpose(pyramid_J[s].astype(np.float32), (2, 1, 0)))
    D_, H_, W_ = I_dhw.shape
    total = D_ * H_ * W_

    grad_mask = gradient_magnitude_3d(I_dhw)
    edge_idx = np.flatnonzero(grad_mask.ravel() > 0)
    n_edge = int(MAX_INDICES * EDGE_RATIO)
    n_rand = MAX_INDICES - n_edge

    edge_pick = (rng.choice(edge_idx, size=n_edge, replace=False)
                 if edge_idx.size > n_edge else edge_idx)
    rand_pick = rng.choice(total, size=min(n_rand, total), replace=False)
    idx = np.unique(np.concatenate([edge_pick, rand_pick]))

    J_lst.append(torch.from_numpy(J_dhw).to(device))
    pts_lst.append(torch.from_numpy(voxel_coords_normalized(idx, (D_, H_, W_))).to(device))
    fix_lst.append(torch.from_numpy(I_dhw.ravel()[idx]).to(device))

    print(f"Niveau {s}: {D_}x{H_}x{W_} | contours={edge_pick.size} | échantillon={idx.size}")

# %% 7 - Modèles : AffineNet3D et MINE3D


class AffineNet3D(nn.Module):
    """
    Matrice affine 4x4 par exponentielle de matrice (9 DOF) :
      0-2 translations | 3-5 rotations | 6-8 échelles
    """

    def __init__(self, device, n_basis=9):
        super().__init__()
        B = torch.zeros(n_basis, 4, 4, dtype=torch.float32, device=device)
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
        self.vL = nn.Parameter(torch.zeros(n_basis, 1, 1, dtype=torch.float32, device=device))
        self.v1 = nn.Parameter(torch.zeros(n_basis, 1, 1, dtype=torch.float32, device=device))

    def forward(self, s: int) -> torch.Tensor:
        """Matrice 4x4. s=0 : inclut les paramètres fins v1."""
        C = torch.sum(self.B * self.vL, dim=0)
        if s == 0:
            C = C + torch.sum(self.B * self.v1, dim=0)
        return torch.linalg.matrix_exp(C)


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
        z = self._net(torch.cat([joint, marginal], dim=0))     # une seule passe réseau
        z_joint, z_marg = z[:n], z[n:]
        return z_joint.mean() - (torch.logsumexp(z_marg, dim=0) - math.log(n))


print("AffineNet3D et MINE3D définis")

# %% 8 - Warp 3D et perte multi-résolution


def warp_points(moving, M, pts):
    """
    moving [D,H,W] | M [4,4] | pts [N,3] normalisés -> intensités du mobile en M·pts, [N].
    Seuls les N points demandés sont interpolés (et non tout le volume).
    """
    ones = torch.ones(pts.shape[0], 1, device=pts.device, dtype=pts.dtype)
    p = torch.cat([pts, ones], dim=1) @ M.T
    p = p[:, :3] / p[:, 3:4].clamp_min(1e-6)
    out = F.grid_sample(moving[None, None], p.view(1, -1, 1, 1, 3), mode="bilinear",
                        padding_mode="zeros", align_corners=True)
    return out.view(-1)


def multi_resolution_loss_3d(affine_net, mine_net, J_lst, pts_lst, fix_lst, L,
                             max_samples=32768, lambda_reg=1e-4):
    loss = 0.0
    for s in range(L - 1, -1, -1):                       # grossier -> fin
        pts, fixed_vals = pts_lst[s], fix_lst[s]
        if pts.shape[0] > max_samples:
            sel = torch.randperm(pts.shape[0], device=pts.device)[:max_samples]
            pts, fixed_vals = pts[sel], fixed_vals[sel]
        moving_vals = warp_points(J_lst[s], affine_net(s), pts)
        loss = loss - (1.0 / L) * mine_net(fixed_vals, moving_vals)
    reg = affine_net.vL.pow(2).mean() + affine_net.v1.pow(2).mean()
    return loss + lambda_reg * reg


@torch.no_grad()
def apply_affine_to_volume(vol_xyz, M_np, device, chunk=2_000_000):
    """Applique une matrice 4x4 à un volume [X,Y,Z] déjà sur la grille de l'IRM (par blocs)."""
    dhw = np.ascontiguousarray(np.transpose(vol_xyz, (2, 1, 0)))
    D, H, W = dhw.shape
    moving = torch.from_numpy(dhw).float().to(device)
    M = torch.from_numpy(np.asarray(M_np, dtype=np.float32)).to(device)
    grid = make_grid_3d(D, H, W, device).reshape(-1, 3)
    out = torch.empty(grid.shape[0], device=device)
    for i in range(0, grid.shape[0], chunk):
        out[i:i + chunk] = warp_points(moving, M, grid[i:i + chunk])
    out = np.transpose(out.view(D, H, W).cpu().numpy(), (2, 1, 0))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


print("Warp et perte prêts")

# %% 9 - Initialisation des modèles et de l'optimiseur
affine_net = AffineNet3D(device).to(device)
mine_net = MINE3D(n_neurons=256, dropout_rate=0.1).to(device)

optimizer = optim.Adam(
    [
        {"params": mine_net.parameters(), "lr": 5e-4},
        {"params": affine_net.vL,         "lr": 2e-3},
        {"params": affine_net.v1,         "lr": 5e-4},
    ],
    amsgrad=True,
)

print("Paramètres AffineNet3D :", sum(p.numel() for p in affine_net.parameters()))
print("Paramètres MINE3D      :", sum(p.numel() for p in mine_net.parameters()))

# %% 10 - Boucle d'entraînement (logs texte, courbe enregistrée à la fin)
mi_curve = []
best_mean, bad_windows = -np.inf, 0
t0 = time.time()

for itr in range(N_ITERS):
    optimizer.zero_grad(set_to_none=True)
    loss = multi_resolution_loss_3d(
        affine_net, mine_net, J_lst, pts_lst, fix_lst,
        L=L, max_samples=MAX_SAMPLES, lambda_reg=LAMBDA_REG)
    loss.backward()
    optimizer.step()
    mi_curve.append((-loss).item())

    if (itr + 1) % LOG_EVERY == 0 or itr == 0:
        print(f"Iter {itr + 1}/{N_ITERS} | MI proxy: {mi_curve[-1]:.6f} | "
              f"elapsed: {time.time() - t0:.1f}s")

    if EARLY_STOP and (itr + 1) % ES_WINDOW == 0:
        cur = float(np.mean(mi_curve[-ES_WINDOW:]))
        if cur > best_mean + ES_MIN_DELTA:
            best_mean, bad_windows = cur, 0
        else:
            bad_windows += 1
            if bad_windows >= ES_PATIENCE:
                print(f"Arrêt anticipé à l'itération {itr + 1} (MI stable)")
                break

print(f"\nEntraînement terminé en {time.time() - t0:.1f}s ({len(mi_curve)} itérations)")

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(mi_curve, color="#6366f1", lw=0.8)
ax.set(title="Courbe d'entraînement (MI plus haute = meilleur alignement)",
       xlabel="Itération", ylabel="MI (proxy)")
ax.grid(alpha=0.3)
save_fig(fig, "training_curve.png")

# Sauvegarde immédiate de la matrice (celle utilisée à l'application)
with torch.no_grad():
    M_final = affine_net(0).cpu().numpy()
np.save(M_PATH, M_final)
np.savetxt(M_TXT_PATH, M_final, fmt="%.6f")
print("Matrice sauvegardée :", M_PATH, "et", M_TXT_PATH)
print(np.round(M_final, 6))

# %% 11 - PET recalé (normalisé) et figure 3 plans (enregistrée, sans widget)
I_full = robust_norm01(I_raw)
J_full = robust_norm01(J_raw)
J_warped = apply_affine_to_volume(J_full, M_final, device)


def save_triplanar(fixed, moving, warped, name, idx=None):
    nx, ny, nz = fixed.shape
    x, y, z = idx if idx is not None else (nx // 2, ny // 2, nz // 2)
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    fig.patch.set_facecolor(BG)
    cols = ["FIXE", "MOBILE (avant)", "RECALÉ (après)"]
    rows = ["SAGITTAL", "CORONAL", "AXIAL"]
    for c, vol in enumerate([fixed, moving, warped]):
        for r, sl in enumerate([vol[x].T, vol[:, y].T, vol[:, :, z].T]):
            ax = axes[r, c]
            ax.imshow(sl, cmap="gray", origin="lower", aspect="auto", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(cols[c], color=FG_TITLE, fontsize=9)
            if c == 0:
                ax.text(-0.04, 0.5, rows[r], transform=ax.transAxes, rotation=90,
                        color="#6c7086", fontsize=8, va="center", ha="right")
    plt.tight_layout()
    save_fig(fig, name)


save_triplanar(I_full, J_full, J_warped, "triplanar_before_after.png")

# %% 12 - Métriques simples (échantillon aléatoire dans l'avant-plan de l'IRM)
diff_before = np.abs(I_full - J_full)
diff_after = np.abs(I_full - J_warped)

fg = np.flatnonzero((I_full > 0.05).ravel())           # PAS les 500000 premiers voxels (= fond)
sel = np.random.choice(fg, size=min(500000, fg.size), replace=False)
ncc_b = float(np.corrcoef(I_full.ravel()[sel], J_full.ravel()[sel])[0, 1])
ncc_a = float(np.corrcoef(I_full.ravel()[sel], J_warped.ravel()[sel])[0, 1])
mse_b, mse_a = float(np.mean(diff_before ** 2)), float(np.mean(diff_after ** 2))
mae_b, mae_a = float(diff_before.mean()), float(diff_after.mean())

print(f"{'Métrique':<8}{'Avant':>12}{'Après':>12}")
print(f"{'MSE':<8}{mse_b:>12.5f}{mse_a:>12.5f}")
print(f"{'MAE':<8}{mae_b:>12.5f}{mae_a:>12.5f}")
print(f"{'NCC':<8}{ncc_b:>12.5f}{ncc_a:>12.5f}")
print("Rappel : en multimodal (IRM vs PET) MSE/MAE/NCC sont peu fiables ; juge sur la NMI (cellule 15).")

# %% 13 - Sauvegarde des résultats d'entraînement


def save_nifti(vol, ref_nib, path):
    img = nib.Nifti1Image(vol.astype(np.float32), ref_nib.affine, ref_nib.header)
    img.header.set_data_dtype(np.float32)
    nib.save(img, str(path))
    print(f"  Sauvegardé : {path}  ({os.path.getsize(path) / 1024 ** 2:.2f} Mo)")


save_nifti(I_full,      I_nib, OUT_DIR / "I_fixed_norm.nii.gz")
save_nifti(J_full,      I_nib, OUT_DIR / "J_moving_norm.nii.gz")
save_nifti(J_warped,    I_nib, OUT_DIR / "J_warped.nii.gz")
save_nifti(diff_before, I_nib, OUT_DIR / "diff_before.nii.gz")
save_nifti(diff_after,  I_nib, OUT_DIR / "diff_after.nii.gz")
np.save(OUT_DIR / "mi_curve.npy", np.array(mi_curve))
torch.save({"affine_net": affine_net.state_dict(), "mine_net": mine_net.state_dict(),
            "M_final": M_final, "mi_curve": mi_curve}, OUT_DIR / "model_weights.pt")
print("Résultats dans :", OUT_DIR)

# %% 14 - Application de M au PET brut (valeurs d'origine conservées)
# Cette cellule recharge la matrice depuis le disque : elle fonctionne aussi après un
# redémarrage du noyau, à condition d'avoir relancé les cellules 0 à 4 et 8.
M_loaded = np.load(M_PATH).astype(np.float32)
J_warped_raw = apply_affine_to_volume(J_raw, M_loaded, device)   # J_raw = même prétraitement

nib.save(nib.Nifti1Image(J_warped_raw.astype(np.float32), I_nib.affine), str(PET_REG_PATH))
print("Sauvegardé :", PET_REG_PATH)
print("forme :", J_warped_raw.shape, "| min/max :", J_warped_raw.min(), J_warped_raw.max())
print("Fraction > 0 :", (J_warped_raw > 0).mean(),
      "(~0.85 est normal : le PET brut contient du bruit de fond)")

# %% 15 - Évaluation : NMI, rapport de corrélation, Dice, HD95
ZOOMS = I_nib.header.get_zooms()[:3]


def clean_mask(m):
    m = ndi.binary_opening(m, iterations=1)
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + np.argmax(sizes))
    return ndi.binary_fill_holes(m)


def nmi_np(a, b, bins=32):
    def norm(x):
        lo, hi = np.percentile(x, [1, 99.5])
        return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    h, _, _ = np.histogram2d(norm(a), norm(b), bins=bins, range=[[0, 1], [0, 1]])
    p = h / h.sum()
    H = lambda q: -(q[q > 0] * np.log(q[q > 0])).sum()
    return (H(p.sum(1)) + H(p.sum(0))) / H(p)


def correlation_ratio(x, y, bins=32):
    edges = np.percentile(x, np.linspace(0, 100, bins + 1))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, bins - 1)
    mu = y.mean()
    num = sum(y[idx == k].size * (y[idx == k].mean() - mu) ** 2
              for k in range(bins) if (idx == k).any())
    return num / (((y - mu) ** 2).sum() + 1e-12)


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum() + 1e-12)


def surface(m):
    return m & ~ndi.binary_erosion(m)


def hd95(a, b, spacing):
    sa, sb = surface(a), surface(b)
    da = ndi.distance_transform_edt(~sb, sampling=spacing)[sa]
    db = ndi.distance_transform_edt(~sa, sampling=spacing)[sb]
    return np.percentile(np.hstack([da, db]), 95)


def make_pet_mask(x, frac=0.2):
    return clean_mask(x > frac * np.percentile(x, 99.5))


# Masque cerveau de l'IRM : nilearn si disponible, sinon seuil sur le template
try:
    from nilearn.datasets import load_mni152_brain_mask
    from nilearn.image import resample_to_img
    m_img = resample_to_img(load_mni152_brain_mask(),
                            nib.Nifti1Image(I_raw, I_nib.affine), interpolation="nearest")
    mask_irm = clean_mask(m_img.get_fdata() > 0.5)
    print("Masque : MNI152 brain mask (nilearn). Attention : ce masque peut venir d'une "
          "version du template légèrement différente de ton MNI152 (décalage de quelques mm).")
except Exception as e:
    print("nilearn indisponible :", e)
    if (I_raw > 0).mean() < 0.5:                       # template sans crâne, fond à 0
        mask_irm = clean_mask(I_raw > 0)
        print("Masque : voxels > 0 de l'IRM")
    else:
        from skimage.filters import threshold_otsu     # import local : scikit-image seulement ici
        mask_irm = clean_mask(I_raw > threshold_otsu(I_raw))
        print("Masque : Otsu (approximatif)")
print("Volume du masque (mL) :", round(mask_irm.sum() * np.prod(ZOOMS) / 1000, 1))

print("\n=== Contrôles ===")
print("forme == IRM :", J_warped_raw.shape == I_raw.shape)
print("NaN/Inf      :", int((~np.isfinite(J_warped_raw)).sum()))

mk_b = mask_irm & (J_raw > 0)
mk_a = mask_irm & (J_warped_raw > 0)
print("\n=== Alignement multimodal (dans le cerveau MNI) ===")
print(f"NMI avant M : {nmi_np(I_raw[mk_b], J_raw[mk_b]):.4f}")
print(f"NMI après M : {nmi_np(I_raw[mk_a], J_warped_raw[mk_a]):.4f}")
print(f"Rapport de corrélation avant : {correlation_ratio(I_raw[mk_b], J_raw[mk_b]):.4f}")
print(f"Rapport de corrélation après : {correlation_ratio(I_raw[mk_a], J_warped_raw[mk_a]):.4f}")

m_b, m_a = make_pet_mask(J_raw), make_pet_mask(J_warped_raw)
print("\n=== Masque PET vs cerveau MNI ===")
print(f"Dice    : {dice(m_b, mask_irm):.4f} -> {dice(m_a, mask_irm):.4f}")
print(f"HD95 mm : {hd95(m_b, mask_irm, ZOOMS):.2f} -> {hd95(m_a, mask_irm, ZOOMS):.2f}")

# %% 15b - Overlay IRM + PET recalé NON normalisé -> data/qc/figures
# Valeurs d'origine des deux volumes : seule la fenêtre d'affichage (vmin/vmax) est réglée.


def _pos(v):
    p = v[v > 0]
    return p if p.size else np.array([0.0, 1.0], dtype=np.float32)


def save_overlay_raw(mri, pet, zooms, out_name, idx=None, alpha=0.5, thr_frac=0.15):
    m_lo, m_hi = np.percentile(_pos(mri), [1, 99])
    p_lo, p_hi = np.percentile(_pos(pet), [1, 99.5])
    thr_val = p_lo + (p_hi - p_lo) * thr_frac               # seuil en unités réelles
    X, Y, Z = mri.shape
    x, y, z = idx if idx is not None else (X // 2, Y // 2, Z // 2)
    sx, sy, sz = zooms[:3]
    planes = [("Sagittal", mri[x].T, pet[x].T, sz / sy, x),
              ("Coronal", mri[:, y].T, pet[:, y].T, sz / sx, y),
              ("Axial", mri[:, :, z].T, pet[:, :, z].T, sy / sx, z)]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, (name, m_sl, p_sl, asp_, i) in zip(axs, planes):
        ax.imshow(m_sl, cmap="gray", origin="lower", aspect=asp_, vmin=m_lo, vmax=m_hi)
        im = ax.imshow(np.ma.masked_less(p_sl, thr_val), cmap="hot", alpha=alpha,
                       origin="lower", aspect=asp_, vmin=p_lo, vmax=p_hi)
        ax.set_title(f"{name} ({i})")
        ax.axis("off")
    fig.colorbar(im, ax=axs, shrink=0.8, label="PET recalé (valeurs d'origine)")
    fig.suptitle("IRM (MNI152) + PET recalé non normalisé", fontsize=12)
    save_fig(fig, out_name, dpi=200)


def save_axial_montage_raw(mri, pet, out_name, n=12, alpha=0.5, thr_frac=0.15):
    m_lo, m_hi = np.percentile(_pos(mri), [1, 99])
    p_lo, p_hi = np.percentile(_pos(pet), [1, 99.5])
    thr_val = p_lo + (p_hi - p_lo) * thr_frac
    zs = np.linspace(int(0.2 * mri.shape[2]), int(0.8 * mri.shape[2]), n).astype(int)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.6 * rows))
    for ax in np.atleast_1d(axs).ravel():
        ax.axis("off")
    for ax, z in zip(np.atleast_1d(axs).ravel(), zs):
        ax.imshow(mri[:, :, z].T, cmap="gray", origin="lower", vmin=m_lo, vmax=m_hi)
        ax.imshow(np.ma.masked_less(pet[:, :, z].T, thr_val), cmap="hot", alpha=alpha,
                  origin="lower", vmin=p_lo, vmax=p_hi)
        ax.set_title(f"z = {z}", fontsize=9)
    fig.suptitle("Coupes axiales : IRM + PET recalé non normalisé", fontsize=12)
    fig.tight_layout()
    save_fig(fig, out_name)


save_overlay_raw(I_raw, J_warped_raw, I_nib.header.get_zooms(), "overlay_irm_pet_recale_raw.png")
save_axial_montage_raw(I_raw, J_warped_raw, "overlay_axial_montage_raw.png")

# %% 16 - PET nettoyé (lissé + masqué) et overlay avec contours
# ATTENTION : le fichier "clean" est LISSÉ et MASQUÉ : il sert à l'affichage et au masque.
# Pour toute analyse quantitative, utilise PET_Ref_recale.nii.gz (valeurs d'origine).
mri = I_raw
pet = J_warped_raw
m_lo, m_hi = np.percentile(_pos(mri), [1, 99])

pet_s = ndi.gaussian_filter(pet, sigma=1.0)
thr = 0.25 * np.percentile(pet_s[pet_s > 0], 99.5)       # monte à 0.30 si du bruit persiste
m = pet_s > thr
m = ndi.binary_opening(m, iterations=2)
m = ndi.binary_erosion(m, iterations=4)                  # détache scalp et bruit
lab, n = ndi.label(m)
if n > 1:
    m = lab == (np.argmax(ndi.sum(m, lab, range(1, n + 1))) + 1)
m = ndi.binary_dilation(m, iterations=4)
pet_mask = ndi.binary_fill_holes(m)
print("Fraction masque :", round(float(pet_mask.mean()), 3), "(attendu ~0.15 à 0.30)")

pet_clean = np.where(pet_mask, pet_s, 0).astype(np.float32)
p_hi = np.percentile(pet_clean[pet_mask], 99.5)
nib.save(nib.Nifti1Image(pet_clean, I_nib.affine), str(PET_CLEAN_PATH))   # extension .nii.gz obligatoire
print("Sauvegardé :", PET_CLEAN_PATH)

sx, sy, sz = I_nib.header.get_zooms()[:3]
X, Y, Z = mri.shape
asp = {"sag": sz / sy, "cor": sz / sx, "axi": sy / sx}


def get(vol, plane, i):
    if plane == "sag":
        return vol[i].T
    if plane == "cor":
        return vol[:, i].T
    return vol[:, :, i].T


def save_contours(x, y, z, alpha=0.5, show_heat=True, show_irm_mask=True,
                  name="overlay_contours_pet_clean.png"):
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, plane, i, title in zip(axs, ["sag", "cor", "axi"], [x, y, z],
                                   ["Sagittal", "Coronal", "Axial"]):
        ax.imshow(get(mri, plane, i), cmap="gray", origin="lower",
                  aspect=asp[plane], vmin=m_lo, vmax=m_hi)
        if show_heat:
            heat = get(pet_clean, plane, i)
            ax.imshow(np.ma.masked_where(heat <= thr, heat), cmap="hot", alpha=alpha,
                      origin="lower", aspect=asp[plane], vmin=thr, vmax=p_hi)
        msk = get(pet_mask, plane, i).astype(float)
        if msk.min() != msk.max():                       # évite l'erreur sur une coupe vide
            ax.contour(msk, levels=[0.5], colors="red", linewidths=1.2, origin="lower")
        if show_irm_mask and "mask_irm" in globals():
            mi_ = get(mask_irm, plane, i).astype(float)
            if mi_.min() != mi_.max():
                ax.contour(mi_, levels=[0.5], colors="cyan", linewidths=1.2, origin="lower")
        ax.set_title(f"{title} ({i})")
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, name, dpi=200)


# Coupes centrales par défaut ; change (x, y, z) pour d'autres coupes
save_contours(X // 2, Y // 2, Z // 2)

# %% 17 - Contrôle d'en-tête : le PET a-t-il des valeurs d'origine intactes ?
for name, path in {"PET flippé (entrée)": PET_PATH, "PET recalé": PET_REG_PATH}.items():
    img = nib.load(str(path))
    hdr = img.header
    data = img.get_fdata(dtype=np.float32)
    finite = data[np.isfinite(data)]
    print(f"\n===== {name} =====")
    print("Type stocké :", np.asanyarray(img.dataobj).dtype, "| type NIfTI :", hdr.get_data_dtype())
    print("scl_slope :", hdr["scl_slope"], "| scl_inter :", hdr["scl_inter"])
    print("Min / Max / Moyenne :", finite.min(), finite.max(), finite.mean())