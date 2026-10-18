"""eval_registration_metrics.py : mesures d'évaluation d'un recalage produit par step02_register_ref.py.

Calcule, AVANT (départ) et APRÈS recalage : Dice et couverture des masques, NMI avec la T1, corrélation,
distances de surface (moyenne et HD95, en mm) entre le masque cérébral du PET déplacé et le masque MNI,
et les échelles anatomiques de la transformation en mm. Écrit metrics.json (lu par build_report.py).

Usage (depuis le dossier qui contient config.yaml et Scripts/) :
  python Scripts\\eval_registration_metrics.py --tfm transforms\\02_dl.tfm --fixed <IRM_MNI.nii.gz> \\
      --fixed-mask <masque_MNI.nii.gz> --moving <PET.nii.gz> --moving-mask <masque_PET.nii.gz> --pitch 0
  --pitch : inclinaison (degrés) du départ si vous voulez comparer au même départ que le script (défaut 0).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step02_register_ref_V2 as S          # réutilise les mêmes conventions et fonctions que le recalage


def warped_mask(T, fixed_mask, A_f, moving_mask, A_m):
    """Masque du PET déplacé sur la grille fixe (booléen), mêmes conventions que report()."""
    import torch
    A_f_t = torch.tensor(A_f, dtype=torch.float64, device=S.DEVICE)
    A_m_inv = torch.tensor(np.linalg.inv(A_m), dtype=torch.float64, device=S.DEVICE)
    T_t = torch.tensor(T, dtype=torch.float64, device=S.DEVICE)
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in fixed_mask.shape], indexing="ij"), -1).reshape(-1, 3)
    vals, valid = S.sample_moving(torch.tensor(moving_mask.astype(np.float32), device=S.DEVICE)[None, None], T_t,
                                  torch.tensor(grid, dtype=torch.float64, device=S.DEVICE), A_f_t, A_m_inv,
                                  moving_mask.shape)
    return ((vals > 0.5) & valid).cpu().numpy().reshape(fixed_mask.shape)


def surface_distances(a, b, spacing):
    """Distances (mm) entre les surfaces de deux masques booléens : (moyenne symétrique, HD95)."""
    if not a.any() or not b.any():
        return float("nan"), float("nan")
    sa = a & ~ndimage.binary_erosion(a)
    sb = b & ~ndimage.binary_erosion(b)
    d_to_b = ndimage.distance_transform_edt(~sb, sampling=spacing)[sa]
    d_to_a = ndimage.distance_transform_edt(~sa, sampling=spacing)[sb]
    both = np.concatenate([d_to_b, d_to_a])
    return float(both.mean()), float(np.percentile(both, 95))


def evaluate(T, fixed, A_f, moving_reg, A_m, fixed_mask, moving_mask, t1):
    rep = S.report(T, fixed, A_f, moving_reg, A_m, fixed_mask, t1, moving_mask)
    wm = warped_mask(T, fixed_mask, A_f, moving_mask, A_m)
    spacing = np.sqrt((A_f[:3, :3] ** 2).sum(axis=0))
    msd, hd95 = surface_distances(wm, fixed_mask, spacing)
    return rep, msd, hd95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfm", required=True)
    ap.add_argument("--fixed", required=True, help="image fixe (IRM MNI ou pseudo-PET) sur la grille du masque")
    ap.add_argument("--fixed-mask", required=True)
    ap.add_argument("--moving", required=True)
    ap.add_argument("--moving-mask", required=True)
    ap.add_argument("--t1", help="IRM pour la NMI (défaut : --fixed)")
    ap.add_argument("--pitch", type=float, default=0.0, help="inclinaison du départ de référence (degrés)")
    ap.add_argument("--dilate", type=int, default=3, help="dilatation du masque PET (comme moving_dilate_vox)")
    ap.add_argument("--out", default="metrics.json")
    a = ap.parse_args()
    import SimpleITK as sitk

    fixed, A_f = S.load_volume(a.fixed)
    fm, A_fm = S.load_volume(a.fixed_mask)
    fm = fm > 0
    moving, A_m = S.load_volume(a.moving)
    mm = S.load_volume(a.moving_mask)[0] > 0
    t1 = S.load_volume(a.t1)[0] if a.t1 else fixed
    if fixed.shape != fm.shape or not np.allclose(A_f, A_fm, atol=1e-3):
        raise SystemExit("L'image fixe et le masque MNI n'ont pas la même grille.")
    moving_reg = S.mask_moving(moving, mm, a.dilate)

    T_end = S.itk_to_ras(sitk.ReadTransform(a.tfm))
    T_start, _ = S.initial_matrix(A_f, fm, A_m, mm, a.pitch)

    r0, msd0, hd0 = evaluate(T_start, fixed, A_f, moving_reg, A_m, fm, mm, t1)
    r1, msd1, hd1 = evaluate(T_end, fixed, A_f, moving_reg, A_m, fm, mm, t1)

    # échelles anatomiques en mm : A_m^-1 . T . A_f restreint à la partie linéaire (voxels fixes -> voxels mobiles),
    # puis on revient aux mm : partie linéaire de T seule (RAS fixe -> RAS mobile) = facteurs de taille mm / mm
    sv = np.linalg.svd(T_end[:3, :3], compute_uv=False)
    out = {
        "dice_start": r0["dice_masque"], "dice_final": r1["dice_masque"],
        "cov_start": r0["masque_couvert"], "cov_final": r1["masque_couvert"],
        "nmi_start": r0.get("nmi_avec_t1"), "nmi_final": r1.get("nmi_avec_t1"),
        "corr_start": r0["correlation"], "corr_final": r1["correlation"],
        "msd_start": msd0, "msd_final": msd1, "hd95_start": hd0, "hd95_final": hd1,
        "scale_min": float(sv.min()), "scale_max": float(sv.max()), "anisotropy": float(sv.max() / sv.min()),
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(f"{k:<14} {v}")
    print(f"\nÉcrit : {a.out}  ->  python build_report.py {a.out}")


if __name__ == "__main__":
    main()