"""
03_segmentation.py : étape 2 du pipeline (segmentation du PET de référence recalé par l'atlas AAL3).

Entrées  (dans data/) :
    mni_template/MNI152.nii                 IRM de référence (grille du PET recalé)
    Reference/PET_Ref_recale.nii.gz         PET recalé, VALEURS D'ORIGINE (pas la version _clean)
    region_map/AAL3v1_1mm.nii               atlas AAL3 (volume d'étiquettes entières)
    region_labels/AAL3v1_1mm.nii.txt        correspondance index -> nom de région
    mni_template/..._09c_mask.nii           (optionnel) masque cerveau pour un contrôle de l'atlas

Sorties :
    data/segmentation/AAL3_sur_grille_PET.nii.gz       atlas rééchantillonné sur la grille du PET
    data/segmentation/pet_ref_parcellise_moyenne.nii.gz  PET où chaque région vaut sa moyenne
    data/segmentation/pet_ref_regional_stats.csv         statistiques par région
    data/qc/figures/qc_atlas_sur_irm.png, qc_atlas_sur_pet.png
    logs/03_segmentation.log

Aucune fenêtre n'est ouverte : toutes les figures sont enregistrées sur disque.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                               # aucun pop-up
import matplotlib.pyplot as plt                     # noqa: E402
import nibabel as nib                               # noqa: E402
import numpy as np                                  # noqa: E402
from nibabel.processing import resample_from_to     # noqa: E402
from scipy import ndimage as ndi                    # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════
# 1 - CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════


def find_project_root() -> Path:
    """Remonte l'arborescence jusqu'à trouver un dossier 'data/'."""
    try:
        start = Path(__file__).resolve().parent
    except NameError:                               # exécuté sans __file__ (notebook)
        start = Path.cwd()
    for base in (start, Path.cwd()):
        for p in [base, *base.parents]:
            if (p / "data").is_dir():
                return p
    raise FileNotFoundError("Dossier 'data/' introuvable : lance le script depuis le projet.")


ROOT = find_project_root()

# --- Entrées ---
MRI_PATH        = ROOT / "data/mni_template/MNI152.nii"
PET_REG_PATH    = ROOT / "data/Reference/PET_Ref_recale.nii.gz"
ATLAS_PATH      = ROOT / "data/region_map/AAL3v1_1mm.nii"
LABELS_PATH     = ROOT / "data/region_labels/AAL3v1_1mm.nii.txt"
BRAIN_MASK_PATH = ROOT / "data/mni_template/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii"  # optionnel

# --- Sorties ---
SEG_DIR    = ROOT / "data/segmentation"             # dossier de sortie de CETTE étape (créé s'il manque)
QC_FIG_DIR = ROOT / "data/qc/figures"
LOG_DIR    = ROOT / "logs"

# --- Réglages ---
MIN_COVERAGE   = 0.9     # une région couverte à moins de 90 % par le PET est signalée
REF_REGION_IDS = None    # ex. list(range(95, 121)) = cervelet + vermis, pour un ratio (SUVR).
                         # None = pas de normalisation (le choix dépend du traceur).

log = logging.getLogger("segmentation")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.StreamHandler(), logging.FileHandler(LOG_DIR / "03_segmentation.log",
                                                           mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


# ══════════════════════════════════════════════════════════════════════════════
# 2 - LECTURE DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════


def load_canonical(path: Path):
    """NIfTI en orientation canonique (RAS) ; premier volume si 4D. -> (image, ndarray)."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
        img = nib.Nifti1Image(data, img.affine)
    return img, data


def read_aal3_labels(txt_path: Path) -> dict[int, str]:
    """Lignes 'index nom [valeur]' -> {index: nom}.
    La 3e colonne peut manquer (lignes 35, 36, 81, 82) : seule l'index et le nom comptent."""
    labels = {}
    for line in Path(txt_path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            labels[int(parts[0])] = parts[1]
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# 3 - ATLAS SUR LA GRILLE DU PET
# ══════════════════════════════════════════════════════════════════════════════


def atlas_on_grid(atlas_img, shape, affine) -> np.ndarray:
    """Rééchantillonne l'atlas sur la grille (shape, affine) en PLUS PROCHE VOISIN (order=0).
    Toute autre interpolation créerait de fausses étiquettes (moyenne de 10 et 12 = 11)."""
    res = resample_from_to(atlas_img, (tuple(shape[:3]), affine), order=0, mode="constant", cval=0)
    return np.rint(res.get_fdata()).astype(np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# 4 - STATISTIQUES RÉGIONALES
# ══════════════════════════════════════════════════════════════════════════════


def region_stats(pet: np.ndarray, atlas: np.ndarray, n_max: int) -> dict:
    """Statistiques par région sur les voxels VALIDES du PET (finis et non nuls).
    Les zéros exacts sont le hors-champ créé par le warp : les inclure biaiserait la moyenne."""
    valid = np.isfinite(pet) & (pet != 0)
    inside = valid & (atlas > 0)

    cnt_atlas = np.bincount(atlas[atlas > 0], minlength=n_max)
    lab = atlas[inside]
    val = pet[inside].astype(np.float64)
    cnt = np.bincount(lab, minlength=n_max)
    s1 = np.bincount(lab, weights=val, minlength=n_max)
    s2 = np.bincount(lab, weights=val ** 2, minlength=n_max)

    mean = np.full(n_max, np.nan)
    np.divide(s1, cnt, out=mean, where=cnt > 0)
    ex2 = np.full(n_max, np.nan)
    np.divide(s2, cnt, out=ex2, where=cnt > 0)
    std = np.sqrt(np.maximum(ex2 - mean ** 2, 0))

    ids_ok = np.flatnonzero(cnt > 0)
    med = np.full(n_max, np.nan)
    med[ids_ok] = ndi.median(pet, labels=np.where(inside, atlas, 0), index=ids_ok)
    return {"cnt": cnt, "cnt_atlas": cnt_atlas, "mean": mean, "median": med, "std": std,
            "valid": valid}


def reference_mean(pet: np.ndarray, atlas: np.ndarray, valid: np.ndarray, ids) -> float:
    """Moyenne du PET dans l'ensemble des régions de référence (voxels valides)."""
    m = np.isin(atlas, ids) & valid
    if not m.any():
        raise ValueError("Aucun voxel valide dans les régions de référence.")
    return float(pet[m].mean())


def write_csv(path: Path, labels: dict, st: dict, vox_mm3: float, ref_mean: float | None) -> None:
    header = ["index", "region", "n_vox_valides", "n_vox_atlas", "couverture",
              "volume_mm3", "moyenne", "mediane", "ecart_type"]
    if ref_mean is not None:
        header.append("ratio_vs_reference")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in sorted(labels):
            if st["cnt_atlas"][i] == 0:
                continue
            row = [i, labels[i], int(st["cnt"][i]), int(st["cnt_atlas"][i]),
                   round(st["cnt"][i] / st["cnt_atlas"][i], 3),
                   round(st["cnt"][i] * vox_mm3, 1),
                   st["mean"][i], st["median"][i], st["std"][i]]
            if ref_mean is not None:
                row.append(st["mean"][i] / ref_mean)
            w.writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
# 5 - VOLUMES DE SORTIE
# ══════════════════════════════════════════════════════════════════════════════


def save_volumes(atlas: np.ndarray, mean: np.ndarray, affine: np.ndarray) -> None:
    seg = nib.Nifti1Image(atlas.astype(np.int16), affine)
    seg.set_data_dtype(np.int16)
    nib.save(seg, str(SEG_DIR / "AAL3_sur_grille_PET.nii.gz"))

    lut = np.where(np.isfinite(mean), mean, 0).astype(np.float32)   # table région -> moyenne
    parc = lut[atlas]                                                # 0 hors atlas
    nib.save(nib.Nifti1Image(parc, affine), str(SEG_DIR / "pet_ref_parcellise_moyenne.nii.gz"))
    log.info("Volumes sauvegardés dans %s", SEG_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# 6 - CONTRÔLES QUALITÉ
# ══════════════════════════════════════════════════════════════════════════════


def check_label_consistency(labels: dict, atlas: np.ndarray) -> None:
    in_img = set(np.unique(atlas).tolist()) - {0}
    log.info("Étiquettes dans l'image : %d | dans le .txt : %d", len(in_img), len(labels))
    log.info("Dans le .txt mais absentes de l'image : %s", sorted(set(labels) - in_img))
    log.info("Dans l'image mais absentes du .txt    : %s", sorted(in_img - set(labels)))


def check_laterality(labels: dict, atlas: np.ndarray, affine: np.ndarray, cnt_atlas) -> list:
    """En RAS, la gauche est à x < 0. Contrôle la position de l'atlas, pas celle du PET."""
    ids = [i for i in sorted(labels) if cnt_atlas[i] > 0]
    cms = ndi.center_of_mass(np.ones(atlas.shape, np.uint8), atlas, ids)
    xyz = nib.affines.apply_affine(affine, np.array(cms))
    bad = [labels[i] for i, (x, _, _) in zip(ids, xyz)
           if (labels[i].endswith("_L") and x > 0) or (labels[i].endswith("_R") and x < 0)]
    log.info("Régions du mauvais côté : %s", bad if bad else "aucune")
    return bad


def check_brain_mask_overlap(atlas: np.ndarray, shape, affine) -> None:
    """Compare l'atlas au masque cerveau du template (si présent).
    - Part de l'atlas située DANS le masque : doit être proche de 100 % si l'atlas est bien placé.
      Une valeur nettement plus basse suggère un décalage atlas/template.
    - Dice (indicatif seulement) : l'atlas ne couvre que la substance grise alors que le masque
      contient aussi la substance blanche, donc le Dice est naturellement bien inférieur à 1."""
    if not BRAIN_MASK_PATH.exists():
        log.info("Masque cerveau absent (%s) : contrôle ignoré", BRAIN_MASK_PATH.name)
        return
    m_img = nib.as_closest_canonical(nib.load(str(BRAIN_MASK_PATH)))
    m = resample_from_to(m_img, (tuple(shape[:3]), affine), order=0).get_fdata() > 0.5
    a = atlas > 0
    inside = (a & m).sum() / max(int(a.sum()), 1)
    dice = 2 * (a & m).sum() / (a.sum() + m.sum() + 1e-12)
    log.info("Part de l'atlas située dans le masque cerveau : %.1f %% (attendu : proche de 100 %%)",
             100 * inside)
    log.info("Dice atlas vs masque cerveau : %.3f (indicatif : l'atlas exclut la substance blanche)", dice)


def save_atlas_qc(base: np.ndarray, atlas: np.ndarray, zooms, name: str, title: str) -> None:
    pos = base[base > 0]
    lo, hi = np.percentile(pos if pos.size else np.array([0.0, 1.0]), [1, 99])
    idx = tuple(s // 2 for s in base.shape)
    sx, sy, sz = zooms[:3]
    asp = [sz / sy, sz / sx, sy / sx]
    sl = lambda v, k: [v[idx[0]].T, v[:, idx[1]].T, v[:, :, idx[2]].T][k]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.5))
    for k, (ax, nm) in enumerate(zip(axs, ["Sagittal", "Coronal", "Axial"])):
        ax.imshow(sl(base, k), cmap="gray", origin="lower", aspect=asp[k], vmin=lo, vmax=hi)
        ax.imshow(np.ma.masked_equal(sl(atlas, k), 0), cmap="nipy_spectral", alpha=0.4,
                  origin="lower", aspect=asp[k], interpolation="nearest")
        ax.set_title(f"{nm} ({idx[k]})")
        ax.axis("off")
    fig.suptitle(title)
    QC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = QC_FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure enregistrée : %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# 7 - PROGRAMME PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    setup_logging()
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    for p in (MRI_PATH, PET_REG_PATH, ATLAS_PATH, LABELS_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Fichier introuvable : {p}")
    log.info("Racine du projet : %s", ROOT)

    # 1) Données
    mri_img, mri = load_canonical(MRI_PATH)
    pet_img, pet = load_canonical(PET_REG_PATH)
    atlas_img, _ = load_canonical(ATLAS_PATH)
    labels = read_aal3_labels(LABELS_PATH)
    zooms = mri_img.header.get_zooms()[:3]
    log.info("PET recalé : %s | atlas : %s | %d régions dans le .txt",
             pet.shape, atlas_img.shape, len(labels))

    if pet.shape != mri.shape or not np.allclose(pet_img.affine, mri_img.affine, atol=1e-3):
        raise ValueError("Le PET recalé n'est pas sur la grille de l'IRM : relance l'étape de recalage.")

    # 2) Atlas sur la grille du PET (plus proche voisin)
    atlas = atlas_on_grid(atlas_img, pet.shape, pet_img.affine)
    check_label_consistency(labels, atlas)

    # 3) Statistiques par région sur le PET brut
    n_max = int(max(atlas.max(), max(labels))) + 1
    st = region_stats(pet, atlas, n_max)
    ref_mean = None
    if REF_REGION_IDS:
        ref_mean = reference_mean(pet, atlas, st["valid"], REF_REGION_IDS)
        log.info("Moyenne de la région de référence : %.5f", ref_mean)

    vox_mm3 = float(np.prod(zooms))
    csv_path = SEG_DIR / "pet_ref_regional_stats.csv"
    write_csv(csv_path, labels, st, vox_mm3, ref_mean)
    log.info("Tableau : %s", csv_path)

    low = [labels[i] for i in sorted(labels)
           if st["cnt_atlas"][i] > 0 and st["cnt"][i] / st["cnt_atlas"][i] < MIN_COVERAGE]
    log.info("Régions couvertes à moins de %.0f %% par le PET : %s", 100 * MIN_COVERAGE, low or "aucune")

    # 4) Volumes de sortie
    save_volumes(atlas, st["mean"], pet_img.affine)

    # 5) Contrôles qualité
    check_laterality(labels, atlas, pet_img.affine, st["cnt_atlas"])
    check_brain_mask_overlap(atlas, pet.shape, pet_img.affine)
    save_atlas_qc(mri, atlas, zooms, "qc_atlas_sur_irm.png", "AAL3 sur IRM MNI152")
    save_atlas_qc(pet, atlas, zooms, "qc_atlas_sur_pet.png", "AAL3 sur PET recalé")
    log.info("Segmentation terminée.")


if __name__ == "__main__":
    main()