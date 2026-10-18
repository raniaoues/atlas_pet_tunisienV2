"""Lecture / écriture de volumes NIfTI et vérifications de base.

Règle d'or : on ne modifie JAMAIS data/RAW. On lit depuis les entrées,
on écrit ailleurs (interim, processed, transforms).
"""
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
def load_nifti(path, hint=""):
    """Charge un volume NIfTI. `hint` précise d'où il devrait venir, pour un message d'erreur clair."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Volume introuvable : {path}. {hint}".strip())
    return nib.load(str(path))


def get_data_float32(img):
    """Renvoie les voxels en float32 (valeurs brutes, sans aucune normalisation)."""
    return np.asarray(img.dataobj, dtype=np.float32)


def save_nifti(data, reference_img, path):
    """Sauvegarde `data` avec la géométrie (affine) de `reference_img`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)   # crée le dossier si besoin
    out = nib.Nifti1Image(np.asarray(data), reference_img.affine)
    nib.save(out, str(path))


def check_is_volume(img, name="volume"):
    """Vérifie que le fichier est bien UN volume 3D (x, y, z), pas un 4D."""
    if len(img.shape) != 3:
        raise ValueError(f"{name} : volume 3D attendu, forme trouvée {img.shape}")


def describe(img):
    """Résumé d'un volume : dimensions, voxel, orientation, type, plage de valeurs."""
    data = get_data_float32(img)
    vmax = float(data.max())
    return {
        "shape": tuple(img.shape),
        "voxel_size_mm": tuple(float(z) for z in img.header.get_zooms()[:3]),
        "orientation": "".join(nib.aff2axcodes(img.affine)),
        "dtype": str(img.get_data_dtype()),
        "min": float(data.min()),
        "max": vmax,
        "n_voxels_at_max": int((data == vmax).sum()),   # beaucoup de voxels au max = plateau (écrêtage ?)
        "has_nan": bool(np.isnan(data).any()),
    }


def same_grid(img_a, img_b, tol=1e-3):
    """Vrai si deux volumes ont les mêmes dimensions ET la même position/orientation."""
    return img_a.shape == img_b.shape and np.allclose(img_a.affine, img_b.affine, atol=tol)


def grid_relation(img_a, img_b, tol=1e-3):
    """Compare deux grilles en millimètres.

    same_linear   : mêmes voxels et mêmes axes
    shift_voxels  : décalage de l'origine de A vers B, en voxels
    integer_shift : le décalage est un nombre entier de voxels
                    (alors passer de A à B = simple recopie, sans interpolation)
    """
    A, B = img_a.affine, img_b.affine
    if not np.allclose(A[:3, :3], B[:3, :3], atol=tol):
        return {"same_linear": False, "shift_voxels": None, "integer_shift": False}
    shift = np.linalg.solve(A[:3, :3], B[:3, 3] - A[:3, 3])
    return {"same_linear": True,
            "shift_voxels": tuple(round(float(s), 3) for s in shift),
            "integer_shift": bool(np.allclose(shift, np.round(shift), atol=tol))}
def resample_nearest(src_img, target_img):
    """Place `src_img` sur la grille de `target_img` par PLUS PROCHE VOISIN (pour des numéros de régions).

    Le calcul se fait en coordonnées en millimètres (via les affines) : l'orientation de stockage
    des deux fichiers n'a donc pas d'importance. Aucune valeur intermédiaire n'est créée.
    """
    M = np.linalg.inv(src_img.affine) @ target_img.affine
    data = np.asarray(src_img.dataobj)
    out = ndimage.affine_transform(data.astype(np.float32), M[:3, :3], offset=M[:3, 3],
                                   output_shape=target_img.shape, order=0, mode="constant", cval=0)
    return np.rint(out).astype(data.dtype)
