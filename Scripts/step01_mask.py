"""Étape 1 : créer UN masque cérébral à partir du PET de référence.

Idée : le cerveau est la zone la plus brillante ; le scalp et la nuque le sont moins.
  1. seuillage          -> on garde les voxels assez brillants
  2. plus grande pièce  -> on enlève les petits îlots isolés
  3. fermeture          -> on referme les petites fentes
  4. comblement         -> on bouche les cavités (ventricules...) coupe par coupe, puis en 3D
  5. nettoyage          -> on ne garde que la plus grosse pièce
Le PET lui-même n'est jamais modifié : le masque est une image à part (1 ou 0).

Le seuil est une FRACTION d'une valeur de référence "robuste" (un percentile élevé,
pas le maximum, car le maximum peut être un seul voxel isolé).

Lancer :  python Scripts\\step01_mask.py
Résultats : le masque + 01_brain_mask.png + 01_brain_mask_montage.png (12 coupes axiales)
            + 01_mask_threshold_comparison.png (plusieurs seuils côte à côte pour choisir le bon).
"""
import numpy as np
from scipy import ndimage

import io_utils
import qc
from config import load_config


def make_brain_mask(data, threshold_fraction, percentile=99.5, closing_iterations=2):
    """Renvoie un masque 3D (uint8, 1 = cerveau, 0 = hors cerveau)."""
    positive = data[data > 0]
    if positive.size == 0:
        raise ValueError("Le volume est vide (aucune valeur positive).")
    reference_value = np.percentile(positive, percentile)

    # 1. Seuillage
    binary = data > threshold_fraction * reference_value

    # 2. On numérote chaque "pièce" connectée, puis on garde la plus grosse.
    labeled, n_pieces = ndimage.label(binary)
    if n_pieces == 0:
        raise ValueError("Aucun voxel au-dessus du seuil : baissez mask.threshold_fraction.")
    sizes = ndimage.sum(binary, labeled, index=range(1, n_pieces + 1))
    mask = labeled == (int(np.argmax(sizes)) + 1)

    # 3. Fermeture : referme les fentes de quelques voxels (dans une boîte autour du cerveau, plus rapide).
    if closing_iterations > 0:
        idx = np.argwhere(mask)
        pad = closing_iterations + 1
        low = np.maximum(idx.min(axis=0) - pad, 0)
        high = np.minimum(idx.max(axis=0) + pad + 1, mask.shape)
        box = tuple(slice(a, b) for a, b in zip(low, high))
        mask[box] = ndimage.binary_closing(mask[box], iterations=closing_iterations)

    # 4. Comblement : d'abord coupe axiale par coupe (remplit les ventricules entourés par le cortex),
    #    puis en 3D (cavités fermées restantes).
    for k in range(mask.shape[2]):
        mask[:, :, k] = ndimage.binary_fill_holes(mask[:, :, k])
    mask = ndimage.binary_fill_holes(mask)

    # 5. Nettoyage final : on ne garde que la plus grosse pièce (supprime les petits fragments flottants).
    labeled, n_pieces = ndimage.label(mask)
    if n_pieces > 1:
        sizes = ndimage.sum(mask, labeled, index=range(1, n_pieces + 1))
        mask = labeled == (int(np.argmax(sizes)) + 1)
    return mask.astype(np.uint8)


def _volume_ml(mask, img):
    return mask.sum() * float(np.prod(img.header.get_zooms()[:3])) / 1000


def run(cfg):
    paths = cfg["paths"]
    settings = cfg["mask"]
    percentile = settings.get("reference_percentile", 99.5)
    closing = settings.get("closing_iterations", 2)

    reference = io_utils.load_nifti(paths["reference_pet"])
    io_utils.check_is_volume(reference, "PET de référence")
    data = io_utils.get_data_float32(reference)

    # --- Le masque retenu ---
    mask = make_brain_mask(data, settings["threshold_fraction"], percentile, closing)
    io_utils.save_nifti(mask, reference, paths["brain_mask"])
    qc.save_three_views(data, paths["qc_dir"] / "01_brain_mask.png",
                        f"Masque cérébral, seuil {settings['threshold_fraction']} (contour rouge)", mask=mask)
    qc.save_axial_montage(data, mask, paths["qc_dir"] / "01_brain_mask_montage.png")

    # --- Comparaison de plusieurs seuils (pour choisir) ---
    fractions = settings.get("try_thresholds", [])
    if fractions:
        masks, titles = [], []
        for f in fractions:
            m = make_brain_mask(data, f, percentile, closing)
            masks.append(m)
            titles.append(f"seuil {f:.2f} : {_volume_ml(m, reference):.0f} mL")
        qc.save_mask_comparison(data, masks, titles, paths["qc_dir"] / "01_mask_threshold_comparison.png")

    print(f"Masque enregistré : {paths['brain_mask']}")
    print(f"Seuil {settings['threshold_fraction']} : {mask.mean() * 100:.1f} % des voxels, "
          f"volume ≈ {_volume_ml(mask, reference):.0f} mL (un cerveau adulte fait environ 1100 à 1400 mL)")
    return paths["brain_mask"]


if __name__ == "__main__":
    run(load_config())