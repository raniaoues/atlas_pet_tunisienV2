"""Compare VOTRE MNI152 (celui de config.yaml) au T1 ICBM 2009c que vous venez de trouver.

Pourquoi : dans vos fichiers de sortie, la grille de votre MNI152 a son origine en z à -74 mm, alors que celle du
T1 ICBM 2009c (même taille 193 x 229 x 193) est à -78 mm. Si c'est la même image avec un en-tête différent,
elle est décalée de 4 mm en z par rapport à l'atlas AAL3 (qui est dans le repère MNI standard). Or le pseudo-PET
mélangeait justement AAL3 et votre T1.

Le script dit : mêmes dimensions ou non, quelle est l'origine de chacun, et de combien de voxels il faut décaler
l'un pour qu'il ressemble le plus à l'autre (corrélation).

Lancer :  python Scripts\\compare_mni.py data\\mni_template\\mni_icbm152_t1_tal_nlin_asym_09c.nii
(le chemin est celui du T1 officiel ; le fichier .nii.gz marche aussi)
"""
import sys

import nibabel as nib
import numpy as np

from config import load_config


def overlap(a, b, shift):
    """Parties communes de a et b quand a[i + shift] est comparé à b[i]."""
    sa, sb = [], []
    for n, k in zip(a.shape, shift):
        if k >= 0:
            sa.append(slice(k, n)); sb.append(slice(0, n - k))
        else:
            sa.append(slice(0, n + k)); sb.append(slice(-k, n))
    return a[tuple(sa)], b[tuple(sb)]


def correlation_at(a, b, shift):
    x, y = overlap(a, b, shift)
    x, y = x[::3, ::3, ::3].ravel(), y[::3, ::3, ::3].ravel()
    return float(np.corrcoef(x, y)[0, 1])


def main(official_path):
    cfg = load_config()
    mine = nib.load(str(cfg["paths"]["mni_template"]))
    official = nib.load(official_path)
    print(f"Votre MNI152   : forme {mine.shape}, voxel {tuple(round(float(z), 2) for z in mine.header.get_zooms()[:3])} mm, "
          f"origine (mm) {np.round(mine.affine[:3, 3], 1)}")
    print(f"T1 ICBM 2009c  : forme {official.shape}, voxel {tuple(round(float(z), 2) for z in official.header.get_zooms()[:3])} mm, "
          f"origine (mm) {np.round(official.affine[:3, 3], 1)}")
    if mine.shape != official.shape:
        print("\nFormes différentes : ce ne sont pas la même grille. Ce ne sont probablement pas la même version du template.")
        return
    a = np.asarray(mine.dataobj, dtype=np.float32)
    b = np.asarray(official.dataobj, dtype=np.float32)

    print("\nRecherche du décalage (en voxels) qui rend les deux images les plus semblables...")
    best = [0, 0, 0]
    print(f"  sans décalage : corrélation {correlation_at(a, b, tuple(best)):.4f}")
    for _ in range(2):                                   # on cherche axe par axe, deux passes
        for axis in (2, 1, 0):
            scores = {}
            for k in range(-8, 9):
                shift = list(best); shift[axis] = k
                scores[k] = correlation_at(a, b, tuple(shift))
            best[axis] = max(scores, key=scores.get)
    print(f"  meilleur décalage (x, y, z) = {tuple(best)} voxels, corrélation {correlation_at(a, b, tuple(best)):.4f}")
    print("\nLecture :")
    print("  - corrélation proche de 1,000 avec décalage (0, 0, 0) : même image, même place.")
    print("  - corrélation proche de 1,000 avec un décalage non nul (par exemple 4 en z) : c'est la MÊME image,")
    print("    mais votre fichier est positionné autrement dans l'espace : à corriger (utilisez le T1 officiel).")
    print("  - corrélation nettement plus basse (par exemple < 0,95) : ce sont deux templates différents.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage : python Scripts\\compare_mni.py chemin\\vers\\mni_icbm152_t1_tal_nlin_asym_09c.nii")
    main(sys.argv[1])