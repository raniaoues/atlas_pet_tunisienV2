"""Étape 0 : inspecter TOUTES les entrées avant de les utiliser (on ne modifie rien).

On vérifie : les 80 sujets + la référence + le MNI152 + la carte de régions (AAL3).
Résultats : 00_inspection.csv, deux figures, et un bilan en 3 niveaux :
  [info]      : constat utile
  [attention] : à surveiller, pas bloquant
  ANOMALIE    : à corriger avant de continuer

Lancer :  python Scripts\\step00_inspect.py
"""
import csv

import numpy as np

import io_utils
import qc
from config import load_config

PLATEAU_MIN_VOXELS = 10   # au-delà, un même maximum partagé par N voxels fait penser à un plafond


def _check_region_map(region_img, mni_img, problems, notes):
    """Vérifie que la carte de régions est exploitable et compatible avec le MNI152."""
    data = np.asarray(region_img.dataobj)
    if not np.allclose(data, np.round(data)):
        problems.append("carte de régions : valeurs non entières (ce n'est pas une carte d'étiquettes)")
    labels = np.unique(data[data != 0])
    if len(labels) == 0:
        problems.append("carte de régions : aucune région (volume vide)")
        return
    notes.append(f"carte de régions : {len(labels)} régions, numéros de {int(labels.min())} à {int(labels.max())}")

    rel = io_utils.grid_relation(region_img, mni_img)
    if not rel["same_linear"]:
        notes.append("carte de régions et MNI152 : voxels ou axes différents -> rééchantillonnage (plus proche voisin) à l'étape 3")
    elif rel["integer_shift"]:
        notes.append(f"carte de régions et MNI152 : même voxel, décalage entier {rel['shift_voxels']} voxels "
                     "-> simple recopie, sans interpolation")
    else:
        problems.append(f"carte de régions et MNI152 : décalage non entier {rel['shift_voxels']} voxels")


def run(cfg):
    paths = cfg["paths"]
    rows, problems, warnings, notes = [], [], [], []

    # --- La référence ---
    reference = io_utils.load_nifti(paths["reference_pet"])
    io_utils.check_is_volume(reference, "PET de référence")
    ref_info = io_utils.describe(reference)
    rows.append({"name": "reference_pet", "role": "reference", **ref_info})

    # --- Les sujets : tous les .nii / .nii.gz de raw_dir ---
    subject_files = sorted(paths["raw_dir"].glob("*.nii*"))
    if not subject_files:
        raise FileNotFoundError(f"Aucun volume NIfTI trouvé dans {paths['raw_dir']}")
    if paths["reference_pet"].name in {f.name for f in subject_files}:
        problems.append("la référence est aussi dans raw_dir : elle serait comptée deux fois")

    plateau = []
    for file in subject_files:
        img = io_utils.load_nifti(file)
        io_utils.check_is_volume(img, file.name)
        info = io_utils.describe(img)
        rows.append({"name": file.name, "role": "subject", **info})

        if info["has_nan"]:
            problems.append(f"{file.name} : contient des NaN")
        for key in ("shape", "voxel_size_mm", "orientation"):   # un sujet doit ressembler à la référence
            if info[key] != ref_info[key]:
                problems.append(f"{file.name} : {key} = {info[key]} (référence : {ref_info[key]})")
        if info["n_voxels_at_max"] >= PLATEAU_MIN_VOXELS:
            plateau.append(f"{file.name} (max {info['max']:.0f} atteint par {info['n_voxels_at_max']} voxels)")
    if plateau:
        warnings.append(f"{len(plateau)} sujet(s) avec un plateau au maximum (écrêtage possible) : " + " ; ".join(plateau))

    # --- MNI152 (tête entière) ---
    mni = io_utils.load_nifti(paths["mni_template"], "Voir paths.mni_template dans config.yaml.")
    io_utils.check_is_volume(mni, "MNI152")
    mni_info = io_utils.describe(mni)
    rows.append({"name": "mni_template", "role": "mni", **mni_info})
    if mni_info["orientation"] != ref_info["orientation"]:
        problems.append(f"MNI152 : orientation {mni_info['orientation']} (PET : {ref_info['orientation']})")

    # --- Carte de régions (AAL3) ---
    region_map = io_utils.load_nifti(paths["region_map"], "Voir paths.region_map dans config.yaml.")
    io_utils.check_is_volume(region_map, "carte de régions")
    rows.append({"name": "region_map", "role": "region_map", **io_utils.describe(region_map)})
    _check_region_map(region_map, mni, problems, notes)
    if not paths["region_labels"].exists():
        warnings.append(f"fichier de noms de régions absent : {paths['region_labels'].name} "
                        "(utile pour nommer les régions dans les résultats)")

    # --- Écriture du tableau et des figures ---
    out_csv = paths["inspection_csv"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    qc.save_three_views(io_utils.get_data_float32(reference),
                        paths["qc_dir"] / "00_reference_pet.png", "PET de référence")
    qc.save_three_views(io_utils.get_data_float32(mni),
                        paths["qc_dir"] / "00_mni152.png", "MNI152")

    # --- Bilan ---
    print(f"{len(rows)} volumes inspectés -> {out_csv}")
    for n in notes:
        print("[info]", n)
    for w in warnings:
        print("[attention]", w)
    for p in problems:
        print("ANOMALIE :", p)
    if not problems:
        print("Aucune anomalie bloquante.")
    return problems


if __name__ == "__main__":
    run(load_config())