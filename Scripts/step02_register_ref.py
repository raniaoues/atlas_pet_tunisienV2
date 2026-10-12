"""Étape 2 (pipeline 1) : recaler le PET de référence sur l'IRM MNI152.

  image MOBILE = PET de référence  (c'est elle qu'on déplace ; ses valeurs ne sont JAMAIS modifiées)
  image FIXE   = pseudo-PET fabriqué à partir du MNI152 et de la carte de régions AAL3
                 (elle a la même grille que le MNI152 : le résultat est donc bien "PET dans le repère MNI152")

POURQUOI UN PSEUDO-PET ?
  Le FDG-PET et l'IRM ne se ressemblent pas (cortex brillant en PET, gris en IRM ; substance blanche sombre en
  PET, claire en IRM). Avec l'information mutuelle, l'ordinateur hésite : trois inclinaisons très différentes
  donnent presque le même score. On lui donne donc comme cible une image qui RESSEMBLE à un PET :
      régions AAL3 (cortex, noyaux gris, cervelet) = brillant (1,0)
      substance blanche (claire sur l'IRM)          = moyen   (0,35)
      ventricules, fond                             = noir    (0)
  puis un flou de 8 mm (la résolution typique d'un PET). Les deux images ayant le même aspect, on peut utiliser
  la CORRÉLATION, qui est bien plus nette que l'information mutuelle. La corrélation ignore l'échelle globale
  des valeurs : aucune normalisation d'intensité n'est nécessaire (consigne du médecin respectée).

LES ÉTAPES
  1. DÉPART : on essaie des inclinaisons de la tête (de -60 à +60 degrés) et des petits décalages. Pour chacun
     on mesure la CORRÉLATION entre le PET et le pseudo-PET (sur une image réduite : c'est rapide).
     On garde plusieurs candidats, pas un seul.
  2. RAFFINEMENT de chaque candidat par corrélation : rigide (6 paramètres), puis similarité (7 : + une échelle),
     puis affine complet (12 paramètres).
  3. CHOIX du modèle : on ne passe à un modèle plus libre que s'il améliore NETTEMENT la corrélation ET reste
     plausible (pas de rotation absurde, pas d'écrasement, pas de cisaillement énorme).
  4. CHOIX du candidat : meilleure corrélation. Une galerie montre tous les candidats avec les mêmes coupes
     (c'est elle qui tranche en cas de doute : `candidate_override` dans config.yaml).
  5. GARDE-FOUS et sauvegardes. Le PET d'origine est interpolé UNE seule fois, à la fin.

Lancer :  python Scripts\\step02_register_ref.py
"""
import csv
import itertools
import time

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial.transform import Rotation

import io_utils
import qc
from config import load_config


# ----------------------------------------------------------------------------------------------
# Outils de lecture et de géométrie
# ----------------------------------------------------------------------------------------------
def _read_float32(path):
    """Lit un volume avec SimpleITK (qui gère la géométrie en mm) et le convertit en float32."""
    return sitk.Cast(sitk.ReadImage(str(path)), sitk.sitkFloat32)


def _array_to_image(array_xyz, like, pixel_type=sitk.sitkUInt8):
    """Tableau (x, y, z) de nibabel -> image SimpleITK ayant la géométrie de `like`."""
    array = np.transpose(array_xyz, (2, 1, 0))
    array = array.astype(np.float32 if pixel_type == sitk.sitkFloat32 else np.uint8)
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(like)
    return image


def _image_to_array(image):
    """Image SimpleITK -> tableau (x, y, z) comme dans nibabel."""
    return np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))


def describe_geometry(name, image):
    """Affiche la géométrie d'un volume : taille, voxel, boîte et centre en millimètres."""
    size = image.GetSize()
    corners = np.array([image.TransformIndexToPhysicalPoint([int(i) for i in idx])
                        for idx in itertools.product(*[(0, s - 1) for s in size])])
    low, high = corners.min(axis=0), corners.max(axis=0)
    print(f"  {name} : {size[0]}x{size[1]}x{size[2]} voxels de {np.round(image.GetSpacing(), 2)} mm | "
          f"boîte {np.round(high - low).astype(int)} mm | centre {np.round((low + high) / 2).astype(int)} mm")
    return high - low


def _silhouette(label_array):
    """Silhouette pleine du cerveau à partir d'une carte de régions."""
    mask = ndimage.binary_closing(label_array > 0, iterations=3)
    for k in range(mask.shape[2]):
        mask[:, :, k] = ndimage.binary_fill_holes(mask[:, :, k])
    return ndimage.binary_fill_holes(mask)


def make_mni_masks(region_img, mni_img, dilation_mm):
    """Renvoie (silhouette, silhouette élargie de `dilation_mm`) du cerveau dans l'espace MNI152."""
    silhouette = _silhouette(io_utils.resample_nearest(region_img, mni_img))
    voxel = float(min(mni_img.header.get_zooms()[:3]))
    steps = int(round(dilation_mm / voxel))
    wide = ndimage.binary_dilation(silhouette, iterations=steps) if steps > 0 else silhouette
    return silhouette.astype(np.uint8), wide.astype(np.uint8)


# ----------------------------------------------------------------------------------------------
# 0. Le pseudo-PET (image FIXE du recalage)
# ----------------------------------------------------------------------------------------------
def make_pseudo_pet(mni_img, region_img, wm_value=0.35, fwhm_mm=8.0, wm_t1_fraction=0.75):
    """Fabrique une image qui ressemble à un PET, exactement sur la grille du MNI152.

    - régions de la carte AAL3 -> 1,0 (brillant)
    - substance blanche (dans la silhouette, hors régions, claire sur l'IRM) -> `wm_value`
    - tout le reste (ventricules, os, fond) -> 0
    - flou gaussien de `fwhm_mm` (FWHM, largeur à mi-hauteur)
    """
    t1 = io_utils.get_data_float32(mni_img)
    cortex = io_utils.resample_nearest(region_img, mni_img) > 0
    brain = _silhouette(cortex)
    reference = np.percentile(t1[brain], 99)
    white = brain & ~cortex & (t1 > wm_t1_fraction * reference)
    pseudo = cortex * 1.0 + white * wm_value
    sigma_voxels = fwhm_mm / 2.355 / float(np.mean(mni_img.header.get_zooms()[:3]))
    return ndimage.gaussian_filter(pseudo.astype(np.float32), sigma_voxels)


def _correlation(a, b):
    """Corrélation de Pearson entre deux séries de valeurs (de -1 à 1, plus c'est haut mieux c'est)."""
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


# ----------------------------------------------------------------------------------------------
# 1. Départ : balayage des inclinaisons, noté par la corrélation avec le pseudo-PET
# ----------------------------------------------------------------------------------------------
def _initial_rigid(fixed_mask, moving_mask):
    """Alignement grossier : on fait coïncider les centres de gravité des deux masques cérébraux."""
    return sitk.CenteredTransformInitializer(
        sitk.Cast(fixed_mask, sitk.sitkFloat32), sitk.Cast(moving_mask, sitk.sitkFloat32),
        sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)


def _start_transform(base, pitch_deg, shift_mm):
    """Transformation de départ : l'alignement des centres, plus une inclinaison et un décalage."""
    start = sitk.Euler3DTransform(base)
    start.SetRotation(np.deg2rad(pitch_deg), 0.0, 0.0)
    start.SetTranslation(tuple(np.array(base.GetTranslation()) + np.array(shift_mm)))
    return start


def scan_pose(base, pseudo, wide_mask, moving_smooth, ones, settings):
    """Pour chaque inclinaison, meilleure corrélation PET / pseudo-PET (sur image réduite 4 fois).

    Les points où le PET n'a pas de données (hors champ de vue) sont ignorés, mais une position qui
    ne laisse presque plus rien à comparer est écartée (`pose_min_coverage`).
    Renvoie une liste de (inclinaison, corrélation, décalage en mm).
    """
    coarse = sitk.Shrink(pseudo, [4, 4, 4])
    target = sitk.GetArrayFromImage(coarse)
    region = sitk.GetArrayFromImage(sitk.Shrink(wide_mask, [4, 4, 4])) > 0
    min_cover = float(settings.get("pose_min_coverage", 0.4))
    low, high = settings["pose_pitch_range_deg"]
    reach = int(settings["pose_shift_range_mm"])
    curve = []
    for pitch in range(int(low), int(high) + 1, int(settings["pose_pitch_step_deg"])):
        best = (-1.0, (0.0, 0.0, 0.0))
        for dy, dz in itertools.product(range(-reach, reach + 1, 4), repeat=2):
            start = _start_transform(base, pitch, (0.0, dy, dz))
            moved = sitk.GetArrayFromImage(
                sitk.Resample(moving_smooth, coarse, start, sitk.sitkLinear, 0.0, sitk.sitkFloat32))
            valid = sitk.GetArrayFromImage(
                sitk.Resample(ones, coarse, start, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)) > 0
            used = region & valid
            if used.sum() < min_cover * region.sum():
                continue
            score = _correlation(target[used], moved[used])
            if np.isfinite(score) and score > best[0]:
                best = (score, (0.0, float(dy), float(dz)))
        curve.append((pitch, best[0], best[1]))
    return curve


def pick_candidates(curve, count, min_separation):
    """Garde les `count` meilleures inclinaisons, espacées d'au moins `min_separation` degrés."""
    chosen = []
    for entry in sorted(curve, key=lambda c: -c[1]):
        if entry[1] < -0.5:        # aucune position valable pour cette inclinaison
            continue
        if all(abs(entry[0] - other[0]) >= min_separation for other in chosen):
            chosen.append(entry)
        if len(chosen) == count:
            break
    if not chosen:
        raise RuntimeError("Aucune position de départ n'a de recouvrement suffisant entre PET et MNI152. "
                           "Vérifiez la géométrie affichée plus haut (ou baissez registration.pose_min_coverage).")
    return chosen


def save_pose_curve(curve, chosen, path):
    """Courbe du balayage de départ : corrélation selon l'inclinaison essayée."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pitches = [c[0] for c in curve]
    scores = [c[1] if c[1] > -0.5 else np.nan for c in curve]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(pitches, scores, "-o", ms=3)
    for k, c in enumerate(chosen, start=1):
        ax.axvline(c[0], color="red", ls="--", lw=0.8)
        ax.text(c[0], np.nanmin(scores), f" {k}", color="red", va="bottom")
    ax.set_xlabel("inclinaison essayée (degrés)")
    ax.set_ylabel("corrélation PET / pseudo-PET")
    ax.set_title("Départ : un vrai pic net = bon signe (candidats en rouge)")
    ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# 2. Raffinement par corrélation
# ----------------------------------------------------------------------------------------------
def _run_stage(fixed, moving, transform, settings, fixed_mask):
    """Un étage de recalage par corrélation : modifie `transform` sur place."""
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsCorrelation()
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(settings["sampling_percentage"], settings["random_seed"])
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetMetricFixedMask(fixed_mask)

    # Descente de gradient. Les "scales" mettent degrés et millimètres sur le même pied.
    reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=settings["iterations"],
                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10,
                                      estimateLearningRate=reg.EachIteration)
    reg.SetOptimizerScalesFromPhysicalShift()

    # Pyramide de résolutions : d'abord les grandes structures (image floue et petite), puis le détail.
    reg.SetShrinkFactorsPerLevel(settings["shrink_factors"])
    reg.SetSmoothingSigmasPerLevel(settings["smoothing_sigmas_mm"])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    reg.SetInitialTransform(transform, inPlace=True)
    reg.Execute(fixed, moving)


def refine(fixed, moving, start, settings, fixed_mask):
    """Départ -> rigide (6 paramètres) -> similarité (7) -> affine complet (12). `start` n'est pas modifié."""
    rigid = sitk.Euler3DTransform(start)
    _run_stage(fixed, moving, rigid, settings, fixed_mask)

    similarity = sitk.Similarity3DTransform()
    similarity.SetCenter(rigid.GetCenter())
    similarity.SetMatrix(rigid.GetMatrix())
    similarity.SetTranslation(rigid.GetTranslation())
    _run_stage(fixed, moving, similarity, settings, fixed_mask)

    affine = sitk.AffineTransform(3)
    affine.SetCenter(similarity.GetCenter())
    affine.SetMatrix(similarity.GetMatrix())
    affine.SetTranslation(similarity.GetTranslation())
    _run_stage(fixed, moving, affine, settings, fixed_mask)
    return rigid, similarity, affine


# ----------------------------------------------------------------------------------------------
# 3. Mesures indépendantes de l'optimiseur
# ----------------------------------------------------------------------------------------------
def make_evaluator(fixed, pseudo_array, moving, moving_smooth, ones, brain_mask, silhouette_array, wide_array):
    """Fabrique `evaluate(transform)` : corrélation, couverture et Dice, calculés pareil pour tous.

    corr     : corrélation PET / pseudo-PET dans le cerveau élargi, là où le PET a des données (LA mesure clé)
    coverage : part de la silhouette du cerveau couverte par le champ de vue du PET
    dice     : recouvrement des formes (information seulement, ne sert plus à décider)
    """
    silhouette = silhouette_array > 0
    wide = wide_array > 0

    def evaluate(transform):
        valid = _image_to_array(sitk.Resample(ones, fixed, transform, sitk.sitkNearestNeighbor,
                                              0, sitk.sitkUInt8)) > 0
        smooth = _image_to_array(sitk.Resample(moving_smooth, fixed, transform, sitk.sitkLinear,
                                               0.0, sitk.sitkFloat32))
        shown = _image_to_array(sitk.Resample(moving, fixed, transform, sitk.sitkLinear,
                                              0.0, sitk.sitkFloat32))
        moved = _image_to_array(sitk.Resample(brain_mask, fixed, transform, sitk.sitkNearestNeighbor,
                                              0, sitk.sitkUInt8)) > 0
        region = wide & valid
        metrics = {
            "corr": _correlation(pseudo_array[region], smooth[region]) if region.sum() > 1000 else float("nan"),
            "coverage": float((silhouette & valid).sum() / silhouette.sum()),
            "dice": float(2.0 * (moved & silhouette).sum() / (moved.sum() + silhouette.sum())),
        }
        return metrics, np.where(valid, shown, 0.0).astype(np.float32)

    return evaluate


# ----------------------------------------------------------------------------------------------
# Figures de contrôle
# ----------------------------------------------------------------------------------------------
def save_detailed_qc(fixed, moving, brain_mask, silhouette_array, transforms, output_path):
    """Contrôle visuel : IRM MNI (gris) + PET en couleurs ; contour cyan = silhouette AAL3 ; rouge = masque PET.

    Les valeurs du PET ne sont pas modifiées ; la fenêtre de couleur sert seulement à l'affichage.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fixed_arr = _image_to_array(fixed)
    silhouette = np.asarray(silhouette_array) > 0
    coords = np.argwhere(silhouette)
    if len(coords) == 0:
        raise ValueError("La silhouette MNI est vide.")
    cx, cy, cz = (int(np.median(coords[:, i])) for i in range(3))
    planes = [("Sagittale", lambda a: a[cx, :, :].T),
              ("Coronale", lambda a: a[:, cy, :].T),
              ("Axiale", lambda a: a[:, :, cz].T)]

    results = []
    for name, transform in transforms:
        pet = _image_to_array(sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32))
        mask = _image_to_array(sitk.Resample(brain_mask, fixed, transform, sitk.sitkNearestNeighbor,
                                             0, sitk.sitkUInt8)) > 0
        results.append((name, pet, mask))

    positive = np.concatenate([p[(p > 0) & m] for _, p, m in results if np.any((p > 0) & m)])
    vmin, vmax = np.percentile(positive, [2, 98])
    vmax = max(vmax, vmin + 1e-6)

    fig, axes = plt.subplots(len(results), 3, figsize=(15, 4 * len(results)), squeeze=False)
    for row, (name, pet, pet_mask) in enumerate(results):
        for col, (plane_name, extract) in enumerate(planes):
            ax = axes[row, col]
            pet_slice, mask_slice = extract(pet), extract(pet_mask.astype(float))
            ax.imshow(extract(fixed_arr), cmap="gray", origin="lower", interpolation="nearest")
            alpha = ((pet_slice > vmin) & (mask_slice > 0)).astype(float) * 0.65
            ax.imshow(np.clip((pet_slice - vmin) / (vmax - vmin), 0, 1), cmap="hot", origin="lower",
                      alpha=alpha, vmin=0, vmax=1, interpolation="nearest")
            sil_slice = extract(silhouette.astype(float))
            if sil_slice.any():
                ax.contour(sil_slice, levels=[0.5], colors=["cyan"], linewidths=1.0)
            if mask_slice.any():
                ax.contour(mask_slice, levels=[0.5], colors=["red"], linewidths=1.0)
            ax.set_title(f"{name} : {plane_name}", fontsize=10)
            ax.axis("off")
    fig.suptitle("Contrôle détaillé du recalage PET vers MNI152\nCyan : silhouette AAL3 | Rouge : masque PET transformé",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_pseudo_check(pseudo, pet, silhouette, path, title=""):
    """LE contrôle le plus parlant : pseudo-PET (ligne 1) et PET recalé (ligne 2), mêmes coupes.

    Sur la ligne 2, le contour rouge est la frontière cortex / substance blanche du pseudo-PET.
    Si le recalage est bon : le cortex brillant du PET est DANS le contour, les ventricules sombres du PET
    sont au même endroit que ceux du pseudo-PET.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = np.argwhere(silhouette > 0)
    cx, cy, cz = (int(np.median(coords[:, i])) for i in range(3))
    cuts = [("Sagittale", lambda a: a[cx].T), ("Coronale", lambda a: a[:, cy].T), ("Axiale", lambda a: a[:, :, cz].T)]
    vmax_pet = float(np.percentile(pet[pet > 0], 99.5))
    level = 0.6 * float(np.percentile(pseudo[pseudo > 0], 99))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.4))
    for col, (name, cut) in enumerate(cuts):
        axes[0, col].imshow(cut(pseudo), cmap="gray", origin="lower")
        axes[0, col].set_title(f"{name} : pseudo-PET (cible)", fontsize=10)
        axes[1, col].imshow(cut(pet), cmap="gray", origin="lower", vmin=0, vmax=vmax_pet)
        axes[1, col].contour(cut(pseudo), levels=[level], colors="red", linewidths=0.8)
        axes[1, col].set_title(f"{name} : PET recalé + contour du pseudo-PET", fontsize=10)
        for row in (0, 1):
            axes[row, col].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Description et garde-fous de la transformation
# ----------------------------------------------------------------------------------------------
def describe_transform(transform):
    """Résume la transformation PET -> IRM : déterminant, rotation, échelles et cisaillements.

    Décomposition B = Q x K : Q = rotation de la tête, K = échelles et cisaillements dans les AXES DU PET
    (x gauche-droite, y avant-arrière, z haut-bas). Le cisaillement y-z est celui d'une inclinaison de gantry
    non corrigée. Un cisaillement donne des étirements principaux inégaux (`scales`) SANS que la tête ait
    changé de taille.
    """
    A = np.array(transform.GetMatrix()).reshape(3, 3)       # FIXE -> MOBILE
    det = float(np.linalg.det(A))
    B = np.linalg.inv(A)                                    # MOBILE -> FIXE : ce qu'on fait subir au PET
    S = np.linalg.svd(B, compute_uv=False)

    Q, K = np.linalg.qr(B)
    signs = np.where(np.diag(K) < 0, -1.0, 1.0)
    Q, K = Q * signs, K * signs[:, None]
    angles = Rotation.from_matrix(Q if np.linalg.det(Q) > 0 else -Q).as_euler("xyz", degrees=True)
    shear = {"xy": np.degrees(np.arctan2(K[0, 1], K[1, 1])),
             "xz": np.degrees(np.arctan2(K[0, 2], K[2, 2])),
             "yz": np.degrees(np.arctan2(K[1, 2], K[2, 2]))}
    return {"determinant": det, "rotation_deg": angles, "scales": S,
            "axis_scales": np.diag(K), "shear_deg": shear}


def check_transform(info, settings):
    """Garde-fous. Renvoie (problèmes bloquants, avertissements)."""
    problems, warnings = [], []
    if info["determinant"] <= 0:
        problems.append("effet MIROIR (déterminant <= 0)")
    biggest = float(np.abs(info["rotation_deg"]).max())
    if biggest > settings["max_rotation_deg"]:
        problems.append(f"rotation de {biggest:.0f} degrés (au-delà de {settings['max_rotation_deg']}) : tête retournée ?")
    elif biggest > settings["warn_rotation_deg"]:
        warnings.append(f"rotation de {biggest:.0f} degrés : fréquent si la tête était penchée dans le TEP, "
                        "mais vérifiez bien la figure de contrôle")
    low, high = settings["scale_range"]
    if info["axis_scales"].min() < low or info["axis_scales"].max() > high:
        problems.append(f"échelle par axe suspecte {np.round(info['axis_scales'], 3)} (plage acceptée {low} à {high})")
    worst_shear = max(abs(v) for v in info["shear_deg"].values())
    if worst_shear > settings["max_shear_deg"]:
        problems.append(f"cisaillement de {worst_shear:.0f} degrés (au-delà de {settings['max_shear_deg']})")
    elif abs(info["shear_deg"]["yz"]) > 8:
        warnings.append(f"cisaillement y-z de {info['shear_deg']['yz']:.0f} degrés : typique d'une inclinaison de gantry "
                        "non corrigée à la conversion DICOM -> NIfTI (balise DICOM 0018,1120 : GantryDetectorTilt)")
    return problems, warnings


def _row(label, transform, metrics, extra=""):
    info = describe_transform(transform)
    return {"etape": label,
            "rotation_x_deg": round(float(info["rotation_deg"][0]), 1),
            "rotation_y_deg": round(float(info["rotation_deg"][1]), 1),
            "rotation_z_deg": round(float(info["rotation_deg"][2]), 1),
            "echelle_x": round(float(info["axis_scales"][0]), 3),
            "echelle_y": round(float(info["axis_scales"][1]), 3),
            "echelle_z": round(float(info["axis_scales"][2]), 3),
            "cisaillement_yz_deg": round(float(info["shear_deg"]["yz"]), 1),
            "correlation": round(metrics["corr"], 4), "couverture": round(metrics["coverage"], 3),
            "dice": round(metrics["dice"], 3), "remarque": extra}


def _print_table(rows):
    print(f"  {'étape':38s} {'rot.x':>6s} {'échelles x/y/z':>17s} {'cis.yz':>7s} {'corr.':>7s} {'couv.':>6s} {'Dice':>6s}")
    for r in rows:
        print(f"  {r['etape']:38s} {r['rotation_x_deg']:6.1f} "
              f"{r['echelle_x']:5.2f}/{r['echelle_y']:4.2f}/{r['echelle_z']:4.2f} "
              f"{r['cisaillement_yz_deg']:6.1f}° {r['correlation']:7.4f} {r['couverture']:6.2f} {r['dice']:6.3f} {r['remarque']}")


def decide_model(settings, transforms, metrics):
    """Choisit rigide / similarité / affine. `transforms` et `metrics` : dicts indexés par ces trois noms.

    Mode `auto` : on part du rigide ; on passe à un modèle plus libre seulement s'il est plausible ET gagne
    au moins `auto_min_corr_gain` de corrélation sur le modèle précédemment retenu.
    """
    forced = settings.get("model", "auto")
    if forced in ("rigid", "similarity", "affine"):
        return forced, f"imposé par model: {forced}"
    min_gain = float(settings.get("auto_min_corr_gain", 0.01))
    choice, reason = "rigid", "point de départ"
    for name in ("similarity", "affine"):
        problems, _ = check_transform(describe_transform(transforms[name]), settings)
        gain = metrics[name]["corr"] - metrics[choice]["corr"]
        if problems:
            reason += f" ; {name} refusé ({problems[0]})"
        elif gain >= min_gain:
            choice, reason = name, f"{name} gagne {gain:+.4f} de corrélation"
        else:
            reason += f" ; {name} : gain insuffisant ({gain:+.4f})"
    return choice, reason


# ----------------------------------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------------------------------
def run(cfg):
    started = time.time()
    stamp = time.strftime("%d/%m/%Y %H:%M")
    paths, settings = cfg["paths"], cfg["registration"]

    # --- Entrées ---
    for key, hint in (("brain_mask", "Lancez d'abord step01_mask.py."),
                      ("region_map", "Elle sert à fabriquer le pseudo-PET (voir paths.region_map).")):
        if not paths[key].exists():
            raise FileNotFoundError(f"Fichier introuvable : {paths[key]}. {hint}")
    fixed = _read_float32(paths["mni_template"])
    moving = _read_float32(paths["reference_pet"])
    mni_img = io_utils.load_nifti(paths["mni_template"])
    region_img = io_utils.load_nifti(paths["region_map"])
    brain_mask = sitk.Cast(sitk.ReadImage(str(paths["brain_mask"])), sitk.sitkUInt8)

    print("Géométrie des volumes (millimètres) :")
    describe_geometry("MNI152 (fixe)  ", fixed)
    pet_box = describe_geometry("PET (mobile)   ", moving)
    if pet_box.min() < 150:
        print(f"[attention] une dimension du PET ne fait que {pet_box.min():.0f} mm : le champ de vue peut couper le cerveau.")

    # --- Masques du MNI152 et pseudo-PET ---
    silhouette_array, wide = make_mni_masks(region_img, mni_img, settings["fixed_mask_dilation_mm"])
    io_utils.save_nifti(wide, mni_img, paths["mni_brain_mask"])
    fixed_mask = _array_to_image(wide, fixed)

    pseudo = make_pseudo_pet(mni_img, region_img,
                             wm_value=settings.get("pseudo_wm_value", 0.35),
                             fwhm_mm=settings.get("pseudo_fwhm_mm", 8.0),
                             wm_t1_fraction=settings.get("pseudo_wm_t1_fraction", 0.75))
    pseudo_path = paths.get("pseudo_pet", paths["interim_dir"] / "02_pseudo_pet.nii.gz")
    io_utils.save_nifti(pseudo, mni_img, pseudo_path)
    qc.save_three_views(pseudo, paths["qc_dir"] / "02_pseudo_pet.png", "Pseudo-PET : image fixe du recalage")
    fixed_reg = _array_to_image(pseudo, fixed, sitk.sitkFloat32)

    # --- Image mobile utilisée POUR LE CALCUL du recalage (le PET d'origine, ou une copie sans scalp) ---
    # Dans tous les cas, c'est le PET d'ORIGINE qui est rééchantillonné à la fin.
    moving_reg = moving
    if settings.get("mask_pet_for_registration", False):
        radius = int(settings.get("registration_mask_dilation_vox", 3))
        around_brain = sitk.BinaryDilate(brain_mask, [radius] * 3)
        moving_reg = moving * sitk.Cast(around_brain, sitk.sitkFloat32)
        print(f"[info] le calcul du recalage utilise une COPIE du PET sans scalp (masque dilaté de {radius} voxels).")

    smooth_mm = float(settings.get("scan_smoothing_mm", 4.0))
    moving_reg_smooth = sitk.SmoothingRecursiveGaussian(moving_reg, smooth_mm)
    moving_eval_smooth = sitk.SmoothingRecursiveGaussian(moving, 3.0)   # pour la mesure de corrélation uniquement
    ones = sitk.Image(moving.GetSize(), sitk.sitkUInt8) + 1             # carte "le PET a des données ici"
    ones.CopyInformation(moving)

    evaluate = make_evaluator(fixed, pseudo, moving, moving_eval_smooth, ones, brain_mask, silhouette_array, wide)

    # --- 1. Départ ---
    print("1) Balayage des inclinaisons de départ (corrélation PET / pseudo-PET)...")
    base = _initial_rigid(fixed_mask, brain_mask)
    curve = scan_pose(base, fixed_reg, fixed_mask, moving_reg_smooth, ones, settings)
    candidates = pick_candidates(curve, settings["n_candidates"], settings["candidate_min_separation_deg"])
    save_pose_curve(curve, candidates, paths["qc_dir"] / "02_pose_scan.png")
    valid_scores = [c[1] for c in curve if c[1] > -0.5]
    spread = max(valid_scores) - min(valid_scores)
    print(f"   corrélation de {min(valid_scores):.3f} à {max(valid_scores):.3f} (écart {spread:.3f})"
          + (" : courbe plate, le départ est peu discriminant" if spread < 0.05 else ""))

    # --- 2. Raffinement de chaque candidat ---
    print(f"2) Raffinement par corrélation de {len(candidates)} candidats (modèle : {settings.get('model', 'auto')})...")
    results = []
    for k, (pitch, score0, shift) in enumerate(candidates, start=1):
        start = _start_transform(base, pitch, shift)
        rigid, similarity, affine = refine(fixed_reg, moving_reg, start, settings, fixed_mask)
        transforms = {"rigid": rigid, "similarity": similarity, "affine": affine}
        shown, metrics_by = {}, {}
        for name, t in transforms.items():
            metrics_by[name], shown[name] = evaluate(t)
        final, reason = decide_model(settings, transforms, metrics_by)
        transform, metrics = transforms[final], metrics_by[final]
        info = describe_transform(transform)
        problems, _ = check_transform(info, settings)
        results.append({"k": k, "pitch": pitch, "start": start, "transforms": transforms,
                        "metrics_by": metrics_by, "shown": shown, "final": final, "reason": reason,
                        "transform": transform, "metrics": metrics, "problems": problems})
        print(f"   candidat {k} (départ {pitch:+d}°) -> {final} ({reason}) : rotation {np.round(info['rotation_deg'], 1)}°, "
              f"corrélation {metrics['corr']:.4f}, couverture {metrics['coverage']:.2f}"
              + (f"  REFUSÉ : {problems[0]}" if problems else ""))

    # --- 3. Choix du candidat : meilleure corrélation parmi les candidats plausibles ---
    pool = [r for r in results if not r["problems"]] or results
    override = settings.get("candidate_override")
    if override:
        chosen = results[int(override) - 1]
        print(f"3) Candidat imposé par candidate_override : {chosen['k']}")
    else:
        chosen = max(pool, key=lambda r: r["metrics"]["corr"])
        print(f"3) Candidat retenu : {chosen['k']} (meilleure corrélation : {chosen['metrics']['corr']:.4f})")
        close = [r for r in pool if r is not chosen and chosen["metrics"]["corr"] - r["metrics"]["corr"] < 0.005]
        if close:
            print(f"[attention] le candidat {close[0]['k']} est presque aussi bon (écart < 0,005) : regardez "
                  "02_candidates.png ; s'ils ont des rotations très différentes, tranchez avec candidate_override.")
    if chosen["metrics"]["corr"] < 0.5:
        print(f"[attention] corrélation finale de {chosen['metrics']['corr']:.2f} seulement : "
              "regardez 02_pseudo_pet.png (le pseudo-PET ressemble-t-il à un PET ?) puis 02_pseudo_check.png.")

    candidate_rows = [_row(f"candidat {r['k']} (départ {r['pitch']:+d}°), {r['final']}", r["transform"], r["metrics"],
                           "<= retenu" if r is chosen else "") for r in results]
    qc.save_gallery([(f"candidat {r['k']} : départ {r['pitch']:+d}°, {r['final']}, corr. {r['metrics']['corr']:.4f}"
                      + ("  <= retenu" if r is chosen else ""), r["shown"][r["final"]]) for r in results],
                    silhouette_array > 0, paths["qc_dir"] / "02_candidates.png",
                    f"Candidats : mêmes coupes, même fenêtre d'affichage (exécuté le {stamp})")

    # --- 4. Étages du candidat retenu ---
    print("4) Étages du candidat retenu (A départ, B rigide, C similarité, D affine complet)...")
    metrics_a, pet_a = evaluate(chosen["start"])
    stage_rows = [
        _row("A départ", chosen["start"], metrics_a),
        _row("B rigide (6 paramètres)", chosen["transforms"]["rigid"], chosen["metrics_by"]["rigid"],
             "<= retenu" if chosen["final"] == "rigid" else ""),
        _row("C similarité (7 paramètres)", chosen["transforms"]["similarity"], chosen["metrics_by"]["similarity"],
             "<= retenu" if chosen["final"] == "similarity" else ""),
        _row("D affine complet (12 paramètres)", chosen["transforms"]["affine"], chosen["metrics_by"]["affine"],
             "<= retenu" if chosen["final"] == "affine" else ""),
    ]
    _print_table(stage_rows)
    save_detailed_qc(fixed, moving, brain_mask, silhouette_array,
                     [("A - Départ", chosen["start"]), ("B - Rigide", chosen["transforms"]["rigid"]),
                      ("C - Similarité", chosen["transforms"]["similarity"]),
                      ("D - Affine complet", chosen["transforms"]["affine"])],
                     paths["qc_dir"] / "02_detailed_overlay.png")
    qc.save_gallery([("A départ", pet_a), ("B rigide", chosen["shown"]["rigid"]),
                     ("C similarité", chosen["shown"]["similarity"]), ("D affine complet", chosen["shown"]["affine"])],
                    silhouette_array > 0, paths["qc_dir"] / "02_stages.png",
                    f"Étages du candidat retenu : mêmes coupes, même fenêtre d'affichage (exécuté le {stamp})")
    save_pseudo_check(pseudo, chosen["shown"][chosen["final"]], silhouette_array,
                      paths["qc_dir"] / "02_pseudo_check.png",
                      f"Pseudo-PET et PET recalé (modèle {chosen['final']}, exécuté le {stamp})")
    del pet_a
    for r in results:
        r["shown"] = None

    metrics_file = paths["registration_metrics"]
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stage_rows[0].keys()))
        writer.writeheader()
        writer.writerows(candidate_rows + stage_rows)

    # --- 5. Transformation finale et garde-fous ---
    transform = chosen["transform"]
    info = describe_transform(transform)
    print(f"Modèle final : {chosen['final']} | déterminant {info['determinant']:.3f} (doit être positif) | "
          f"rotations {np.round(info['rotation_deg'], 1)}°")
    print(f"   Par axe du PET : échelles x/y/z = {np.round(info['axis_scales'], 3)} | cisaillements (degrés) "
          f"xy {info['shear_deg']['xy']:.1f}, xz {info['shear_deg']['xz']:.1f}, yz {info['shear_deg']['yz']:.1f}")
    problems, warnings = check_transform(info, settings)
    for w in warnings:
        print("[attention]", w)
    accepted = not problems
    prefix = "" if accepted else "02_REJETE_"

    # --- 6. Sauvegardes (si rejet : fichiers préfixés REJETE) ---
    out_image = paths["reference_in_mni"] if accepted else paths["interim_dir"] / (prefix + "reference_in_mni.nii.gz")
    out_image.parent.mkdir(parents=True, exist_ok=True)
    if accepted:
        paths["reference_transform"].parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(transform, str(paths["reference_transform"]))
    # Le PET d'ORIGINAL est interpolé UNE seule fois ici (linéaire), avec 0 hors du champ de vue.
    sitk.WriteImage(sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32), str(out_image))
    print(f"PET recalé : {out_image}")
    if accepted:
        mask_mni = sitk.Resample(brain_mask, fixed, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        sitk.WriteImage(mask_mni, str(paths["brain_mask_in_mni"]))
        print(f"Masque déplacé dans l'espace MNI : {paths['brain_mask_in_mni']}")

    # --- 7. Contrôle visuel final ---
    reg_img = io_utils.load_nifti(out_image)
    if not io_utils.same_grid(mni_img, reg_img):
        raise RuntimeError("Le PET recalé n'est pas sur la grille du MNI152 : anomalie.")
    figure = paths["qc_dir"] / ("02_reference_in_mni.png" if accepted else prefix + "reference_in_mni.png")
    qc.save_registration_check(io_utils.get_data_float32(mni_img), io_utils.get_data_float32(reg_img), figure,
                               f"Contrôle du recalage PET de référence -> MNI152 (exécuté le {stamp})"
                               + ("" if accepted else " (REJETÉ)"), silhouette_array > 0)
    print(f"Figures dans {paths['qc_dir']} : 02_pseudo_pet, 02_pseudo_check, 02_detailed_overlay, "
          f"02_candidates, 02_stages, 02_pose_scan")
    print(f"Tableau des mesures : {metrics_file} | durée {time.time() - started:.0f} s")

    if not accepted:
        raise RuntimeError("Recalage rejeté : " + " ; ".join(problems) +
                           ". Les fichiers 02_REJETE_* ont été écrits : regardez la figure avant toute décision.")
    return paths["reference_in_mni"]


if __name__ == "__main__":
    run(load_config())