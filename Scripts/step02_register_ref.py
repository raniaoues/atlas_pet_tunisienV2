"""Étape 2 (pipeline 1) : recaler le PET de référence sur l'IRM MNI152.

Problème résolu : le PET et l'IRM n'ont ni la même position, ni la même taille, ni la même grille.
On cherche la transformation qui superpose le PET à l'IRM, puis on place le PET sur la grille de l'IRM.

  image FIXE   = MNI152            (ne bouge jamais, sert de repère)
  image MOBILE = PET de référence  (c'est elle qu'on déplace)

POURQUOI CETTE STRATÉGIE
  Un TEP et une IRM ne se ressemblent pas : l'information mutuelle (IM) est le bon critère, mais sur nos
  données son paysage est plat (plusieurs inclinaisons de la tête donnent presque la même IM). L'IM seule ne peut
  donc pas décider. On procède du plus robuste au plus fin :

  1. DÉPART PAR LA FORME. On balaye les inclinaisons de la tête (de -60 à +60 degrés) et on mesure, pour chacune,
     le recouvrement (Dice) entre le masque cérébral du PET et la silhouette du cerveau du MNI152. Cela ne dépend
     pas des contrastes du PET. On garde plusieurs départs (candidats), pas un seul.
  2. RAFFINEMENT PAR IM de chaque candidat : rigide (6 paramètres), puis similarité (7 : + une échelle unique).
  3. COMPARAISON avec des mesures INDÉPENDANTES de l'optimiseur : Dice, part du cerveau couverte par le champ de
     vue, information mutuelle normalisée (NMI) calculée dans le cerveau. Le meilleur candidat est retenu ;
     une galerie montre tous les candidats avec les mêmes coupes (c'est elle qui tranche en cas de doute).
  4. Pour chaque candidat on calcule aussi l'AFFINE COMPLET (12 paramètres : échelles et cisaillements libres).
     `model: auto` ne le retient que s'il améliore NETTEMENT le Dice ET l'information mutuelle normalisée par
     rapport à la similarité, tout en restant plausible. Un cisaillement peut être réel : une inclinaison de
     gantry non corrigée à la conversion DICOM -> NIfTI déforme le volume en parallélogramme, et seule une
     transformation affine complète peut le corriger. `model: similarity` ou `affine` force le choix.

MASQUE FIXE : le MNI152 est une tête entière (crâne, scalp) alors que le PET montre surtout le cerveau. La
similarité n'est calculée que dans le cerveau du MNI152 (silhouette de la carte de régions, légèrement élargie).
Essayé et rejeté : restreindre aussi la similarité au masque du PET (cela dégrade le recalage sur nos données).

Les valeurs du PET ne sont jamais normalisées : elles sont seulement interpolées, UNE seule fois, à la fin.

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


def _array_to_image(array_xyz, like):
    """Tableau (x, y, z) de nibabel -> image SimpleITK ayant la géométrie de `like`."""
    image = sitk.GetImageFromArray(np.transpose(array_xyz, (2, 1, 0)).astype(np.uint8))
    image.CopyInformation(like)
    return image


def _image_to_array(image):
    """Image SimpleITK -> tableau (x, y, z) comme dans nibabel."""
    return np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))


def describe_geometry(name, image):
    """Affiche la géométrie d'un volume : taille, voxel, boîte et centre en millimètres (repère de SimpleITK)."""
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
# 1. Départ par la forme
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


def scan_pose(base, silhouette, moving_mask, settings):
    """Pour chaque inclinaison, meilleur Dice entre le masque du PET et la silhouette du MNI152 (image réduite 4 fois).

    Renvoie une liste de (inclinaison, Dice, décalage en mm).
    """
    coarse = sitk.Shrink(silhouette, [4, 4, 4])
    target = sitk.GetArrayFromImage(coarse) > 0
    low, high = settings["pose_pitch_range_deg"]
    reach = int(settings["pose_shift_range_mm"])
    curve = []
    for pitch in range(int(low), int(high) + 1, int(settings["pose_pitch_step_deg"])):
        best = (-1.0, (0.0, 0.0, 0.0))
        for dy, dz in itertools.product(range(-reach, reach + 1, 4), repeat=2):
            start = _start_transform(base, pitch, (0.0, dy, dz))
            moved = sitk.GetArrayFromImage(
                sitk.Resample(moving_mask, coarse, start, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)) > 0
            dice = 2.0 * (moved & target).sum() / max(moved.sum() + target.sum(), 1)
            if dice > best[0]:
                best = (float(dice), (0.0, float(dy), float(dz)))
        curve.append((pitch, best[0], best[1]))
    return curve


def pick_candidates(curve, count, min_separation):
    """Garde les `count` meilleures inclinaisons, espacées d'au moins `min_separation` degrés."""
    chosen = []
    for entry in sorted(curve, key=lambda c: -c[1]):
        if all(abs(entry[0] - other[0]) >= min_separation for other in chosen):
            chosen.append(entry)
        if len(chosen) == count:
            break
    return chosen


# ----------------------------------------------------------------------------------------------
# 2. Raffinement par information mutuelle
# ----------------------------------------------------------------------------------------------
def _run_stage(fixed, moving, transform, settings, fixed_mask):
    """Un étage de recalage par IM : modifie `transform` sur place."""
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=settings["mutual_information_bins"])
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
    """Départ -> rigide (6 paramètres) -> similarité (7 paramètres). `start` n'est pas modifié."""
    rigid = sitk.Euler3DTransform(start)
    _run_stage(fixed, moving, rigid, settings, fixed_mask)
    similarity = sitk.Similarity3DTransform()
    similarity.SetCenter(rigid.GetCenter())
    similarity.SetMatrix(rigid.GetMatrix())
    similarity.SetTranslation(rigid.GetTranslation())
    _run_stage(fixed, moving, similarity, settings, fixed_mask)
    return rigid, similarity


def to_affine(similarity, fixed, moving, settings, fixed_mask):
    """Étage D : affine complet (12 paramètres), qui part de la similarité."""
    affine = sitk.AffineTransform(3)
    affine.SetCenter(similarity.GetCenter())
    affine.SetMatrix(similarity.GetMatrix())
    affine.SetTranslation(similarity.GetTranslation())
    _run_stage(fixed, moving, affine, settings, fixed_mask)
    return affine


# ----------------------------------------------------------------------------------------------
# 3. Mesures indépendantes de l'optimiseur
# ----------------------------------------------------------------------------------------------
def _nmi(a, b, bins=32):
    """Information mutuelle normalisée (de 1 à 2, plus c'est haut mieux c'est)."""
    a = np.clip(a, *np.percentile(a, [0.5, 99.5]))
    b = np.clip(b, *np.percentile(b, [0.5, 99.5]))
    joint, _, _ = np.histogram2d(a, b, bins=bins)
    p = joint / joint.sum()

    def entropy(q):
        q = q[q > 0]
        return float(-(q * np.log(q)).sum())

    return (entropy(p.sum(axis=1)) + entropy(p.sum(axis=0))) / entropy(p)


def make_evaluator(fixed, moving, brain_mask, silhouette_array, wide_array):
    """Fabrique `evaluate(transform)` : Dice, couverture et NMI, calculés pareil pour tous les candidats."""
    mni = _image_to_array(fixed)
    silhouette = silhouette_array > 0
    wide = wide_array > 0

    def evaluate(transform):
        # Le PET est rééchantillonné avec -1 hors du champ de vue : on sait ainsi où il a des données.
        pet = _image_to_array(sitk.Resample(moving, fixed, transform, sitk.sitkLinear, -1.0, sitk.sitkFloat32))
        valid = pet >= 0
        moved = _image_to_array(sitk.Resample(brain_mask, fixed, transform, sitk.sitkNearestNeighbor,
                                              0, sitk.sitkUInt8)) > 0
        region = wide & valid
        metrics = {
            "dice": float(2.0 * (moved & silhouette).sum() / (moved.sum() + silhouette.sum())),
            "coverage": float((silhouette & valid).sum() / silhouette.sum()),
            "nmi": _nmi(mni[region], pet[region]) if region.sum() > 1000 else float("nan"),
        }
        return metrics, np.where(valid, pet, 0.0).astype(np.float32)

    return evaluate


def describe_transform(transform):
    """Résume la transformation PET -> IRM : déterminant, rotation, échelles et cisaillements.

    - `scales` : valeurs singulières (étirements principaux), utilisées par les garde-fous.
    - `axis_scales`, `shear_deg` : décomposition B = Q x K (Q rotation, K triangulaire) dans les axes du PET :
      échelle le long de x (gauche-droite), y (avant-arrière), z (haut-bas), et cisaillements entre axes.
      Le cisaillement y-z (l'axe haut-bas penché vers l'avant) est celui d'une inclinaison de gantry.
    """
    A = np.array(transform.GetMatrix()).reshape(3, 3)       # FIXE -> MOBILE
    det = float(np.linalg.det(A))
    B = np.linalg.inv(A)                                    # MOBILE -> FIXE : ce qu'on fait subir au PET
    U, S, Vt = np.linalg.svd(B)
    rotation = U @ Vt
    angles = Rotation.from_matrix(rotation if np.linalg.det(rotation) > 0 else -rotation).as_euler("xyz", degrees=True)

    Q, K = np.linalg.qr(B)
    signs = np.where(np.diag(K) < 0, -1.0, 1.0)
    Q, K = Q * signs, K * signs[:, None]
    shear = {"xy": np.degrees(np.arctan2(K[0, 1], K[1, 1])),
             "xz": np.degrees(np.arctan2(K[0, 2], K[2, 2])),
             "yz": np.degrees(np.arctan2(K[1, 2], K[2, 2]))}
    return {"determinant": det, "rotation_deg": angles, "scales": S,
            "axis_scales": np.diag(K), "shear_deg": shear}


def check_transform(info, settings):
    """Garde-fous. Renvoie (problèmes bloquants, avertissements).

    Bloquant : effet miroir, rotation énorme (tête retournée), échelle absurde.
    Avertissement : rotation importante mais plausible (tête penchée dans le TEP), échelles inégales.
    """
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
    if info["scales"].min() < low or info["scales"].max() > high:
        problems.append(f"échelle suspecte {np.round(info['scales'], 3)} (plage acceptée {low} à {high})")
    elif info["scales"].max() / info["scales"].min() > 1.15:
        warnings.append(f"échelles très inégales {np.round(info['scales'], 3)} : une tête réelle varie rarement "
                        "de plus de 10 % d'un axe à l'autre. Regardez la décomposition par axe (cisaillement ?)")
    if abs(info["shear_deg"]["yz"]) > 8:
        warnings.append(f"cisaillement y-z de {info['shear_deg']['yz']:.0f} degrés : typique d'une inclinaison de gantry "
                        "non corrigée à la conversion DICOM -> NIfTI. À vérifier dans l'en-tête DICOM "
                        "(balise 0018,1120 : GantryDetectorTilt)")
    return problems, warnings


def _row(label, transform, metrics, extra=""):
    info = describe_transform(transform)
    return {"etape": label, "rotation_x_deg": round(float(info["rotation_deg"][0]), 1),
            "rotation_y_deg": round(float(info["rotation_deg"][1]), 1),
            "rotation_z_deg": round(float(info["rotation_deg"][2]), 1),
            "echelle_x": round(float(info["axis_scales"][0]), 3), "echelle_y": round(float(info["axis_scales"][1]), 3),
            "echelle_z": round(float(info["axis_scales"][2]), 3),
            "cisaillement_yz_deg": round(float(info["shear_deg"]["yz"]), 1),
            "dice": round(metrics["dice"], 3), "couverture": round(metrics["coverage"], 3),
            "nmi": round(metrics["nmi"], 4), "remarque": extra}


def _print_table(rows):
    print(f"  {'étape':40s} {'rot.x':>6s} {'échelles x/y/z':>17s} {'cis.yz':>7s} {'Dice':>6s} {'couv.':>6s} {'NMI':>7s}")
    for r in rows:
        print(f"  {r['etape']:40s} {r['rotation_x_deg']:6.1f} {r['echelle_x']:5.2f}/{r['echelle_y']:4.2f}/{r['echelle_z']:4.2f} "
              f"{r['cisaillement_yz_deg']:6.1f}° {r['dice']:6.3f} {r['couverture']:6.2f} {r['nmi']:7.4f} {r['remarque']}")


def decide_model(settings, similarity, affine, m_sim, m_aff):
    """Renvoie 'similarity' ou 'affine' (et la raison), selon `model`."""
    model = settings["model"]
    if model in ("similarity", "affine"):
        return model, f"imposé par model: {model}"
    problems, _ = check_transform(describe_transform(affine), settings)
    gain_dice, gain_nmi = m_aff["dice"] - m_sim["dice"], m_aff["nmi"] - m_sim["nmi"]
    if problems:
        return "similarity", f"affine non plausible ({problems[0]})"
    if gain_dice >= settings["auto_min_dice_gain"] and gain_nmi >= settings["auto_min_nmi_gain"]:
        return "affine", f"l'affine gagne {gain_dice:+.3f} de Dice et {gain_nmi:+.4f} de NMI"
    return "similarity", f"gain de l'affine insuffisant ({gain_dice:+.3f} de Dice, {gain_nmi:+.4f} de NMI)"


# ----------------------------------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------------------------------
def run(cfg):
    started = time.time()
    paths, settings = cfg["paths"], cfg["registration"]

    # --- Entrées ---
    for key, hint in (("brain_mask", "Lancez d'abord step01_mask.py."),
                      ("region_map", "Elle sert à délimiter le cerveau du MNI152 (voir paths.region_map).")):
        if not paths[key].exists():
            raise FileNotFoundError(f"Fichier introuvable : {paths[key]}. {hint}")
    fixed = _read_float32(paths["mni_template"])
    moving = _read_float32(paths["reference_pet"])
    mni_img = io_utils.load_nifti(paths["mni_template"])
    brain_mask = sitk.Cast(sitk.ReadImage(str(paths["brain_mask"])), sitk.sitkUInt8)

    print("Géométrie des volumes (millimètres) :")
    describe_geometry("MNI152 (fixe)  ", fixed)
    pet_box = describe_geometry("PET (mobile)   ", moving)
    if pet_box.min() < 150:
        print(f"[attention] une dimension du PET ne fait que {pet_box.min():.0f} mm : le champ de vue peut couper le cerveau.")

    # --- Masques du MNI152 : silhouette du cerveau et version élargie (pour la similarité) ---
    silhouette_array, wide = make_mni_masks(io_utils.load_nifti(paths["region_map"]), mni_img,
                                            settings["fixed_mask_dilation_mm"])
    io_utils.save_nifti(wide, mni_img, paths["mni_brain_mask"])
    fixed_mask = _array_to_image(wide, fixed)
    silhouette = _array_to_image(silhouette_array, fixed)
    evaluate = make_evaluator(fixed, moving, brain_mask, silhouette_array, wide)

    # --- 1. Départ par la forme ---
    print("1) Balayage des inclinaisons de départ (recouvrement des masques)...")
    base = _initial_rigid(fixed_mask, brain_mask)
    curve = scan_pose(base, silhouette, brain_mask, settings)
    candidates = pick_candidates(curve, settings["n_candidates"], settings["candidate_min_separation_deg"])
    qc.save_pose_curve([c[0] for c in curve], [c[1] for c in curve], [c[0] for c in candidates],
                       paths["qc_dir"] / "02_pose_scan.png")
    spread = max(c[1] for c in curve) - min(c[1] for c in curve)
    print(f"   Dice de {min(c[1] for c in curve):.3f} à {max(c[1] for c in curve):.3f} "
          f"(écart {spread:.3f}{' : courbe plate, le départ par la forme est peu discriminant' if spread < 0.05 else ''})")

    # --- 2. Raffinement de chaque candidat : rigide, similarité, affine complet ---
    print(f"2) Raffinement par information mutuelle de {len(candidates)} candidats (modèle : {settings['model']})...")
    results = []
    for k, (pitch, dice0, shift) in enumerate(candidates, start=1):
        start = _start_transform(base, pitch, shift)
        rigid, similarity = refine(fixed, moving, start, settings, fixed_mask)
        affine = to_affine(similarity, fixed, moving, settings, fixed_mask)
        m_sim, pet_sim = evaluate(similarity)
        m_aff, pet_aff = evaluate(affine)
        final, reason = decide_model(settings, similarity, affine, m_sim, m_aff)
        transform, metrics, pet = (affine, m_aff, pet_aff) if final == "affine" else (similarity, m_sim, pet_sim)
        info = describe_transform(transform)
        problems, _ = check_transform(info, settings)
        results.append({"k": k, "pitch": pitch, "start": start, "rigid": rigid, "similarity": similarity,
                        "affine": affine, "m_sim": m_sim, "m_aff": m_aff, "pet_sim": pet_sim, "pet_aff": pet_aff,
                        "final": final, "reason": reason, "transform": transform, "metrics": metrics, "pet": pet,
                        "problems": problems})
        print(f"   candidat {k} (départ {pitch:+d}°) -> {final} ({reason}) : rotation {np.round(info['rotation_deg'], 1)}°, "
              f"Dice {metrics['dice']:.3f}, NMI {metrics['nmi']:.4f}" + (f"  REFUSÉ : {problems[0]}" if problems else ""))

    # --- 3. Choix du candidat ---
    # Règle : meilleur Dice (la forme) ; entre candidats à égalité (0,005 près), meilleure information mutuelle normalisée.
    pool = [r for r in results if not r["problems"]] or results
    override = settings.get("candidate_override")
    if override:
        chosen = results[int(override) - 1]
        print(f"3) Candidat imposé par candidate_override : {chosen['k']}")
    else:
        top = max(r["metrics"]["dice"] for r in pool)
        ties = [r for r in pool if r["metrics"]["dice"] >= top - 0.005]
        chosen = max(ties, key=lambda r: r["metrics"]["nmi"])
        print(f"3) Candidat retenu : {chosen['k']} (meilleur Dice, départage par l'information mutuelle normalisée)")
        best_nmi = max(pool, key=lambda r: r["metrics"]["nmi"])
        if best_nmi is not chosen and best_nmi["metrics"]["nmi"] - chosen["metrics"]["nmi"] > 0.003:
            print(f"[attention] la forme préfère nettement le candidat {chosen['k']}, l'information mutuelle le "
                  f"candidat {best_nmi['k']} : regardez 02_candidates.png avant de décider (candidate_override).")

    candidate_rows = [_row(f"candidat {r['k']} (départ {r['pitch']:+d}°), {r['final']}", r["transform"], r["metrics"],
                           "<= retenu" if r is chosen else "") for r in results]
    qc.save_gallery([(f"candidat {r['k']} : départ {r['pitch']:+d}°, {r['final']}, Dice {r['metrics']['dice']:.3f}, "
                      f"NMI {r['metrics']['nmi']:.4f}" + ("  <= retenu" if r is chosen else ""), r["pet"])
                     for r in results], silhouette_array > 0, paths["qc_dir"] / "02_candidates.png",
                    "Candidats : mêmes coupes, même fenêtre d'affichage")

    # --- 4. Comparaison des étages pour le candidat retenu ---
    print("4) Étages du candidat retenu (A départ, B rigide, C similarité, D affine complet)...")
    metrics_a, pet_a = evaluate(chosen["start"])
    metrics_b, pet_b = evaluate(chosen["rigid"])
    stage_rows = [_row("A départ par la forme", chosen["start"], metrics_a),
                  _row("B rigide (6 paramètres)", chosen["rigid"], metrics_b),
                  _row("C similarité (7 paramètres)", chosen["similarity"], chosen["m_sim"],
                       "<= retenu" if chosen["final"] == "similarity" else ""),
                  _row("D affine complet (12 paramètres)", chosen["affine"], chosen["m_aff"],
                       "<= retenu" if chosen["final"] == "affine" else "")]
    _print_table(stage_rows)
    qc.save_gallery([("A départ par la forme", pet_a), ("B rigide", pet_b),
                     ("C similarité", chosen["pet_sim"]), ("D affine complet", chosen["pet_aff"])],
                    silhouette_array > 0, paths["qc_dir"] / "02_stages.png",
                    "Étages du candidat retenu : mêmes coupes, même fenêtre d'affichage")
    del pet_a, pet_b
    for r in results:
        r["pet"] = r["pet_sim"] = r["pet_aff"] = None
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

    # --- 6. Sauvegardes (si rejet : fichiers préfixés REJETE, pour pouvoir regarder ce qui s'est passé) ---
    out_image = paths["reference_in_mni"] if accepted else paths["interim_dir"] / (prefix + "reference_in_mni.nii.gz")
    out_image.parent.mkdir(parents=True, exist_ok=True)
    if accepted:
        paths["reference_transform"].parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(transform, str(paths["reference_transform"]))
    # Le PET d'origine est interpolé UNE seule fois ici (linéaire), avec 0 hors du champ de vue.
    sitk.WriteImage(sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32), str(out_image))
    print(f"PET recalé : {out_image}")
    if accepted:
        mask_mni = sitk.Resample(brain_mask, fixed, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        sitk.WriteImage(mask_mni, str(paths["brain_mask_in_mni"]))
        print(f"Masque déplacé dans l'espace MNI : {paths['brain_mask_in_mni']}")

    # --- 7. Contrôle visuel (relecture avec nibabel, comme les autres étapes) ---
    reg_img = io_utils.load_nifti(out_image)
    if not io_utils.same_grid(mni_img, reg_img):
        raise RuntimeError("Le PET recalé n'est pas sur la grille du MNI152 : anomalie.")
    figure = paths["qc_dir"] / ("02_reference_in_mni.png" if accepted else prefix + "reference_in_mni.png")
    qc.save_registration_check(io_utils.get_data_float32(mni_img), io_utils.get_data_float32(reg_img), figure,
                               "Contrôle du recalage PET de référence -> MNI152" + ("" if accepted else " (REJETÉ)"),
                               silhouette_array > 0)
    print(f"Figures : {figure} | {paths['qc_dir'] / '02_candidates.png'} | {paths['qc_dir'] / '02_stages.png'} | "
          f"{paths['qc_dir'] / '02_pose_scan.png'}")
    print(f"Tableau des mesures : {metrics_file} | durée {time.time() - started:.0f} s")

    if not accepted:
        raise RuntimeError("Recalage rejeté : " + " ; ".join(problems) +
                           ". Les fichiers 02_REJETE_* ont été écrits : regardez la figure avant toute décision.")
    return paths["reference_in_mni"]


if __name__ == "__main__":
    run(load_config())