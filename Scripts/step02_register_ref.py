"""step02_register_ref.py : recalage AFFINE 3D par descente de gradient PyTorch (adaptation 3D de HomographyNet 2D),
piloté par config.yaml. Aucun recalage non linéaire, aucune dépendance à ANTs : transformation globale seulement
(rigide -> similarité -> affine), exportée en .tfm SimpleITK.

ACQUIS DE LA VERSION PRÉCÉDENTE
  * PET masqué (dilaté de `moving_dilate_vox` voxels) pour le critère et les mesures ; l'export final reste le PET
    d'origine ;
  * recalage par étapes rigide -> similarité -> affine (affine au niveau le plus fin seulement) ;
  * critères lisses : `soft_mi`, `soft_nmi`, `ncc`, `mine` ;
  * enchaînement d'essais (`init_from`), plage d'inclinaisons qui s'élargit si le meilleur départ est en bordure.

NOUVEAUTÉS DE CETTE VERSION
  1. PAS DÉCROISSANT dans chaque niveau (cosinus, de 100 % à 10 %) : moins de bruit en fin de niveau avec
     l'échantillonnage stochastique.
  2. RÉGULARISATION branchée (`regularization`) : pénalise l'étirement / le cisaillement des étapes similarité et
     affine.
  3. GARDE-FOU par étape : le critère est réévalué sur un échantillon FIXE (150 000 voxels) avant / après chaque
     étape ; si l'étape a dégradé le critère, on garde le résultat précédent.
  4. CHOIX DU DÉPART en deux temps : tous les départs sont évalués grossièrement, puis les 3 meilleurs sont affinés
     au niveau suivant avant de décider (les scores des départs étaient presque identiques à l'échelle grossière).
  5. Plus d'échantillons au niveau le plus fin (`samples_fine`, 2 x `samples` par défaut).
  6. MESURES INDÉPENDANTES du critère : Dice et couverture entre le masque cérébral du PET déplacé et le masque MNI,
     en plus de la corrélation et de `nmi_avec_t1`. Les mêmes mesures sont calculées au DÉPART (avant recalage) :
     c'est la référence pour juger si un recalage apporte quelque chose. Le seuil absolu `min_nmi` est
     peu parlant (une NMI PET/IRM correcte vaut souvent 1.03 - 1.06) : comparez au départ.
  7. CONTRÔLE DE PLAUSIBILITÉ de la transformation (anisotropie, échelle) : alerte, et écarte l'essai du choix final.
  8. SÉLECTION : tous les essais sont lancés (sauf `stop_on_success: true`) et le meilleur est retenu selon
     `select_metric` (nmi_avec_t1, dice ou correlation).
  9. Surcharges PAR ESSAI : model, iterations, samples, regularization, starts, levels (nombre de niveaux fins).
     Un essai qui part d'une transformation existante (`init_from` ou --init) n'utilise par défaut que les 2 niveaux
     les plus fins.

CE QUI CHANGE EN 3D PAR RAPPORT AU SCRIPT 2D
  * Coordonnées en millimètres (affines NIfTI), pas de grille normalisée commune.
  * Aucun miroir possible : exp(G) a un déterminant > 0.
  * Échantillonnage stochastique de voxels dans le masque du cerveau MNI.
  * Point de départ indispensable : centres des cerveaux superposés + inclinaisons testées.

UTILISATION (depuis le dossier qui contient config.yaml)
  python Scripts\\step02_register_ref.py --selftest
  python Scripts\\step02_register_ref.py
  python Scripts\\step02_register_ref.py --attempt mi_t1
  python Scripts\\step02_register_ref.py --attempt mi_refine --init transforms\\02_dl_mi_t1.tfm

JUGEZ le résultat avec les figures et avec des mesures indépendantes du critère optimisé.
"""
import argparse
import shutil
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy import ndimage

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RAS_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])       # NIfTI/nibabel travaille en RAS, SimpleITK en LPS


# ----------------------------------------------------------------------------------------------
# A. Lecture et conventions de coordonnées
# ----------------------------------------------------------------------------------------------
def load_volume(path):
    """Renvoie (tableau (x, y, z) en float32, affine 4x4 voxel -> mm RAS)."""
    img = nib.load(str(path))
    if len(img.shape) != 3:
        raise ValueError(f"{path} : volume 3D attendu, forme trouvée {img.shape}")
    return np.asarray(img.dataobj, dtype=np.float32), np.array(img.affine, dtype=np.float64)


def itk_to_ras(transform):
    """Transformation SimpleITK (LPS, fixe -> mobile) -> matrice 4x4 en coordonnées RAS."""
    origin = np.array(transform.TransformPoint((0.0, 0.0, 0.0)))
    columns = [np.array(transform.TransformPoint(tuple(e))) - origin for e in np.eye(3)]
    lps = np.eye(4)
    lps[:3, :3] = np.stack(columns, axis=1)
    lps[:3, 3] = origin
    return RAS_LPS @ lps @ RAS_LPS


def ras_to_itk(matrix_ras):
    """Matrice 4x4 RAS (fixe -> mobile) -> transformation affine SimpleITK (LPS)."""
    import SimpleITK as sitk
    lps = RAS_LPS @ matrix_ras @ RAS_LPS
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(tuple(lps[:3, :3].ravel()))
    transform.SetTranslation(tuple(lps[:3, 3]))
    return transform


def center_of_mass_ras(mask, affine):
    """Centre de gravité d'un masque, en mm RAS."""
    return (affine @ np.append(np.argwhere(mask).mean(axis=0), 1.0))[:3]


def initial_matrix(A_f, fixed_mask, A_m, moving_mask, pitch_deg=0.0):
    """Départ (fixe -> mobile, RAS) : centres des deux cerveaux superposés, après une inclinaison `pitch_deg`
    autour de l'axe gauche-droite. Renvoie (matrice 4x4, centre du cerveau fixe en mm)."""
    c_f = center_of_mass_ras(fixed_mask, A_f)
    c_m = center_of_mass_ras(moving_mask, A_m)
    a = np.deg2rad(pitch_deg)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c_m - R @ c_f
    return T, c_f


def mask_moving(moving, moving_mask, dilate_vox=3):
    """PET destiné au CRITÈRE : les voxels hors du masque cérébral (dilaté de `dilate_vox` voxels) sont mis à zéro.
    Le PET d'origine n'est pas modifié (il reste utilisé pour l'export final)."""
    keep = ndimage.binary_dilation(moving_mask, iterations=int(dilate_vox)) if dilate_vox > 0 else moving_mask
    return np.where(keep, moving, 0.0).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# B. Modèle de transformation et critères
# ----------------------------------------------------------------------------------------------
class Affine3D(nn.Module):
    """Transformation affine 3D : p_mobile = L (p_fixe - c) + m0 + t, avec L = L0 . exp(G).

    G (3x3) et t (3) valent 0 au départ : on part exactement du point de départ `init_ras`.
    exp(G) a un déterminant exp(trace G) > 0 : AUCUN miroir possible.
    model : rigid (G antisymétrique : rotation), similarity (+ une échelle), affine (G libre : 12 paramètres).
    """

    def __init__(self, init_ras, center_ras, model="affine"):
        super().__init__()
        kw = dict(dtype=torch.float64)
        L0 = torch.tensor(init_ras[:3, :3], **kw)
        b0 = torch.tensor(init_ras[:3, 3], **kw)
        c = torch.tensor(center_ras, **kw)
        self.register_buffer("L0", L0)
        self.register_buffer("c", c)
        self.register_buffer("m0", L0 @ c + b0)          # image du centre par le point de départ
        self.G = nn.Parameter(torch.zeros(3, 3, **kw))
        self.t = nn.Parameter(torch.zeros(3, **kw))
        self.model = model

    def generator(self):
        if self.model == "affine":
            return self.G
        skew = 0.5 * (self.G - self.G.T)
        if self.model == "rigid":
            return skew
        if self.model == "similarity":
            return skew + (torch.trace(self.G) / 3.0) * torch.eye(3, dtype=self.G.dtype, device=self.G.device)
        raise ValueError(f"model inconnu : {self.model}")

    def matrix(self):
        L = self.L0 @ torch.linalg.matrix_exp(self.generator())
        offset = self.m0 + self.t - L @ self.c
        top = torch.cat([L, offset[:, None]], dim=1)
        bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=top.dtype, device=top.device)
        return torch.cat([top, bottom], dim=0)

    def regularizer(self):
        """Pénalise l'étirement / le cisaillement (partie symétrique de G) par rapport au début de l'étape : la
        transformation ne s'écarte du résultat précédent que si le critère y gagne vraiment."""
        G = self.generator()
        sym = 0.5 * (G + G.T)
        return (sym ** 2).sum()


def _unit(x):
    """Ramène des valeurs dans [0, 1] (bornes = percentiles 1 et 99, sans gradient). Sert au calcul du critère
    seulement : les données ne sont pas modifiées."""
    lo, hi = torch.quantile(x.detach(), torch.tensor([0.01, 0.99], dtype=x.dtype, device=x.device))
    return ((x - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)


def loss_ncc(a, b):
    """-corrélation (cible et PET de même contraste, ex. pseudo-PET). Ignore l'échelle globale des valeurs."""
    a, b = a - a.mean(), b - b.mean()
    return -(a * b).sum() / (torch.sqrt((a * a).sum() * (b * b).sum()) + 1e-8)


def _soft_joint(a, b, bins=48, sigma_bins=1.5):
    """Histogramme joint lisse (noyaux gaussiens, largeur sigma_bins classes) : renvoie (pab, pa, pb)."""
    a, b = _unit(a), _unit(b)
    centers = torch.linspace(0.0, 1.0, bins, dtype=a.dtype, device=a.device)
    sigma = sigma_bins / bins

    def memberships(x):
        w = torch.exp(-((x[:, None] - centers[None, :]) ** 2) / (2 * sigma ** 2))
        return w / (w.sum(dim=1, keepdim=True) + 1e-8)

    wa, wb = memberships(a), memberships(b)
    pab = wa.T @ wb / a.shape[0]
    return pab, pab.sum(dim=1, keepdim=True), pab.sum(dim=0, keepdim=True)


def loss_soft_mi(a, b):
    """-information mutuelle (différentiable ; multimodal : PET contre T1)."""
    pab, pa, pb = _soft_joint(a, b)
    return -(pab * torch.log((pab + 1e-10) / (pa * pb + 1e-10))).sum()


def loss_soft_nmi(a, b):
    """-information mutuelle normalisée (Ha + Hb) / Hab : plus robuste aux changements de recouvrement."""
    pab, pa, pb = _soft_joint(a, b)
    ha = -(pa * torch.log(pa + 1e-10)).sum()
    hb = -(pb * torch.log(pb + 1e-10)).sum()
    hab = -(pab * torch.log(pab + 1e-10)).sum()
    return -(ha + hb) / (hab + 1e-10)


class MINE(nn.Module):
    """Estimateur neuronal d'information mutuelle (borne inférieure de Donsker-Varadhan), version simplifiée."""

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, a, b):
        joint = self.net(torch.stack([a, b], dim=1)).squeeze(1)
        shuffled = b[torch.randperm(b.shape[0], device=b.device)]
        marginal = self.net(torch.stack([a, shuffled], dim=1)).squeeze(1)
        return joint.mean() - (torch.logsumexp(marginal, dim=0) - float(np.log(b.shape[0])))


def make_loss(name):
    """Renvoie (fonction de perte, module supplémentaire à entraîner ou None)."""
    if name == "ncc":
        return loss_ncc, None
    if name == "soft_mi":
        return loss_soft_mi, None
    if name == "soft_nmi":
        return loss_soft_nmi, None
    if name == "mine":
        net = MINE().to(DEVICE)
        return (lambda a, b: -net(_unit(a), _unit(b))), net
    raise ValueError(f"critère inconnu : {name} (ncc, soft_mi, soft_nmi ou mine)")


# ----------------------------------------------------------------------------------------------
# C. Pyramide, échantillonnage, optimisation
# ----------------------------------------------------------------------------------------------
_LEVEL_CACHE = {}       # évite de refiltrer les volumes à chaque étape / chaque départ testé


def build_levels(fixed, A_f, moving, A_m, fixed_mask, factors, sigmas_mm):
    """Pyramide gaussienne : à chaque niveau, les deux images sont floutées (sigma en mm) ; l'image fixe est de plus
    sous-échantillonnée d'un facteur `factor`. Le mobile garde sa résolution : on l'échantillonne par grid_sample.
    Les niveaux sont mis en cache (clé : volumes, facteur, sigma)."""
    vox_f = np.sqrt((A_f[:3, :3] ** 2).sum(axis=0))
    vox_m = np.sqrt((A_m[:3, :3] ** 2).sum(axis=0))
    levels = []
    for factor, sigma in zip(factors, sigmas_mm):
        key = (id(fixed), id(moving), id(fixed_mask), int(factor), float(sigma))
        if key not in _LEVEL_CACHE:
            smooth_f = ndimage.gaussian_filter(fixed, sigma / vox_f) if sigma > 0 else fixed
            smooth_m = ndimage.gaussian_filter(moving, sigma / vox_m) if sigma > 0 else moving
            sub = (slice(None, None, int(factor)),) * 3
            mask_l = fixed_mask[sub]
            coords = np.argwhere(mask_l)
            level = {
                "factor": int(factor),
                "ijk": torch.tensor(coords * int(factor), dtype=torch.float64, device=DEVICE),
                "values": torch.tensor(smooth_f[sub][mask_l], dtype=torch.float32, device=DEVICE),
                "moving": torch.tensor(smooth_m, dtype=torch.float32, device=DEVICE)[None, None],
            }
            # les tableaux sont conservés avec le niveau : leur id() reste valide tant que le cache existe
            _LEVEL_CACHE[key] = (fixed, moving, fixed_mask, level)
        levels.append(_LEVEL_CACHE[key][3])
    return levels


def sample_moving(moving, T, ijk_f, A_f, A_m_inv, size_m):
    """Valeurs du volume mobile aux positions des voxels fixes `ijk_f`, après la transformation T (fixe -> mobile).

    Chaîne : indices fixes -> mm (A_f) -> mm mobile (T) -> indices mobiles (A_m^-1) -> coordonnées de grid_sample.
    Renvoie (valeurs, validité) : valide = le point tombe dans le champ de vue du mobile.
    """
    n = ijk_f.shape[0]
    homogeneous = torch.cat([ijk_f, torch.ones(n, 1, dtype=ijk_f.dtype, device=ijk_f.device)], dim=1)
    ijk_m = homogeneous @ (A_m_inv @ T @ A_f).T
    scale = torch.tensor([size_m[0] - 1, size_m[1] - 1, size_m[2] - 1], dtype=ijk_m.dtype, device=ijk_m.device)
    normalized = 2.0 * ijk_m[:, :3] / scale - 1.0           # ordre (i, j, k) = (x, y, z)
    grid = normalized[:, [2, 1, 0]]                          # grid_sample veut (W, H, D) = (k, j, i)
    valid = (grid.abs() <= 1.0).all(dim=1)
    values = F.grid_sample(moving, grid.float().view(1, n, 1, 1, 3), mode="bilinear",
                           padding_mode="zeros", align_corners=True).view(n)
    return values, valid


def make_eval_set(level, n_eval=150000, seed=123):
    """Indices FIXES (toujours les mêmes) servant à comparer deux transformations sans bruit d'échantillonnage."""
    n = level["ijk"].shape[0]
    if n <= n_eval:
        return torch.arange(n, device=DEVICE)
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=g)[:n_eval].to(DEVICE)


@torch.no_grad()
def eval_loss(level, T, loss_fn, idx, A_f_t, A_m_inv, size_m):
    """Valeur DÉTERMINISTE du critère pour la transformation T (tenseur 4x4) sur l'échantillon fixe `idx`."""
    values, valid = sample_moving(level["moving"], T, level["ijk"][idx], A_f_t, A_m_inv, size_m)
    if valid.float().mean() < 0.2 or int(valid.sum()) < 500:
        return float("inf")
    return float(loss_fn(level["values"][idx][valid], values[valid]))


def register(fixed, A_f, moving, A_m, fixed_mask, init_ras, center, model="affine", loss_name="ncc",
             factors=(4, 2, 1), sigmas_mm=(4.0, 2.0, 1.0), iterations=(150, 150, 150),
             lr_translation=1.0, lr_linear=5e-3, n_samples=40000, n_samples_fine=None, reg_weight=0.0,
             seed=0, verbose=True, coarsest=None):
    """Recalage par niveaux de la pyramide, avec UN modèle.
    Renvoie (matrice 4x4 RAS fixe -> mobile, historique du critère, score déterministe au dernier niveau).

    `coarsest` : facteur du niveau le plus grossier de TOUTE la pyramide (sert à calculer les pas ; à fournir quand
    on n'exécute qu'une partie des niveaux, sinon les pas seraient trop grands).
    `n_samples_fine` : nombre d'échantillons au niveau de facteur 1 (par défaut `n_samples`).
    Garde-fou : si le critère déterministe du résultat est pire que celui du départ, le départ est renvoyé."""
    torch.manual_seed(seed)
    levels = build_levels(fixed, A_f, moving, A_m, fixed_mask, factors, sigmas_mm)
    A_f_t = torch.tensor(A_f, dtype=torch.float64, device=DEVICE)
    A_m_inv = torch.tensor(np.linalg.inv(A_m), dtype=torch.float64, device=DEVICE)
    net = Affine3D(init_ras, center, model).to(DEVICE)
    loss_fn, extra = make_loss(loss_name)
    coarsest = coarsest or max(factors)

    last = levels[-1]
    eval_idx = make_eval_set(last)
    T_start = net.matrix().detach()
    score_start = eval_loss(last, T_start, loss_fn, eval_idx, A_f_t, A_m_inv, moving.shape) if extra is None else None

    history = []
    for level, n_iter in zip(levels, iterations):
        scale = level["factor"] / coarsest                  # pas plus petits aux niveaux fins
        groups = [{"params": [net.t], "lr": lr_translation * scale}, {"params": [net.G], "lr": lr_linear * scale}]
        if extra is not None:
            groups.append({"params": extra.parameters(), "lr": 1e-3})
        optimizer = torch.optim.Adam(groups)
        # pas décroissant dans le niveau : 100 % -> 10 % (cosinus)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda i, n=n_iter: 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * min(i, n) / max(n, 1))))
        n_level = level["ijk"].shape[0]
        ns = n_samples_fine if (n_samples_fine and level["factor"] == 1) else n_samples
        skipped = 0
        start_len = len(history)
        for _ in range(n_iter):
            pick = torch.randint(0, n_level, (min(ns, n_level),), device=DEVICE)
            values, valid = sample_moving(level["moving"], net.matrix(), level["ijk"][pick], A_f_t, A_m_inv,
                                          moving.shape)
            if valid.float().mean() < 0.2 or int(valid.sum()) < 500:     # presque rien à comparer
                skipped += 1
                continue
            data_loss = loss_fn(level["values"][pick][valid], values[valid])
            loss = data_loss
            if reg_weight > 0 and model != "rigid":
                loss = loss + reg_weight * net.regularizer()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            optimizer.step()
            scheduler.step()
            history.append(data_loss.item())                # l'historique ne contient que le critère, sans pénalité
        if verbose:
            recent = history[start_len:][-10:]
            tail = np.mean(recent) if recent else float("nan")
            print(f"   [{model}] niveau x{level['factor']} : critère {tail:+.4f}" + (
                f" ({skipped} itérations ignorées : trop peu de recouvrement, vérifiez le départ)" if skipped else ""))

    T_end = net.matrix().detach()
    if extra is None:
        score_end = eval_loss(last, T_end, loss_fn, eval_idx, A_f_t, A_m_inv, moving.shape)
        if score_end > score_start:
            if verbose:
                print(f"   [{model}] étape rejetée : critère de contrôle {score_end:+.4f} pire que le départ "
                      f"{score_start:+.4f} ; résultat précédent conservé.")
            return T_start.cpu().numpy(), history, score_start
        if verbose:
            print(f"   [{model}] critère de contrôle (échantillon fixe) : {score_start:+.4f} -> {score_end:+.4f}")
        return T_end.cpu().numpy(), history, score_end
    score = float(np.mean(history[-10:])) if history else float("inf")
    return T_end.cpu().numpy(), history, score


def register_staged(fixed, A_f, moving, A_m, fixed_mask, init_ras, center, model="affine", loss_name="ncc",
                    factors=(4, 2, 1), sigmas_mm=(4.0, 2.0, 1.0), iterations=(150, 150, 150), coarsest=None, **kw):
    """Recalage par ÉTAPES : rigide (niveaux grossiers) -> similarité (niveaux intermédiaires et fin) -> affine
    (niveau le plus fin seulement). Chaque étape part du résultat de la précédente.
    model='rigid' ou 'similarity' arrête la chaîne à l'étape correspondante.
    Renvoie (matrice, historique, score déterministe de la dernière étape)."""
    levels = list(zip(factors, sigmas_mm, iterations))
    n = len(levels)
    if model == "rigid":
        plan = [("rigid", levels)]
    else:
        plan = [("rigid", levels[:max(n - 1, 1)]), ("similarity", levels[max(n - 2, 0):])]
        if model == "affine":
            plan.append(("affine", levels[-1:]))
    coarsest = coarsest or max(factors)
    current, history, score = init_ras, [], None
    for stage_model, stage_levels in plan:
        f, s, it = zip(*stage_levels)
        current, h, score = register(fixed, A_f, moving, A_m, fixed_mask, current, center, stage_model, loss_name,
                                     f, s, it, coarsest=coarsest, **kw)
        history += h
    return current, history, score


def pick_start(fixed, A_f, moving, A_m, fixed_mask, moving_mask, starts_deg, loss_name, factors, sigmas,
               n_samples, seed, max_extend=3, shortlist=3, coarse_iters=60, refine_iters=100):
    """Choisit l'inclinaison de départ en deux temps (évaluation RIGIDE) :
      1. tous les départs au niveau le plus grossier (la plage s'étend automatiquement si le meilleur est en bordure,
         jusqu'à `max_extend` fois, même pas) ;
      2. les `shortlist` meilleurs sont affinés au niveau suivant ; le critère déterministe décide.
    Renvoie (matrice de départ, centre)."""
    scored = {}
    coarsest = max(factors)

    def evaluate(pitch):
        init0, centre0 = initial_matrix(A_f, fixed_mask, A_m, moving_mask, pitch)
        T_q, _, score = register(fixed, A_f, moving, A_m, fixed_mask, init0, centre0, "rigid", loss_name,
                                 factors[:1], sigmas[:1], (coarse_iters,), n_samples=n_samples, seed=seed,
                                 verbose=False, coarsest=coarsest)
        scored[pitch] = (score, T_q, centre0)
        print(f"   départ {pitch:+.0f}° : critère {score:+.4f}")

    pitches = sorted(float(p) for p in starts_deg)
    for p in pitches:
        evaluate(p)
    step = abs(pitches[1] - pitches[0]) if len(pitches) > 1 else 15.0
    for _ in range(max_extend):
        ordered = sorted(scored)
        best = min(ordered, key=lambda p: scored[p][0])
        if best == ordered[0]:
            new = best - step
        elif best == ordered[-1]:
            new = best + step
        else:
            break
        if abs(new) > 180 or new in scored:
            break
        print(f"   le meilleur départ ({best:+.0f}°) est en bordure : plage étendue à {new:+.0f}°")
        evaluate(new)

    ordered = sorted(scored, key=lambda p: scored[p][0])[:max(shortlist, 1)]
    if len(factors) > 1 and len(ordered) > 1:
        print(f"   affinage des {len(ordered)} meilleurs départs au niveau x{factors[1]}...")
        finals = {}
        for p in ordered:
            _, T_q, centre0 = scored[p]
            T_r, _, score = register(fixed, A_f, moving, A_m, fixed_mask, T_q, centre0, "rigid", loss_name,
                                     factors[1:2], sigmas[1:2], (refine_iters,), n_samples=n_samples, seed=seed,
                                     verbose=False, coarsest=coarsest)
            finals[p] = (score, T_r, centre0)
            print(f"   départ {p:+.0f}° affiné : critère {score:+.4f}")
    else:
        finals = {p: scored[p] for p in ordered}
    best = min(finals, key=lambda p: finals[p][0])
    all_p = sorted(scored)
    if best in (all_p[0], all_p[-1]) and len(all_p) > 1:
        print("   ATTENTION : le meilleur départ reste en bordure de la plage testée ; vérifiez la figure.")
    print(f"   départ retenu : {best:+.0f}°")
    return finals[best][1], finals[best][2]


@torch.no_grad()
def report(T_ras, fixed, A_f, moving, A_m, fixed_mask, t1=None, moving_mask=None, n=300000):
    """Mesures de contrôle DANS LE MASQUE CÉRÉBRAL MNI : corrélation avec la cible, couverture et, si `t1` est
    fourni, information mutuelle normalisée avec la vraie IRM. Si `moving_mask` est fourni : Dice et couverture entre
    le masque cérébral du PET déplacé et le masque MNI (indépendants de tout critère d'intensité ; le masque MNI
    peut être tronqué à la base, donc ne comparez ces valeurs qu'entre elles).
    `moving` doit être le PET masqué (même volume que pour l'optimisation)."""
    A_f_t = torch.tensor(A_f, dtype=torch.float64, device=DEVICE)
    A_m_inv = torch.tensor(np.linalg.inv(A_m), dtype=torch.float64, device=DEVICE)
    T_t = torch.tensor(T_ras, dtype=torch.float64, device=DEVICE)
    smooth = ndimage.gaussian_filter(moving, 3.0 / np.sqrt((A_m[:3, :3] ** 2).sum(axis=0)))
    coords = np.argwhere(fixed_mask)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(coords), min(n, len(coords)), replace=False)
    ijk = torch.tensor(coords[pick], dtype=torch.float64, device=DEVICE)
    values, valid = sample_moving(torch.tensor(smooth, dtype=torch.float32, device=DEVICE)[None, None],
                                  T_t, ijk, A_f_t, A_m_inv, moving.shape)
    valid = valid.cpu().numpy()
    warped = values.cpu().numpy()[valid]
    target = fixed[tuple(coords[pick][valid].T)]
    out = {"couverture": float(valid.mean()), "correlation": float(np.corrcoef(target, warped)[0, 1])}
    if t1 is not None:
        a = np.clip(t1[tuple(coords[pick][valid].T)], *np.percentile(t1[fixed_mask], [0.5, 99.5]))
        b = np.clip(warped, *np.percentile(warped, [0.5, 99.5]))
        joint, _, _ = np.histogram2d(a, b, bins=32)
        p = joint / joint.sum()
        h = lambda q: float(-(q[q > 0] * np.log(q[q > 0])).sum())
        out["nmi_avec_t1"] = (h(p.sum(axis=1)) + h(p.sum(axis=0))) / h(p)
    if moving_mask is not None:
        step = 2
        axes = [np.arange(0, s, step) for s in fixed_mask.shape]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        vm, vv = sample_moving(torch.tensor(moving_mask.astype(np.float32), device=DEVICE)[None, None], T_t,
                               torch.tensor(grid, dtype=torch.float64, device=DEVICE), A_f_t, A_m_inv,
                               moving_mask.shape)
        warped_mask = ((vm > 0.5) & vv).cpu().numpy()
        ref = fixed_mask[::step, ::step, ::step].reshape(-1)
        inter = float((warped_mask & ref).sum())
        out["dice_masque"] = 2.0 * inter / max(float(warped_mask.sum() + ref.sum()), 1.0)
        out["masque_couvert"] = inter / max(float(ref.sum()), 1.0)
    return out


def sanity_check(T_ras, max_aniso=1.15, scale_range=(0.75, 1.35)):
    """Alerte (sans bloquer) si la transformation est suspecte : valeurs singulières de la partie linéaire.
    Renvoie True si elle est plausible."""
    s = np.linalg.svd(T_ras[:3, :3], compute_uv=False)
    problems = []
    if s.max() / s.min() > max_aniso:
        problems.append(f"anisotropie {s.max() / s.min():.2f} > {max_aniso}")
    if s.min() < scale_range[0] or s.max() > scale_range[1]:
        problems.append(f"échelle {s.min():.2f}-{s.max():.2f} hors de {scale_range}")
    if problems:
        print("   ATTENTION, transformation suspecte : " + " ; ".join(problems) + ". Vérifiez la figure.")
    return not problems


# ----------------------------------------------------------------------------------------------
# D. Test de cohérence sur volumes fictifs
# ----------------------------------------------------------------------------------------------
def selftest():
    """Crée deux volumes fictifs (grilles et repères différents, transformation connue), recale, et compare.

    Valide les conventions (RAS/LPS, indices, ordre des axes de grid_sample) et la chaîne par étapes : si l'erreur
    dépasse 1 mm, il y a une erreur, et il ne faut PAS utiliser le script sur les vraies données.
    """
    import SimpleITK as sitk
    tmp = Path(tempfile.mkdtemp())
    shape_f = (48, 56, 44)
    A_f = np.diag([2.0, 2.0, 2.0, 1.0])
    A_f[:3, 3] = [-48.0, -56.0, -44.0]
    ii, jj, kk = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in shape_f], indexing="ij")

    def blob(c, r, v):
        return v * np.exp(-((ii - c[0]) ** 2 + (jj - c[1]) ** 2 + (kk - c[2]) ** 2) / (2 * r ** 2))

    fixed = (blob((24, 28, 22), 12, 1.0) + blob((16, 24, 26), 4, 1.5) + blob((32, 34, 18), 5, 1.2)
             + blob((26, 20, 30), 3, 2.0)).astype(np.float32)
    fixed_mask = fixed > 0.1 * fixed.max()
    nib.save(nib.Nifti1Image(fixed, A_f), str(tmp / "fixed.nii.gz"))
    fixed_sitk = sitk.ReadImage(str(tmp / "fixed.nii.gz"))

    shape_m, voxel_m = (40, 52, 36), np.array([2.4, 2.0, 2.6])
    centre_m_ras = np.array([20.0, -15.0, 10.0])
    A_m = np.diag(list(voxel_m) + [1.0])
    A_m[:3, 3] = centre_m_ras - voxel_m * (np.array(shape_m) - 1) / 2
    nib.save(nib.Nifti1Image(np.zeros(shape_m, np.float32), A_m), str(tmp / "grid.nii.gz"))
    grid = sitk.ReadImage(str(tmp / "grid.nii.gz"))

    euler = sitk.Euler3DTransform()
    euler.SetRotation(np.deg2rad(6), np.deg2rad(-4), np.deg2rad(8))
    linear = np.array(euler.GetMatrix()).reshape(3, 3) * 1.04
    centre_m_lps = RAS_LPS[:3, :3] @ centre_m_ras
    true_mf = sitk.AffineTransform(3)               # espace du PET -> espace fixe (sens du rééchantillonnage ITK)
    true_mf.SetMatrix(tuple(linear.ravel()))
    true_mf.SetTranslation(tuple(-linear @ centre_m_lps))
    moving_sitk = sitk.Resample(fixed_sitk, grid, true_mf, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    moving = np.transpose(sitk.GetArrayFromImage(moving_sitk), (2, 1, 0)).astype(np.float32)
    moving_mask = moving > 0.1 * moving.max()
    moving = mask_moving(moving, moving_mask, 2)    # même chemin que sur les vraies données (PET masqué)
    expected = itk_to_ras(true_mf.GetInverse())     # fixe -> mobile

    T0, centre = initial_matrix(A_f, fixed_mask, A_m, moving_mask, 0.0)
    T, _, _ = register_staged(fixed, A_f, moving, A_m, fixed_mask, T0, centre, model="affine", loss_name="ncc",
                              factors=(2, 1), sigmas_mm=(4.0, 0.0), iterations=(150, 150), n_samples=6000,
                              reg_weight=0.0, verbose=False)
    points = np.argwhere(fixed_mask)[::40]
    P = A_f @ np.c_[points, np.ones(len(points))].T
    err_start = np.linalg.norm((T0 @ P - expected @ P)[:3], axis=0).mean()
    err_end = np.linalg.norm((T @ P - expected @ P)[:3], axis=0).mean()

    back = sitk.Resample(moving_sitk, fixed_sitk, ras_to_itk(T), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    back = np.transpose(sitk.GetArrayFromImage(back), (2, 1, 0))
    both = fixed_mask & (back > 0)
    corr = float(np.corrcoef(fixed[both], back[both])[0, 1])
    rep0 = report(T0, fixed, A_f, moving, A_m, fixed_mask, None, moving_mask)
    rep1 = report(T, fixed, A_f, moving, A_m, fixed_mask, None, moving_mask)
    print(f"Erreur moyenne avant recalage : {err_start:.2f} mm | après : {err_end:.2f} mm")
    print(f"Corrélation fixe / mobile rééchantillonné par SimpleITK avec la transformation trouvée : {corr:.3f}")
    print(f"Dice des masques : départ {rep0['dice_masque']:.3f} -> final {rep1['dice_masque']:.3f} (information)")
    ok = err_end < 1.0 and corr > 0.95
    print("RÉSULTAT : OK, conventions cohérentes." if ok else
          "RÉSULTAT : ÉCHEC. Ne pas utiliser sur les vraies données ; envoyez-moi ce message.")
    return ok


# ----------------------------------------------------------------------------------------------
# D2. Figure de contrôle : PET recalé superposé à l'IRM MNI
# ----------------------------------------------------------------------------------------------
def save_overlay(t1_path, pet_path, out_png, title="", outline_mask_path=None):
    """Superpose le PET recalé (couleurs chaudes) à l'IRM (gris) : coupes axiale, coronale, sagittale.

    Ligne du haut : PET + IRM + contour du masque cérébral MNI. Ligne du bas : PET seul avec le même contour.
    Le seuil et l'échelle de couleur ne servent qu'à l'AFFICHAGE : les données du PET ne sont pas modifiées.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nibabel.processing import resample_from_to

    t1_img = nib.as_closest_canonical(nib.load(str(t1_path)))
    pet_img = nib.as_closest_canonical(nib.load(str(pet_path)))
    if pet_img.shape != t1_img.shape or not np.allclose(pet_img.affine, t1_img.affine, atol=1e-3):
        pet_img = resample_from_to(pet_img, (t1_img.shape, t1_img.affine), order=1)
    t1 = np.asarray(t1_img.dataobj, dtype=np.float32)
    pet = np.asarray(pet_img.dataobj, dtype=np.float32)
    outline = None
    if outline_mask_path and Path(outline_mask_path).exists():
        m_img = nib.as_closest_canonical(nib.load(str(outline_mask_path)))
        if m_img.shape == t1.shape:
            outline = np.asarray(m_img.dataobj) > 0

    positive = pet[pet > 0]
    if positive.size == 0:
        print("   figure ignorée : le PET recalé est vide (aucun voxel > 0).")
        return
    vmax = float(np.percentile(positive, 99.5))
    show_floor = 0.10 * vmax                                  # affichage seulement
    t_lo, t_hi = np.percentile(t1[t1 > 0], [1, 99]) if (t1 > 0).any() else (0.0, 1.0)

    inside = pet > show_floor
    centre = np.round(np.argwhere(inside).mean(axis=0)).astype(int) if inside.any() else \
        np.array(t1.shape) // 2
    zx, zy, zz = t1_img.header.get_zooms()[:3]
    views = [("Axiale", lambda a: a[:, :, centre[2]].T, zy / zx),
             ("Coronale", lambda a: a[:, centre[1], :].T, zz / zx),
             ("Sagittale", lambda a: a[centre[0], :, :].T, zz / zy)]

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for col, (name, cut, aspect) in enumerate(views):
        for row in range(2):
            ax = axes[row, col]
            if row == 0:
                ax.imshow(cut(t1), cmap="gray", origin="lower", vmin=t_lo, vmax=t_hi, aspect=aspect)
            else:
                ax.set_facecolor("black")
            pet_slice = np.ma.masked_where(cut(pet) <= show_floor, cut(pet))
            ax.imshow(pet_slice, cmap="hot", origin="lower", vmin=show_floor, vmax=vmax,
                      alpha=0.55 if row == 0 else 1.0, aspect=aspect)
            if outline is not None and cut(outline).any():
                ax.contour(cut(outline).astype(float), levels=[0.5], colors="cyan", linewidths=0.8,
                           origin="lower")
            ax.set_title(f"{name} ({'PET + IRM' if row == 0 else 'PET seul'})", fontsize=10)
            ax.axis("off")
    fig.suptitle(title or "PET recalé sur l'IRM MNI", fontsize=12)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=130)
    plt.close(fig)
    print(f"Figure de contrôle : {out_png}")


# ----------------------------------------------------------------------------------------------
# E. Ligne de commande (tout est lu dans config.yaml ; la ligne de commande surcharge ponctuellement)
# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Recalage affine 3D par PyTorch (sans ANTs), piloté par config.yaml.")
    ap.add_argument("--selftest", action="store_true", help="test de cohérence sur volumes fictifs")
    ap.add_argument("--config", default="config.yaml", help="fichier de configuration du projet")
    ap.add_argument("--attempt", nargs="+", help="ne lancer que ces essais (noms dans registration_dl.attempts)")
    ap.add_argument("--moving")
    ap.add_argument("--fixed-mask")
    ap.add_argument("--moving-mask")
    ap.add_argument("--t1", help="IRM de fond pour la figure et l'information mutuelle de contrôle")
    ap.add_argument("--init", help="transformation de départ SimpleITK (.tfm) ; remplace les inclinaisons")
    ap.add_argument("--starts", type=float, nargs="+")
    ap.add_argument("--model", choices=["rigid", "similarity", "affine"])
    ap.add_argument("--iterations", type=int)
    ap.add_argument("--samples", type=int)
    ap.add_argument("--regularization", type=float, help="poids de la pénalité d'étirement / cisaillement")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    import SimpleITK as sitk
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    root = cfg_path.parent                                   # chemins relatifs au dossier de config.yaml
    P, R, D = cfg.get("paths", {}), cfg.get("registration", {}), cfg.get("registration_dl", {})

    def path(value):
        if value is None:
            return None
        p = Path(str(value))
        return str(p if p.is_absolute() else root / p)

    def named(template, name):                               # "..._{name}.tfm" -> nom de l'essai
        return path(template.format(name=name)) if template else None

    def per_level(value, n):
        if isinstance(value, (list, tuple)):
            vals = tuple(int(v) for v in value)
            if len(vals) != n:
                ap.error(f"'iterations' doit avoir {n} valeurs (une par niveau) ou être un entier.")
            return vals
        return (int(value),) * n

    m_path = path(args.moving or D.get("moving") or P.get("reference_pet"))
    fm_path = path(args.fixed_mask or D.get("fixed_mask") or P.get("mni_brain_mask"))
    mm_path = path(args.moving_mask or D.get("moving_mask") or P.get("brain_mask"))
    t1_path = path(args.t1 or D.get("t1") or P.get("mni_template"))
    model = args.model or D.get("model", "affine")
    n_iter = args.iterations or D.get("iterations", 150)
    n_samples = args.samples or D.get("samples", 40000)
    n_samples_fine = int(D.get("samples_fine", 2 * n_samples))
    seed = args.seed if args.seed is not None else D.get("random_seed", R.get("random_seed", 0))
    starts_deg = args.starts if args.starts is not None else D.get("starts", [-90, -60, -30, 0, 30])
    min_nmi = float(D.get("min_nmi", 1.03))
    dilate_vox = int(D.get("moving_dilate_vox", 3))
    reg_weight = args.regularization if args.regularization is not None else float(D.get("regularization", 0.1))
    lr_translation = float(D.get("lr_translation", 1.0))
    lr_linear = float(D.get("lr_linear", 5e-3))
    factors = tuple(int(f) for f in D.get("shrink_factors", R.get("shrink_factors", [4, 2, 1])))
    sigmas = tuple(float(s) for s in D.get("smoothing_sigmas_mm", R.get("smoothing_sigmas_mm", [4, 2, 1])))
    if len(factors) != len(sigmas):
        ap.error("shrink_factors et smoothing_sigmas_mm doivent avoir la même longueur.")
    select_metric = D.get("select_metric", "nmi_avec_t1")
    metric_key = {"nmi_avec_t1": "nmi", "dice": "dice", "correlation": "corr"}
    if select_metric not in metric_key:
        ap.error("select_metric doit valoir nmi_avec_t1, dice ou correlation.")
    stop_on_success = bool(D.get("stop_on_success", False))

    attempts = D.get("attempts") or [{"name": "pseudo", "fixed": D.get("fixed"), "loss": D.get("loss", "ncc")}]
    if args.attempt:
        attempts = [a for a in attempts if a["name"] in args.attempt]
        if not attempts:
            ap.error(f"aucun essai ne porte le nom {args.attempt} dans config.yaml")

    for label, p in [("moving", m_path), ("fixed_mask", fm_path), ("moving_mask", mm_path)]:
        if not p or not Path(p).exists():
            ap.error(f"fichier introuvable pour '{label}' : {p} (vérifiez config.yaml)")

    started = time.time()
    moving, A_m = load_volume(m_path)
    fixed_mask, A_fm = load_volume(fm_path)
    fixed_mask = fixed_mask > 0
    moving_mask = load_volume(mm_path)[0] > 0
    if moving_mask.shape != moving.shape:
        ap.error(f"le masque du PET {moving_mask.shape} n'a pas la grille du PET {moving.shape}.")
    moving_reg = mask_moving(moving, moving_mask, dilate_vox)    # PET masqué : sert au critère et aux mesures
    print(f"Appareil de calcul : {DEVICE} | modèle : {model} (par étapes) | PET masqué (dilatation {dilate_vox} vox) "
          f"| régularisation {reg_weight} | sélection : {select_metric}")

    results = []
    for att in attempts:
        name, loss_name = att["name"], att["loss"]
        a_model = att.get("model", model)
        a_samples = int(att.get("samples", n_samples))
        a_samples_fine = int(att.get("samples_fine", 2 * a_samples if "samples" in att else n_samples_fine))
        a_reg = float(att.get("regularization", reg_weight))
        a_starts = att.get("starts", starts_deg)
        a_iters = per_level(att.get("iterations", n_iter), len(factors))
        f_path = path(att["fixed"]) if att.get("fixed") else None
        print(f"\n=== Essai '{name}' : image fixe = {Path(f_path).name if f_path else None} | critère = {loss_name} "
              f"| modèle = {a_model} ===")
        if not f_path or not Path(f_path).exists():
            print(f"   IGNORÉ : fichier introuvable ({f_path}).")
            continue
        fixed, A_f = load_volume(f_path)
        if fixed.shape != fixed_mask.shape or not np.allclose(A_f, A_fm, atol=1e-3):
            print(f"   IGNORÉ : l'image fixe {fixed.shape} et le masque {fixed_mask.shape} n'ont pas la même "
                  "grille (forme ou affine). Utilisez une image fixe sur la grille du masque.")
            continue
        t1 = None
        if t1_path and Path(t1_path).exists():
            t1_vol, A_t1 = load_volume(t1_path)
            if t1_vol.shape == fixed.shape and np.allclose(A_t1, A_f, atol=1e-3):
                t1 = t1_vol
            else:
                print("   Note : l'IRM de contrôle n'est pas sur la grille de l'image fixe : "
                      "nmi_avec_t1 non calculée pour cet essai.")
        _LEVEL_CACHE.clear()                                 # les niveaux dépendent de l'image fixe

        # --- point de départ : --init > init_from (autre essai) > recherche d'inclinaison ---
        init_file = args.init
        if not init_file and att.get("init_from"):
            prev = next((r["tfm"] for r in results if r["name"] == att["init_from"]), None)
            prev = prev or named(D.get("out", "transforms/02_dl_{name}.tfm"), att["init_from"])
            if prev and Path(prev).exists():
                init_file = prev
                print(f"   Départ : transformation de l'essai '{att['init_from']}' ({Path(prev).name})")
            else:
                print(f"   Note : transformation de l'essai '{att['init_from']}' introuvable ; départ standard.")
        if init_file:
            init, centre = itk_to_ras(sitk.ReadTransform(init_file)), center_of_mass_ras(fixed_mask, A_f)
        elif len(a_starts) > 1:
            print("Choix de l'inclinaison de départ (rigide, deux temps)...")
            init, centre = pick_start(fixed, A_f, moving_reg, A_m, fixed_mask, moving_mask, a_starts, loss_name,
                                      factors, sigmas, a_samples, seed)
        else:
            init, centre = initial_matrix(A_f, fixed_mask, A_m, moving_mask, float(a_starts[0]))

        base = report(init, fixed, A_f, moving_reg, A_m, fixed_mask, t1, moving_mask)
        print("Départ (avant recalage) : " + " | ".join(f"{k} = {v:.4f}" for k, v in base.items()))

        # un essai qui part d'une transformation existante n'utilise que les niveaux fins (par défaut 2)
        k = int(att.get("levels", 2 if init_file else len(factors)))
        k = max(1, min(k, len(factors)))
        print(f"Recalage par étapes ({a_model}) sur {k} niveau(x) de la pyramide gaussienne...")
        T, _, _ = register_staged(fixed, A_f, moving_reg, A_m, fixed_mask, init, centre, a_model, loss_name,
                                  factors[-k:], sigmas[-k:], a_iters[-k:], coarsest=max(factors),
                                  lr_translation=lr_translation, lr_linear=lr_linear, n_samples=a_samples,
                                  n_samples_fine=a_samples_fine, reg_weight=a_reg, seed=seed)

        transform = ras_to_itk(T)
        out_tfm = named(D.get("out", "transforms/02_dl_{name}.tfm"), name)
        Path(out_tfm).parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(transform, out_tfm)
        print(f"Transformation écrite : {out_tfm} (SimpleITK, fixe -> mobile)")
        sane = sanity_check(T)

        result = report(T, fixed, A_f, moving_reg, A_m, fixed_mask, t1, moving_mask)
        print("Contrôle : " + " | ".join(f"{k_} = {v:.4f}" for k_, v in result.items()))
        try:
            import registration as reg
            info = reg.describe_transform(transform)
            print(f"Rotations (degrés) {np.round(info['rotation_deg'], 1)} | échelles anatomiques "
                  f"{np.round(info['mni_scales'], 3)} | cisaillement y-z {info['shear_deg']['yz']:.1f}°")
        except Exception:
            sv = np.linalg.svd(T[:3, :3], compute_uv=False)
            print(f"Valeurs singulières (échelles) : {np.round(sv, 3)}")

        fixed_sitk = sitk.ReadImage(f_path)
        res_path = named(D.get("resampled"), name)
        if res_path:
            Path(res_path).parent.mkdir(parents=True, exist_ok=True)
            moving_sitk = sitk.Cast(sitk.ReadImage(m_path), sitk.sitkFloat32)      # PET d'origine, non masqué
            sitk.WriteImage(sitk.Resample(moving_sitk, fixed_sitk, transform, sitk.sitkLinear, 0.0,
                                          sitk.sitkFloat32), res_path)
            print(f"PET recalé : {res_path}")
        mask_out = named(D.get("mask_out"), name)
        if mask_out:
            Path(mask_out).parent.mkdir(parents=True, exist_ok=True)
            mask_sitk = sitk.Cast(sitk.ReadImage(mm_path), sitk.sitkUInt8)
            sitk.WriteImage(sitk.Resample(mask_sitk, fixed_sitk, transform, sitk.sitkNearestNeighbor, 0,
                                          sitk.sitkUInt8), mask_out)
            print(f"Masque déplacé : {mask_out}")
        overlay = named(D.get("overlay", "reports/qc_figures/02_dl_{name}_overlay.png"), name)
        if res_path and t1_path and Path(t1_path).exists():
            nmi_txt = f" | nmi = {result['nmi_avec_t1']:.3f}" if "nmi_avec_t1" in result else ""
            save_overlay(t1_path, res_path, overlay,
                         title=f"Essai '{name}' ({loss_name}) : corr = {result['correlation']:.3f}{nmi_txt}"
                               f" | Dice masques = {result['dice_masque']:.3f}",
                         outline_mask_path=fm_path)

        nmi = result.get("nmi_avec_t1")
        results.append({"name": name, "tfm": out_tfm, "nmi": nmi, "corr": result["correlation"],
                        "dice": result["dice_masque"], "sane": sane})
        if t1 is not None and loss_name != "ncc" and f_path == t1_path:
            print("   (nmi_avec_t1 n'est pas indépendante ici : l'essai optimisait cette même IRM. "
                  "Validez avec le Dice des masques et la figure.)")
        if stop_on_success and nmi is not None and nmi >= min_nmi and sane:
            print(f"Essai '{name}' retenu : nmi_avec_t1 >= {min_nmi} (stop_on_success). Essais suivants ignorés.")
            break

    if not results:
        print("\nAucun essai n'a pu être exécuté : vérifiez les chemins et les grilles ci-dessus.")
        raise SystemExit(1)

    print("\n=== Récapitulatif ===")
    for r in results:
        nmi_txt = f"{r['nmi']:.4f}" if r["nmi"] is not None else "n/a"
        print(f"   {r['name']:<12} nmi_avec_t1 = {nmi_txt} | corr = {r['corr']:.4f} | Dice masques = {r['dice']:.4f}"
              f" | plausible = {'oui' if r['sane'] else 'NON'}")

    pool = [r for r in results if r["sane"]] or results
    key = metric_key[select_metric]
    if all(r.get(key) is None for r in pool):
        key = "corr"
    chosen = max(pool, key=lambda r: -np.inf if r.get(key) is None else r[key])
    if not chosen["sane"]:
        print("\nATTENTION : aucun essai n'a une transformation plausible ; regardez les figures.")
    if chosen["nmi"] is not None and chosen["nmi"] < min_nmi:
        print(f"\nATTENTION : nmi_avec_t1 de l'essai retenu ({chosen['nmi']:.3f}) sous le seuil {min_nmi}. "
              "Comparez au 'Départ (avant recalage)' affiché plus haut et regardez la figure.")
    final = path(D.get("out_final", "transforms/02_dl.tfm"))
    Path(final).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(chosen["tfm"], final)
    print(f"\nTransformation retenue ('{chosen['name']}', selon {key}) copiée vers : {final}")
    print(f"Durée : {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()