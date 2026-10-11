"""Contrôle qualité : figures pour VOIR ce que le code a fait.

Un recalage ou un masque qui échoue ne déclenche pas d'erreur : seul
l'oeil le remarque. D'où ces figures.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # pas de fenêtre : on écrit directement dans un fichier
import matplotlib.pyplot as plt
import numpy as np


def save_three_views(data, path, title="", mask=None):
    """Enregistre 3 coupes (sagittale, coronale, axiale) au milieu du volume.

    Si `mask` est fourni, son contour est tracé en rouge par-dessus.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cx, cy, cz = (s // 2 for s in data.shape)

    # Chaque coupe = un tableau 2D ; .T et origin="lower" les remettent à l'endroit.
    slices = [
        ("Sagittale", data[cx, :, :], None if mask is None else mask[cx, :, :]),
        ("Coronale",  data[:, cy, :], None if mask is None else mask[:, cy, :]),
        ("Axiale",    data[:, :, cz], None if mask is None else mask[:, :, cz]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, img2d, m2d) in zip(axes, slices):
        ax.imshow(img2d.T, cmap="gray", origin="lower")
        if m2d is not None and m2d.any():
            ax.contour(m2d.T, levels=[0.5], colors="red", linewidths=0.8)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_mask_comparison(data, masks, titles, path):
    """Une ligne par masque candidat (3 coupes + contour rouge), pour choisir le bon seuil d'un coup d'oeil."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cx, cy, cz = (s // 2 for s in data.shape)
    fig, axes = plt.subplots(len(masks), 3, figsize=(10, 3.2 * len(masks)), squeeze=False)
    for row, (mask, title) in enumerate(zip(masks, titles)):
        views = [(data[cx], mask[cx]), (data[:, cy], mask[:, cy]), (data[:, :, cz], mask[:, :, cz])]
        for col, (img2d, m2d) in enumerate(views):
            ax = axes[row, col]
            ax.imshow(img2d.T, cmap="gray", origin="lower")
            if m2d.any():
                ax.contour(m2d.T, levels=[0.5], colors="red", linewidths=0.8)
            ax.axis("off")
            if col == 0:
                ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def save_axial_montage(data, mask, path, n_slices=12, margin=15, columns=4):
    """Mosaïque de coupes axiales réparties du bas au haut du masque, avec son contour rouge.

    Permet de vérifier TOUT le cerveau d'un coup d'oeil (et pas seulement les 3 coupes centrales).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    where = np.argwhere(mask > 0)
    low = np.maximum(where.min(axis=0) - margin, 0)
    high = np.minimum(where.max(axis=0) + margin + 1, mask.shape)
    z_values = np.unique(np.linspace(where[:, 2].min(), where[:, 2].max(), n_slices).astype(int))
    rows = int(np.ceil(len(z_values) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, z in zip(axes.ravel(), z_values):
        img2d = data[low[0]:high[0], low[1]:high[1], z]
        m2d = mask[low[0]:high[0], low[1]:high[1], z]
        ax.imshow(img2d.T, cmap="gray", origin="lower")
        if m2d.any():
            ax.contour(m2d.T, levels=[0.5], colors="red", linewidths=0.8)
        ax.set_title(f"coupe z = {z}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def save_registration_check(background, moving, path, title="", outline=None):
    """Contrôle d'un recalage : 3 coupes passant par le centre du cerveau du PET.

    Ligne 1 : l'IRM (gris) avec le PET recalé par-dessus (couleurs chaudes, semi-transparent).
    Ligne 2 (si `outline` est fourni) : le PET recalé avec la silhouette du cerveau (issue de la carte de régions) en rouge.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    positive = moving[moving > 0]
    threshold = 0.2 * np.percentile(positive, 99.5)
    cx, cy, cz = (int(round(v)) for v in np.argwhere(moving > threshold).mean(axis=0))

    rows = 2 if outline is not None else 1
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4.2 * rows), squeeze=False)
    names = ["Sagittale", "Coronale", "Axiale"]
    cuts = [(slice(cx, cx + 1), 0), (slice(cy, cy + 1), 1), (slice(cz, cz + 1), 2)]

    def take(volume, axis, index):
        return np.take(volume, index, axis=axis)

    for col, ((_, axis), index) in enumerate(zip(cuts, (cx, cy, cz))):
        bg2d = take(background, axis, index)
        pet2d = take(moving, axis, index)
        ax = axes[0, col]
        ax.imshow(bg2d.T, cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_less(pet2d, threshold).T, cmap="hot", alpha=0.5, origin="lower")
        ax.set_title(f"{names[col]} : IRM + PET", fontsize=10)
        ax.axis("off")
        if outline is not None:
            ax = axes[1, col]
            ax.imshow(pet2d.T, cmap="gray", origin="lower")
            out2d = take(outline, axis, index)
            if out2d.any():
                ax.contour(out2d.T, levels=[0.5], colors="red", linewidths=0.8)
            ax.set_title(f"{names[col]} : PET + silhouette", fontsize=10)
            ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _boundaries(label2d):
    """Pixels situés à la frontière entre deux régions (ou entre une région et le fond)."""
    edge = np.zeros(label2d.shape, dtype=bool)
    edge[:-1, :] |= label2d[:-1, :] != label2d[1:, :]
    edge[1:, :] |= label2d[:-1, :] != label2d[1:, :]
    edge[:, :-1] |= label2d[:, :-1] != label2d[:, 1:]
    edge[:, 1:] |= label2d[:, :-1] != label2d[:, 1:]
    return edge & (label2d > 0)


def save_regions_check(pet, labels, path, title=""):
    """Contrôle d'une carte de régions posée sur un PET.

    Ligne 1 : 3 coupes passant par le centre du cerveau, régions en couleurs semi-transparentes.
    Ligne 2 : 3 coupes axiales (bas, milieu, haut) avec seulement les FRONTIÈRES des régions :
              elles doivent suivre le cortex du PET.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    positive = pet[pet > 0]
    threshold = 0.2 * np.percentile(positive, 99.5)
    cx, cy, cz = (int(round(v)) for v in np.argwhere(pet > threshold).mean(axis=0))
    z_has = np.nonzero(((labels > 0) & (pet > threshold)).sum(axis=(0, 1)))[0]
    levels = [int(np.quantile(z_has, q)) for q in (0.2, 0.5, 0.8)]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8.4))
    views = [("Sagittale", pet[cx], labels[cx]), ("Coronale", pet[:, cy], labels[:, cy]),
             ("Axiale", pet[:, :, cz], labels[:, :, cz])]
    vmax = max(int(labels.max()), 1)
    for ax, (name, p2, l2) in zip(axes[0], views):
        ax.imshow(p2.T, cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_equal(l2.T, 0), cmap="nipy_spectral", alpha=0.4, vmin=1, vmax=vmax,
                  origin="lower", interpolation="nearest")
        ax.set_title(f"{name} : régions", fontsize=10)
        ax.axis("off")
    for ax, z in zip(axes[1], levels):
        p2, l2 = pet[:, :, z], labels[:, :, z]
        ax.imshow(p2.T, cmap="gray", origin="lower")
        edge = _boundaries(l2)
        ax.imshow(np.ma.masked_where(~edge.T, edge.T), cmap="autumn", origin="lower", interpolation="nearest")
        ax.set_title(f"Axiale z = {z} : frontières", fontsize=10)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def save_gallery(rows, outline, path, title=""):
    """Galerie de comparaison : une ligne par résultat, MÊMES coupes et MÊME fenêtre d'affichage partout.

    rows    : liste de (étiquette, volume PET sur la grille du MNI152)
    outline : silhouette du cerveau du MNI152 (tracée en rouge)
    Colonnes : sagittale, coronale, axiale haute, axiale basse (le bas du cerveau révèle une inclinaison fausse :
    on doit y voir le cervelet à l'intérieur du contour, pas les lobes frontaux).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    where = np.argwhere(outline)
    cx, cy, cz = (int(round(v)) for v in where.mean(axis=0))
    z_low = int(np.quantile(where[:, 2], 0.2))
    first = rows[0][1]
    vmax = float(np.percentile(first[first > 0], 99.5))
    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.5 * len(rows)), squeeze=False)
    for r, (label, pet) in enumerate(rows):
        panels = [("sagittale", pet[cx], outline[cx]), ("coronale", pet[:, cy], outline[:, cy]),
                  ("axiale haute", pet[:, :, cz], outline[:, :, cz]),
                  (f"axiale basse (z={z_low})", pet[:, :, z_low], outline[:, :, z_low])]
        for c, (name, p2, o2) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(p2.T, cmap="gray", origin="lower", vmin=0, vmax=vmax)
            if o2.any():
                ax.contour(o2.T, levels=[0.5], colors="red", linewidths=0.8)
            ax.axis("off")
            if r == 0:
                ax.set_title(name, fontsize=10)
            if c == 0:
                ax.text(0.02, 0.97, label, transform=ax.transAxes, va="top", color="yellow", fontsize=8,
                        bbox=dict(facecolor="black", alpha=0.6, pad=2))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=75)
    plt.close(fig)


def save_pose_curve(pitches, dices, chosen_pitches, path):
    """Courbe du balayage de départ : recouvrement des masques (Dice) selon l'inclinaison essayée."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(pitches, dices, "-o", ms=3)
    for k, p in enumerate(chosen_pitches, start=1):
        ax.axvline(p, color="red", ls="--", lw=0.8)
        ax.text(p, min(dices), f" {k}", color="red", va="bottom")
    ax.set_xlabel("inclinaison essayée (degrés)")
    ax.set_ylabel("recouvrement des masques (Dice)")
    ax.set_title("Départ par la forme (les candidats retenus sont en rouge)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
