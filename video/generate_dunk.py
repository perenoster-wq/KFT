#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génère une courte vidéo MP4 (GIF optionnel) : une caméra de sport suit un
joueur de basket qui réalise un dunk, avec un mouvement fluide.

Fonctionnement 100 % local et open-source :
  * rendu procédural 2.5D (NumPy + Pillow) : terrain, panneau, cercle, filet,
    joueur articulé (cinématique inverse simplifiée) et ballon avec physique ;
  * caméra « suiveuse » simulée par un ressort amorti (pas de saccades) ;
  * flou de mouvement par sous-images moyennées et anti-crénelage (supersampling) ;
  * encodage H.264 via le binaire FFmpeg fourni par le paquet `imageio-ffmpeg`.

Pourquoi pas un modèle de diffusion (diffusers) ici ?
  Les poids des modèles sont hébergés sur huggingface.co, joignable depuis
  l'environnement d'exécution. Même avec les poids, un modèle text-to-video
  tourne en pratique trop lentement sur 2 vCPU sans GPU.

Gestion des contraintes matérielles :
  * détection des CPU disponibles (affinité + quota cgroup) et de la RAM ;
  * preset automatique (haute / moyenne / basse résolution) ;
  * benchmark des premières images : si la durée estimée dépasse le budget
    (--budget), le script relance automatiquement avec un preset plus léger ;
  * MemoryError -> relance avec un preset plus léger ;
  * FFmpeg indisponible ou échec d'encodage -> repli sur un GIF animé.

Usage :
    python generate_dunk.py                    # auto
    python generate_dunk.py --quality low      # forcer la basse qualité
    python generate_dunk.py --gif              # exporter aussi un GIF
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "output" / "dunk_suivi_camera.mp4"

# ----------------------------------------------------------------- constantes
FPS = 30
DURATION = 7.0                # secondes de vidéo
FOCAL_1280 = 1000.0           # focale (px) pour une image de 1280 px de large
HORIZON_RATIO = 0.42          # position de l'horizon (fraction de la hauteur)
CAM_Z = -10.0                 # position de la caméra sur l'axe de profondeur
WALL_Z = 12.0                 # gradins au fond de la salle
PLAYER_Z = -0.9               # profondeur du joueur
HOOP_X = 0.60                 # position latérale du panier
BOARD_Z = 0.0                 # plan du panneau
RIM_Y = 3.05                  # hauteur du cercle (m)
RIM_Z = -0.4                  # profondeur du centre du cercle
RIM_R = 0.229                 # rayon du cercle
BALL_R = 0.12
T_RELEASE = 2.90              # instant où le ballon quitte la main
GRAVITY = 9.81
ANKLE_H = 0.08                # hauteur de la cheville au sol
LEG = 0.50                    # longueur cuisse / tibia
TORSO_LEN = 0.62              # bassin -> épaule
UPPER_ARM = 0.33
FOREARM = 0.31
RUN_PERIOD = 0.62             # durée d'un cycle de foulée (s)

# Couleurs
JERSEY = (214, 40, 57)
JERSEY_DARK = (140, 22, 36)
SKIN = (198, 134, 90)
SKIN_DARK = (140, 90, 58)
SHORTS = (28, 42, 84)
SHORTS_DARK = (16, 26, 54)
SOCK = (236, 236, 236)
SHOE = (244, 244, 244)
HAIR = (28, 18, 14)
ORANGE = (236, 112, 30)
FLOOR_FAR = (120, 70, 40)
FLOOR_NEAR = (186, 122, 72)
LANE = (190, 62, 52)


@dataclass(frozen=True)
class Preset:
    name: str
    width: int    # largeur de sortie
    height: int   # hauteur de sortie
    ss: int       # facteur de supersampling (anti-crénelage)
    subs: int     # sous-images par frame (flou de mouvement)


PRESETS = [
    Preset("haute", 1280, 720, 2, 3),
    Preset("moyenne", 960, 540, 2, 2),
    Preset("basse", 640, 360, 1, 1),
]


class Downgrade(Exception):
    """Levée quand le benchmark prévoit de dépasser le budget de temps."""


# ------------------------------------------------------------- ressources
def effective_cpu_count() -> int:
    """Nombre de CPU réellement utilisables (affinité + quota cgroup v2)."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota, period = fh.read().split()
        if quota != "max":
            n = min(n, max(1, math.floor(float(quota) / float(period))))
    except (OSError, ValueError):
        pass
    return max(1, n)


def available_ram_gb() -> float | None:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return None


def auto_preset_index(cpus: int, ram_gb: float | None) -> int:
    if cpus >= 4 and (ram_gb is None or ram_gb >= 6):
        return 0
    if cpus >= 2 and (ram_gb is None or ram_gb >= 2):
        return 1
    return 2


def load_font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for path in (f"/usr/share/fonts/truetype/dejavu/{name}", name):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


# ------------------------------------------------------------ interpolation
class Track:
    """Interpolation de Hermite cubique (tangentes de Catmull-Rom), C1 continue."""

    def __init__(self, times, values):
        self.t = np.asarray(times, dtype=float)
        v = np.asarray(values, dtype=float)
        if v.ndim == 1:
            v = v[:, None]
        self.v = v
        n = len(self.t)
        m = np.zeros_like(v)
        if n >= 2:
            m[0] = (v[1] - v[0]) / (self.t[1] - self.t[0])
            m[-1] = (v[-1] - v[-2]) / (self.t[-1] - self.t[-2])
        for i in range(1, n - 1):
            m[i] = (v[i + 1] - v[i - 1]) / (self.t[i + 1] - self.t[i - 1])
        self.m = m

    def __call__(self, t: float) -> np.ndarray:
        t = min(max(float(t), self.t[0]), self.t[-1])
        i = int(np.searchsorted(self.t, t, side="right") - 1)
        i = min(max(i, 0), len(self.t) - 2)
        t0, t1 = self.t[i], self.t[i + 1]
        h = t1 - t0
        u = (t - t0) / h
        h00 = (1 + 2 * u) * (1 - u) ** 2
        h10 = u * (1 - u) ** 2
        h01 = u * u * (3 - 2 * u)
        h11 = u * u * (u - 1)
        return h00 * self.v[i] + h10 * h * self.m[i] + h01 * self.v[i + 1] + h11 * h * self.m[i + 1]


def dirv(a_deg: float) -> np.ndarray:
    """Direction d'un membre : 0° = vers le bas, 90° = vers l'avant (+X), 180° = vers le haut."""
    r = math.radians(a_deg)
    return np.array([math.sin(r), -math.cos(r)])


def upv(a_deg: float) -> np.ndarray:
    """Direction du tronc : 0° = vertical, positif = penché vers l'avant."""
    r = math.radians(a_deg)
    return np.array([math.sin(r), math.cos(r)])


# -------------------------------------------------------------- animation
POSE_FIELDS = ("torso", "thL", "kL", "uaL", "eL", "thR", "kR", "uaR", "eR", "lift")

# (t, torse, cuisseG, genouG, brasG, coudeG, cuisseD, genouD, brasD, coudeD, saut)
MANUAL = [
    (1.20, 10, 50, 60, -30, 60, -15, 35, -40, 60, 0.00),   # dernier pas, pose d'appel
    (1.45, 22, 70, 95, -65, 95, -10, 60, -75, 95, 0.00),   # accroupi, bras en arrière
    (1.72, 6, 25, 20, 100, 30, -30, 12, 110, 30, 0.22),    # impulsion
    (2.00, -2, 45, 75, 150, 10, 20, 80, 160, 5, 0.72),     # envol, bras qui montent
    (2.25, -8, 60, 95, 170, 0, 40, 100, 176, -2, 1.05),    # bras tendu vers le cercle
    (2.45, -10, 50, 80, 150, 10, 30, 70, 178, -5, 1.15),   # apogée : main au-dessus du cercle
    (2.68, -2, 30, 55, 110, 30, -10, 40, 172, -20, 1.08),  # amorce de la descente du bras
    (2.90, 4, 25, 45, 60, 60, -15, 35, 160, -50, 0.95),    # relâché : ballon dans l'arceau
    (3.02, 8, 25, 45, 20, 60, -10, 35, 110, -15, 0.75),    # bras qui se rabat
    (3.12, 12, 30, 60, 20, 50, -10, 40, 60, 0, 0.50),      # retombée
    (3.50, 15, 20, 40, -10, 60, -5, 20, -10, 60, 0.12),
    (3.80, 18, 40, 75, -20, 70, 15, 65, -30, 60, 0.00),    # réception amortie
    (4.20, 6, 10, 15, -10, 30, -5, 10, -10, 30, 0.00),
]

# Position horizontale (X) du bassin dans le temps
X_KEYS = [
    (0.0, -7.8), (0.6, -5.0), (1.2, -2.6), (1.5, -1.9), (1.8, -1.2),
    (2.25, 0.05), (2.45, 0.30), (2.68, 0.33), (2.9, 0.20), (3.4, 0.50),
    (4.2, 1.2), (5.2, 2.6), (6.2, 4.1), (DURATION + 0.6, 5.5),
]


def run_rows(t0: float, t1: float, speed_fn, step: float = 1 / 12) -> list[list[float]]:
    """Lignes de cinématique pour une foulée (cycle de course procédural)."""
    rows = []
    n = int(math.floor((t1 - t0) / step + 1e-9))
    for k in range(n + 1):
        t = t0 + k * step
        s = speed_fn(t)
        ph = 2 * math.pi * t / RUN_PERIOD
        amp = 16 + 30 * s
        sn, cs = math.sin(ph), math.cos(ph)
        rows.append([
            t,
            5 + 7 * s,                       # torse
            amp * sn,                        # cuisse G
            10 + 60 * s * max(0.0, cs),      # genou G
            -0.8 * amp * sn,                 # bras G
            80 + 10 * s,                     # coude G
            -amp * sn,                       # cuisse D
            10 + 60 * s * max(0.0, -cs),     # genou D
            0.8 * amp * sn,                  # bras D
            80 + 10 * s,                     # coude D
            0.0,                             # saut
        ])
    return rows


class Animation:
    """Cinématique du joueur, trajectoire de la caméra et du ballon (pré-calculées)."""

    WX, WY = 4.0, 2.5          # raideur du ressort de la caméra (horizontal / vertical)

    def __init__(self):
        run_a = run_rows(0.0, 1.1, lambda t: min(1.0, 0.55 + 0.45 * t / 0.9))
        run_b = run_rows(4.6, DURATION + 0.6, lambda t: 0.45 + 0.45 * min(1.0, (t - 4.6) / 1.5))
        rows = sorted(run_a + [list(r) for r in MANUAL] + run_b, key=lambda r: r[0])
        self._pose = Track([r[0] for r in rows], [r[1:] for r in rows])
        self._x = Track([k[0] for k in X_KEYS], [[k[1]] for k in X_KEYS])
        self._cam_t, self._cam_x, self._cam_y = self._simulate_camera()
        self._ball_t, self._ball_x, self._ball_y = self._simulate_ball()

    # -- état du joueur
    def pose(self, t: float) -> dict:
        d = dict(zip(POSE_FIELDS, self._pose(t)))
        d["x"] = float(self._x(t)[0])
        return d

    def vx(self, t: float, h: float = 1e-3) -> float:
        return (float(self._x(t + h)[0]) - float(self._x(t - h)[0])) / (2 * h)

    def skeleton(self, t: float) -> dict:
        p = self.pose(t)
        thL, kL = p["thL"], p["kL"]
        thR, kR = p["thR"], p["kR"]
        # hauteur du bassin pour que le pied le plus bas touche le sol (+ saut)
        ground_hip = ANKLE_H + max(
            LEG * math.cos(math.radians(thL)) + LEG * math.cos(math.radians(thL - kL)),
            LEG * math.cos(math.radians(thR)) + LEG * math.cos(math.radians(thR - kR)),
        )
        hip = np.array([p["x"], ground_hip + p["lift"]])
        sh = hip + upv(p["torso"]) * TORSO_LEN
        head = sh + upv(p["torso"]) * 0.22
        kneeL = hip + dirv(thL) * LEG
        ankleL = kneeL + dirv(thL - kL) * LEG
        kneeR = hip + dirv(thR) * LEG
        ankleR = kneeR + dirv(thR - kR) * LEG
        elbowL = sh + dirv(p["uaL"]) * UPPER_ARM
        wristL = elbowL + dirv(p["uaL"] + p["eL"]) * FOREARM
        elbowR = sh + dirv(p["uaR"]) * UPPER_ARM
        wristR = elbowR + dirv(p["uaR"] + p["eR"]) * FOREARM
        held = wristR + dirv(p["uaR"] + p["eR"]) * 0.10
        return dict(hip=hip, sh=sh, head=head, kneeL=kneeL, ankleL=ankleL,
                    kneeR=kneeR, ankleR=ankleR, elbowL=elbowL, wristL=wristL,
                    elbowR=elbowR, wristR=wristR, held=held, torso=p["torso"])

    # -- caméra : ressort amorti (critique) qui suit le joueur avec un léger anticipé
    def _simulate_camera(self):
        dt = 1 / 240
        ts = np.arange(0.0, DURATION + 0.6 + dt, dt)
        cx = float(self._x(0.0)[0])
        vcx = 0.0
        cy = 1.6 + 0.3 * self.pose(0.0)["lift"]
        vcy = 0.0
        xs, ys = [], []
        for t in ts:
            tx = float(self._x(t)[0]) + 0.3 * self.vx(t)
            ty = 1.6 + 0.3 * self.pose(t)["lift"]
            vcx += (self.WX ** 2 * (tx - cx) - 2 * self.WX * vcx) * dt
            cx += vcx * dt
            vcy += (self.WY ** 2 * (ty - cy) - 2 * self.WY * vcy) * dt
            cy += vcy * dt
            xs.append(cx)
            ys.append(cy)
        return ts, np.array(xs), np.array(ys)

    def camera(self, t: float) -> tuple[float, float]:
        return float(np.interp(t, self._cam_t, self._cam_x)), float(np.interp(t, self._cam_t, self._cam_y))

    # -- ballon : main jusqu'au relâché, puis chute, rebonds
    def _simulate_ball(self):
        p0 = self.skeleton(T_RELEASE)["held"]
        x, y = float(p0[0]), float(p0[1])
        vx, vy = 0.0, 0.0
        bounces = 0
        dt = 1 / 600
        t = T_RELEASE
        ts, xs, ys = [t], [x], [y]
        while t < DURATION + 1.0:
            vy -= GRAVITY * dt
            x += vx * dt
            y += vy * dt
            if y < BALL_R:
                y = BALL_R
                vy = -vy * 0.6
                if bounces == 0:
                    vx = 0.9              # premier rebond : le ballon part vers l'avant
                bounces += 1
                if abs(vy) < 0.4:
                    vy = 0.0
            if vy == 0.0:
                vx *= 0.985           # frottement au sol
            t += dt
            ts.append(t)
            xs.append(x)
            ys.append(y)
        return np.array(ts), np.array(xs), np.array(ys)

    def ball(self, t: float, sk: dict) -> tuple[float, float]:
        if t < T_RELEASE:
            return float(sk["held"][0]), float(sk["held"][1])
        return float(np.interp(t, self._ball_t, self._ball_x)), float(np.interp(t, self._ball_t, self._ball_y))


# ---------------------------------------------------------------- rendu
class SceneRenderer:
    """Dessine une frame à la résolution de travail (sur-échantillonnée)."""

    X0, X1 = -40.0, 40.0          # étendue du fond (gradins)
    YT, YB = 12.0, -1.0           # bornes verticales du fond

    def __init__(self, preset: Preset, anim: Animation):
        self.preset = preset
        self.anim = anim
        self.ss = preset.ss
        self.W = preset.width * preset.ss
        self.H = preset.height * preset.ss
        self.f = FOCAL_1280 * self.W / 1280.0
        self.hz = self.H * HORIZON_RATIO
        self.sw = self.f / (WALL_Z - CAM_Z)      # échelle au niveau des gradins
        self.sp = self.f / (PLAYER_Z - CAM_Z)    # échelle au niveau du joueur
        self.cx = 0.0
        self.cy = 1.6
        self.font_hud = load_font(max(12, int(preset.height * 0.045)))
        self.font_small = load_font(max(10, int(preset.height * 0.03)), bold=False)
        self.panorama = self._build_panorama()
        self.floor_grad = self._build_floor_gradient()
        self.vignette = self._build_vignette(preset.width, preset.height)

    # -- projection perspective : (X, Y, Z) monde -> pixels canevas
    def P(self, X: float, Y: float, Z: float) -> tuple[float, float]:
        s = self.f / (Z - CAM_Z)
        return (self.W / 2 + (X - self.cx) * s, self.hz - (Y - self.cy) * s)

    def S(self, Z: float) -> float:
        return self.f / (Z - CAM_Z)

    # -- fond : gradins + bannières + publicité, pré-rendus une fois
    def _build_panorama(self) -> Image.Image:
        sw = self.sw
        rng = np.random.default_rng(7)
        pw = int(round((self.X1 - self.X0) * sw))
        ph = int(round((self.YT - self.YB) * sw))
        img = Image.new("RGB", (pw, ph), (10, 12, 22))
        d = ImageDraw.Draw(img)

        def px(X):
            return (X - self.X0) * sw

        def py(Y):
            return (self.YT - Y) * sw

        for r in range(ph):                       # dégradé vertical
            Y = self.YT - r / sw
            u = min(1.0, max(0.0, (8.0 - Y) / 7.0))
            col = tuple(int(a + (b - a) * u) for a, b in zip((9, 12, 24), (40, 44, 70)))
            d.line([(0, r), (pw, r)], fill=col)

        palette = [(190, 60, 50), (60, 90, 180), (220, 200, 160), (40, 40, 52),
                   (200, 200, 210), (230, 160, 40), (80, 140, 90), (120, 50, 140)]
        Y = 1.6
        while Y <= 8.0:                           # rangées de spectateurs
            for X in np.arange(self.X0, self.X1, 0.2):
                col = palette[rng.integers(len(palette))]
                k = rng.uniform(0.35, 0.9)
                col = tuple(int(c * k) for c in col)
                cxp, cyp = px(X + rng.uniform(-0.05, 0.05)), py(Y + rng.uniform(-0.05, 0.05))
                rr = 0.11 * sw
                d.ellipse([cxp - rr, cyp - rr, cxp + rr, cyp + rr], fill=col)
            Y += 0.45

        font_banner = load_font(max(8, int(0.8 * sw)))
        for X in np.arange(-36.0, 40.0, 12.0):    # bannières
            x0, x1 = px(X), px(X + 6.0)
            d.rectangle([x0, py(2.0), x1, py(0.9)], fill=(170, 22, 40))
            d.text(((x0 + x1) / 2, (py(2.0) + py(0.9)) / 2), "KFT", font=font_banner,
                   fill=(245, 245, 245), anchor="mm")

        for X in np.arange(self.X0, self.X1, 0.5):  # bandeau LED
            col = palette[rng.integers(len(palette))]
            d.rectangle([px(X), py(0.6), px(X + 0.45), py(0.1)], fill=col)

        for _ in range(180):                        # projecteurs
            X = rng.uniform(self.X0, self.X1)
            Y = rng.uniform(9.0, 10.5)
            rr = rng.uniform(0.12, 0.35) * sw
            d.ellipse([px(X) - rr, py(Y) - rr, px(X) + rr, py(Y) + rr], fill=(250, 250, 235))
        return img

    def _build_floor_gradient(self) -> Image.Image:
        rows = np.arange(self.H, dtype=float)
        u = np.clip((rows - self.hz) / max(1.0, self.H - self.hz), 0.0, 1.0)[:, None]
        far, near = np.array(FLOOR_FAR, float), np.array(FLOOR_NEAR, float)
        col = far + (near - far) * u                # (H, 3)
        arr = np.repeat(col[:, None, :], self.W, axis=1).astype(np.uint8)
        return Image.fromarray(arr, "RGB")

    @staticmethod
    def _build_vignette(w: int, h: int) -> np.ndarray:
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2
        return (1.0 - 0.28 * np.clip(r2 / 2.0, 0, 1)).astype(np.float32)

    # -- primitives
    def _line_floor(self, d, pts, color, width_m: float):
        proj = [self.P(X, 0.0, Z) for X, Z in pts]
        zmid = sum(Z for _, Z in pts) / len(pts)
        w = max(1, int(round(width_m * self.S(zmid))))
        d.line(proj, fill=color, width=w, joint="curve")

    def _capsule(self, d, a, b, ra: float, rb: float, color, z: float = PLAYER_Z):
        """Membre arrondi (a, b : points monde XY ; ra, rb : rayons en mètres)."""
        s = self.f / (z - CAM_Z)
        pa = np.array(self.P(a[0], a[1], z))
        pb = np.array(self.P(b[0], b[1], z))
        sa, sb = ra * s, rb * s
        v = pb - pa
        L = float(np.hypot(*v))
        if L < 1e-6:
            d.ellipse([pa[0] - sa, pa[1] - sa, pa[0] + sa, pa[1] + sa], fill=color)
            return
        n = np.array([-v[1], v[0]]) / L
        poly = [tuple(pa + n * sa), tuple(pb + n * sb), tuple(pb - n * sb), tuple(pa - n * sa)]
        d.polygon(poly, fill=color)
        d.ellipse([pa[0] - sa, pa[1] - sa, pa[0] + sa, pa[1] + sa], fill=color)
        d.ellipse([pb[0] - sb, pb[1] - sb, pb[0] + sb, pb[1] + sb], fill=color)

    # -- scène
    def draw_scene(self, t: float) -> Image.Image:
        cx, cy = self.anim.camera(t)
        self.cx, self.cy = cx, cy
        W, H, hz, sw = self.W, self.H, self.hz, self.sw
        img = Image.new("RGB", (W, H), (10, 12, 22))

        # fond (gradins), décalé par parallaxe : le plan est lointain
        dx = W / 2 + (self.X0 - cx) * sw
        dy = hz - (self.YT - cy) * sw
        img.paste(self.panorama, (int(round(dx)), int(round(dy))))

        # sol : du pied des gradins jusqu'au bas de l'image
        yb = int(round(hz + cy * sw))
        yb = min(max(yb, 0), H)
        if yb < H:
            img.paste(self.floor_grad.crop((0, yb, W, H)), (0, yb))
        d = ImageDraw.Draw(img)
        d.line([(0, yb), (W, yb)], fill=(30, 22, 30), width=max(1, int(2 * self.ss)))

        # lames du parquet (lignes parallèles à la profondeur)
        cx_k = int(math.floor(cx / 0.6))
        for k in range(cx_k - 60, cx_k + 61):
            X = k * 0.6
            self._line_floor(d, [(X, -9.5), (X, 11.9)], (150, 96, 56), 0.012)

        # marquages du terrain
        hx = HOOP_X
        lane = [(hx - 2.45, -4.6), (hx + 2.45, -4.6), (hx + 2.45, 1.2), (hx - 2.45, 1.2)]
        d.polygon([self.P(X, 0.0, Z) for X, Z in lane], fill=LANE)
        white = (245, 245, 245)
        self._line_floor(d, [(hx - 2.45, -4.6), (hx - 2.45, 1.2)], white, 0.05)
        self._line_floor(d, [(hx + 2.45, -4.6), (hx + 2.45, 1.2)], white, 0.05)
        self._line_floor(d, [(hx - 2.45, -4.6), (hx + 2.45, -4.6)], white, 0.05)
        self._line_floor(d, [(-30.0, 1.2), (30.0, 1.2)], white, 0.05)
        arc = [(hx + 7.24 * math.sin(a), RIM_Z - 7.24 * math.cos(a))
               for a in np.linspace(-1.35, 1.35, 60)]
        self._line_floor(d, arc, white, 0.05)

        # poteau, panneau, cercle (partie arrière), filet (arrière)
        self._draw_pole(d)
        self._draw_board(d)
        self._draw_rim_and_net(d, t, front=False)

        # ombre du joueur
        sk = self.anim.skeleton(t)
        lift = self.anim.pose(t)["lift"]
        sx_, sy_ = self.P(sk["hip"][0], 0.0, PLAYER_Z)
        s = self.sp
        rx = 0.5 * s * (1 - 0.25 * min(lift, 1.0))
        ry = 0.12 * s * (1 - 0.25 * min(lift, 1.0))
        d.ellipse([sx_ - rx, sy_ - ry, sx_ + rx, sy_ + ry], fill=(90, 52, 30))

        self._draw_player(d, sk)

        # ballon
        bx, by = self.anim.ball(t, sk)
        px_, py_ = self.P(bx, by, PLAYER_Z)
        self._draw_ball(d, px_, py_, BALL_R * self.sp, t)

        # cercle (partie avant) et filet (avant)
        self._draw_rim_and_net(d, t, front=True)
        return img

    def _draw_pole(self, d):
        base = self.P(HOOP_X, 0.0, 1.0)
        top = self.P(HOOP_X, 2.9, 1.0)
        w = max(2, int(0.12 * self.S(1.0)))
        d.line([base, top], fill=(110, 110, 120), width=w)
        arm_a = self.P(HOOP_X, 2.9, 1.0)
        arm_b = self.P(HOOP_X, 2.9, BOARD_Z)
        d.line([arm_a, arm_b], fill=(110, 110, 120), width=max(2, int(0.06 * self.S(0.5))))

    def _draw_board(self, d):
        hx = HOOP_X
        z = BOARD_Z
        outer = [self.P(hx - 0.9, 2.9, z), self.P(hx + 0.9, 2.9, z),
                 self.P(hx + 0.9, 4.1, z), self.P(hx - 0.9, 4.1, z)]
        d.polygon(outer, fill=(246, 246, 246), outline=(180, 180, 190))
        inner = [self.P(hx - 0.295, 3.15, z), self.P(hx + 0.295, 3.15, z),
                 self.P(hx + 0.295, 3.6, z), self.P(hx - 0.295, 3.6, z)]
        d.polygon(inner, outline=(220, 40, 30), width=max(2, int(0.05 * self.S(z))))

    def _draw_rim_and_net(self, d, t: float, front: bool):
        """Cercle + filet. front=True : moitié avant ; False : moitié arrière."""
        amp = 0.10 * math.exp(-max(0.0, t - T_RELEASE) / 0.5) if t > T_RELEASE else 0.0
        bottom_y = 2.62 - amp
        n = 12
        strands = []
        for k in range(n):
            a = 2 * math.pi * k / n
            back = math.sin(a) > 0
            if back != (not front):
                continue
            rings = []
            for f in (0.0, 0.35, 0.7, 1.0):
                r = RIM_R * (1 - 0.5 * f)
                y = RIM_Y - f * (RIM_Y - bottom_y)
                rings.append(self.P(HOOP_X + r * math.cos(a), y, RIM_Z + r * math.sin(a)))
            strands.append(rings)
        net_col = (235, 235, 235)
        w = max(1, int(0.008 * self.S(RIM_Z)))
        for rings in strands:
            d.line(rings, fill=net_col, width=w)
        # cercle
        pts_arc = []
        for k in range(96):
            a = 2 * math.pi * k / 96
            Z = RIM_Z + RIM_R * math.sin(a)
            is_back = math.sin(a) > 0
            if is_back == (not front):
                pts_arc.append(self.P(HOOP_X + RIM_R * math.cos(a), RIM_Y, Z))
            else:
                if pts_arc:
                    pts_arc.append(None)
        segs, cur = [], []
        for p in pts_arc:
            if p is None:
                if len(cur) > 1:
                    segs.append(cur)
                cur = []
            else:
                cur.append(p)
        if len(cur) > 1:
            segs.append(cur)
        wr = max(2, int(0.028 * self.S(RIM_Z)))
        for seg in segs:
            d.line(seg, fill=(255, 112, 32), width=wr, joint="curve")

    def _draw_player(self, d, sk: dict):
        hip, sh, head = sk["hip"], sk["sh"], sk["head"]
        up = upv(sk["torso"])
        perp = np.array([up[1], -up[0]])          # perpendiculaire au tronc

        # jambe éloignée (couleurs assombries)
        self._leg(d, hip, sk["kneeL"], sk["ankleL"], far=True)
        self._arm(d, sh, sk["elbowL"], sk["wristL"], far=True)

        # tronc : maillot, short
        torso_poly = [
            tuple(np.array(self.P(*(hip + perp * 0.17), PLAYER_Z))),
            tuple(np.array(self.P(*(sh + perp * 0.19), PLAYER_Z))),
            tuple(np.array(self.P(*(sh - perp * 0.19), PLAYER_Z))),
            tuple(np.array(self.P(*(hip - perp * 0.17), PLAYER_Z))),
        ]
        d.polygon(torso_poly, fill=JERSEY)
        self._capsule(d, hip + up * 0.05, hip - up * 0.05, 0.17, 0.17, SHORTS)
        self._capsule(d, sh, head, 0.05, 0.05, SKIN)       # cou

        # tête
        hx_, hy_ = self.P(head[0], head[1], PLAYER_Z)
        r = 0.115 * self.sp
        d.ellipse([hx_ - r, hy_ - r, hx_ + r, hy_ + r], fill=SKIN)
        d.chord([hx_ - r, hy_ - r, hx_ + r, hy_ + r], 180, 360, fill=HAIR)
        ex, ey = self.P(head[0] + 0.06, head[1] + 0.01, PLAYER_Z)
        re = 0.014 * self.sp
        d.ellipse([ex - re, ey - re, ex + re, ey + re], fill=(40, 25, 20))

        # jambe proche, bras proche
        self._leg(d, hip, sk["kneeR"], sk["ankleR"], far=False)
        self._arm(d, sh, sk["elbowR"], sk["wristR"], far=False)

    def _leg(self, d, hip, knee, ankle, far: bool):
        skin = SKIN_DARK if far else SKIN
        shorts = SHORTS_DARK if far else SHORTS
        sock = (190, 190, 190) if far else SOCK
        shoe = (200, 200, 200) if far else SHOE
        self._capsule(d, hip, knee, 0.085, 0.07, skin)
        self._capsule(d, hip, hip + (knee - hip) * 0.75, 0.11, 0.10, shorts)
        self._capsule(d, knee, ankle, 0.07, 0.05, skin)
        self._capsule(d, ankle, ankle + (knee - ankle) * 0.25, 0.062, 0.062, sock)
        self._capsule(d, ankle, ankle + np.array([0.15, -0.02]), 0.06, 0.045, shoe)

    def _arm(self, d, sh, elbow, wrist, far: bool):
        skin = SKIN_DARK if far else SKIN
        sleeve = JERSEY_DARK if far else JERSEY
        self._capsule(d, sh, elbow, 0.052, 0.045, skin)
        self._capsule(d, sh, sh + (elbow - sh) * 0.5, 0.068, 0.062, sleeve)
        self._capsule(d, elbow, wrist, 0.047, 0.04, skin)
        hand = wrist + (wrist - elbow) / max(1e-6, np.hypot(*(wrist - elbow))) * 0.05
        self._capsule(d, hand, hand, 0.052, 0.052, skin)

    def _draw_ball(self, d, cx_, cy_, r, t: float):
        d.ellipse([cx_ - r, cy_ - r, cx_ + r, cy_ + r], fill=ORANGE, outline=(120, 50, 10))
        ang = 3.0 * (t - T_RELEASE) if t > T_RELEASE else 0.0
        lw = max(1, int(r * 0.12))
        for a in (ang, ang + math.pi / 2):
            ex, ey = r * math.cos(a), r * math.sin(a)
            d.line([(cx_ - ex, cy_ - ey), (cx_ + ex, cy_ + ey)], fill=(30, 20, 15), width=lw)
        d.arc([cx_ - r, cy_ - r, cx_ + r, cy_ + r], 60, 120, fill=(30, 20, 15), width=lw)

    # -- post-traitement : moyenne des sous-images, sous-échantillonnage, HUD
    def frame(self, t: float) -> np.ndarray:
        p = self.preset
        acc = np.zeros((self.H, self.W, 3), dtype=np.float32)
        for k in range(p.subs):
            tt = t + ((k + 0.5) / p.subs - 0.5) * (0.5 / FPS)  # obturateur de 180°
            acc += np.asarray(self.draw_scene(tt), dtype=np.float32)
        acc /= p.subs
        img = Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8), "RGB")
        if self.ss > 1:
            img = img.resize((p.width, p.height), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) * self.vignette[..., None]
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
        self._hud(out, t)
        return np.asarray(out, dtype=np.uint8)

    def _hud(self, img: Image.Image, t: float):
        d = ImageDraw.Draw(img)
        w, h = img.size
        m = int(h * 0.04)
        d.ellipse([m, m, m + h * 0.03, m + h * 0.03], fill=(230, 30, 30))
        d.text((m + h * 0.045, m - 1), "REC", font=self.font_hud, fill=(245, 245, 245))
        secs = t
        tc = f"{int(secs // 60):02d}:{int(secs % 60):02d}.{int((secs % 1) * 100):02d}"
        d.text((m, h - m), tc, font=self.font_small, fill=(230, 230, 230), anchor="lb")
        d.text((w - m, h - m), "KFT · suivi caméra", font=self.font_small,
               fill=(200, 200, 200), anchor="rb")


# ------------------------------------------------------------- sorties
class Mp4Sink:
    def __init__(self, path: Path, fps: int):
        import imageio.v2 as iio  # import local : erreur explicite si absent
        self.path = path
        self.writer = iio.get_writer(
            str(path), fps=fps, codec="libx264", pixelformat="yuv420p",
            macro_block_size=2,
            ffmpeg_params=["-crf", "20", "-preset", "medium", "-movflags", "+faststart"],
        )

    def write(self, frame: np.ndarray):
        self.writer.append_data(frame)

    def close(self):
        self.writer.close()


class GifSink:
    """Repli : GIF animé à 15 i/s, largeur max 640 px."""

    def __init__(self, path: Path, fps: int, out_w: int, out_h: int):
        self.path = path
        self.frames: list[Image.Image] = []
        self.size = (min(640, out_w), int(round(out_h * min(640, out_w) / out_w)))
        self.duration = int(round(2 * 1000 / fps))

    def write(self, frame: np.ndarray):
        self.frames.append(Image.fromarray(frame).resize(self.size, Image.BILINEAR))

    def close(self):
        if not self.frames:
            return
        first, rest = self.frames[0], self.frames[1::2]
        first.save(self.path, save_all=True, append_images=rest, duration=self.duration, loop=0,
                   optimize=True)
        self.frames = []


# ------------------------------------------------------------ pilotage
def render_video(anim: Animation, preset: Preset, out: Path, sink_kind: str,
                 budget: float | None, gif: bool = False) -> Path:
    renderer = SceneRenderer(preset, anim)
    n_frames = int(round(DURATION * FPS))
    if sink_kind == "mp4":
        sink = Mp4Sink(out, FPS)
    else:
        sink = GifSink(out, FPS, preset.width, preset.height)
    print(f"  Rendu {preset.name} : {preset.width}x{preset.height}, "
          f"supersampling x{preset.ss}, flou x{preset.subs}, {n_frames} images", flush=True)
    t_start = time.perf_counter()
    done = False
    try:
        for i in range(n_frames):
            frame = renderer.frame(i / FPS)
            sink.write(frame)
            elapsed = time.perf_counter() - t_start
            if budget is not None and i + 1 == 6:
                est = elapsed / (i + 1) * n_frames
                print(f"  Benchmark : ~{est:.0f} s estimées (budget {budget:.0f} s)", flush=True)
                if est > budget:
                    raise Downgrade(est)
            if i % 15 == 0 or i == n_frames - 1:
                eta = elapsed / (i + 1) * (n_frames - i - 1)
                print(f"  image {i + 1}/{n_frames}  ({elapsed:5.1f} s, reste ~{eta:4.0f} s)", flush=True)
        done = True
    finally:
        sink.close()
        if not done:
            out.unlink(missing_ok=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dunk de basket filmé par une caméra suiveuse (open-source, local).")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Fichier MP4 de sortie.")
    ap.add_argument("--quality", choices=["auto", "haute", "moyenne", "basse"], default="auto",
                    help="Qualité (auto = adaptée au CPU/RAM disponibles).")
    ap.add_argument("--budget", type=float, default=600.0,
                    help="Durée de rendu cible en secondes (mode auto uniquement).")
    ap.add_argument("--gif", action="store_true", help="Exporter aussi un GIF animé.")
    args = ap.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cpus = effective_cpu_count()
    ram = available_ram_gb()
    print(f"Matériel : {cpus} CPU utilisable(s), RAM disponible : "
          f"{'inconnue' if ram is None else f'{ram:.1f} Go'}", flush=True)

    anim = Animation()
    names = [p.name for p in PRESETS]
    adaptive = args.quality == "auto"
    idx = auto_preset_index(cpus, ram) if adaptive else names.index(args.quality)
    sink_kind = "mp4"
    out = args.out

    while True:
        preset = PRESETS[idx]
        try:
            target = out if sink_kind == "mp4" else out.with_suffix(".gif")
            budget = args.budget if (adaptive and idx < len(PRESETS) - 1) else None
            result = render_video(anim, preset, target, sink_kind, budget)
            print(f"Vidéo générée : {result}", flush=True)
            break
        except Downgrade as exc:
            print(f"  Trop lent (~{exc.args[0]:.0f} s) : passage à un preset plus léger.", flush=True)
            idx += 1
        except MemoryError:
            if idx >= len(PRESETS) - 1:
                print("Erreur : mémoire insuffisante même en basse qualité.", file=sys.stderr)
                return 1
            print("  Mémoire insuffisante : passage à un preset plus léger.", flush=True)
            idx += 1
        except (OSError, RuntimeError) as exc:
            if sink_kind == "mp4":
                print(f"  Encodage MP4 impossible ({exc}). Repli sur un GIF animé.", flush=True)
                sink_kind = "gif"
            else:
                print(f"Erreur : {exc}", file=sys.stderr)
                return 1

    if args.gif and sink_kind == "mp4":
        # GIF d'aperçu léger, rendu avec le même preset que la vidéo
        gif_path = out.with_suffix(".gif")
        render_video(anim, PRESETS[idx], gif_path, "gif", None)
        print(f"GIF généré : {gif_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
