import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random

# ==============================================================================
# 5 JAPANESE FLOWER SPECIES - BOTANICAL PALETTES & METADATA
# ==============================================================================
FLOWERS = [
    {
        "id": "sakura",
        "name": "Sakura (Somei-Yoshino Cherry Blossom)",
        "meaning": "Pure beauty, mono no aware & spring renewal",
        "primary_bgr": (242, 222, 255),
        "secondary_bgr": (175, 105, 245),
        "anther_bgr": (90, 225, 255),
        "glow_bgr": (225, 185, 255),
    },
    {
        "id": "kiku",
        "name": "Kiku (Imperial Chrysanthemum)",
        "meaning": "Longevity, nobility & autumn royal grace",
        "primary_bgr": (110, 240, 255),
        "secondary_bgr": (28, 155, 248),
        "anther_bgr": (75, 225, 255),
        "glow_bgr": (85, 220, 255),
    },
    {
        "id": "tsubaki",
        "name": "Tsubaki (Yabu-Tsubaki Red Camellia)",
        "meaning": "Steadfast love, devotion & noble elegance",
        "primary_bgr": (105, 70, 250),
        "secondary_bgr": (42, 18, 180),
        "anther_bgr": (55, 220, 255),
        "glow_bgr": (95, 75, 245),
    },
    {
        "id": "ajisai",
        "name": "Ajisai (Temari Hydrangea)",
        "meaning": "Heartfelt gratitude, harmony & summer rain",
        "primary_bgr": (252, 205, 155),
        "secondary_bgr": (240, 155, 210),
        "anther_bgr": (255, 240, 225),
        "glow_bgr": (248, 185, 165),
    },
    {
        "id": "higanbana",
        "name": "Higanbana (Red Spider Lily)",
        "meaning": "Passionate memories & autumn equinox",
        "primary_bgr": (65, 58, 255),
        "secondary_bgr": (24, 14, 185),
        "anther_bgr": (95, 205, 255),
        "glow_bgr": (60, 55, 255),
    }
]


# ==============================================================================
# TRUE BOTANICAL 3D PARAMETRIC PETAL SURFACE ENGINE
# ==============================================================================
def make_botanical_contour_tau(n_pts=51):
    lin = np.linspace(-1.0, 1.0, n_pts, dtype=np.float32)
    return np.sign(lin) * (np.abs(lin) ** 1.25)


TAU_BOUNDARY = make_botanical_contour_tau(51)
SHELL_SCALES = [0.85, 0.70, 0.55, 0.40, 0.25]
VEIN_U = np.linspace(0.05, 0.78, 10, dtype=np.float32)

# Precompute a 14-point organic notched 3D petal contour for falling petal particles
TAU_PARTICLE = np.linspace(-1.0, 1.0, 14, dtype=np.float32)


def eval_botanical_points_3d(
    u, w_norm, length, width, r0, z0, azimuth, open_angle,
    arch_curv=0.22, cup_depth=0.18,
    ruffle_amp=0.0, ruffle_freq=4.5, ruffle_phase=0.0,
    twist=0.0, rot_parent=None, offset_parent=None
):
    """
    Evaluates a 3D curved botanical petal surface:
    - open_angle: base hinge angle from the flower's +Z axis (0 = upright bud, pi/2 = flat open).
    - arch_curv: longitudinal curvature along u (positive = recurved outward, negative = incurved bowl).
    - cup_depth: smooth parabolic transverse cupping across the petal width w_norm.
    """
    theta_chord = open_angle + arch_curv * 0.55 * (u ** 1.15)
    theta_tan = open_angle + arch_curv * (u ** 1.05)

    r_spine = r0 + length * u * np.sin(theta_chord)
    z_spine = z0 + length * u * np.cos(theta_chord)

    cup = cup_depth * (width * 0.5) * (w_norm ** 2)

    if ruffle_amp > 0.0:
        ruffle = ruffle_amp * (width * 0.5) * (w_norm ** 2) * np.sin(ruffle_freq * math.pi * u + ruffle_phase)
    else:
        ruffle = 0.0

    dn = cup + ruffle
    r_3d = r_spine - dn * np.cos(theta_tan)
    z_3d = z_spine + dn * np.sin(theta_tan)
    y_local = w_norm * (width * 0.5)

    phi = azimuth + twist * u
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)

    x_3d = r_3d * cos_p - y_local * sin_p
    y_3d = r_3d * sin_p + y_local * cos_p
    pts = np.stack([x_3d, y_3d, z_3d], axis=-1).astype(np.float32)

    if rot_parent is not None:
        pts = pts @ rot_parent.T
    if offset_parent is not None:
        pts = pts + offset_parent
    return pts


def compute_petal_uw_contour(tau, p_len=0.45, p_wid=0.55, notch_depth=0.0, notch_width=0.11, belly_shift=0.20):
    abs_t = np.abs(tau)
    theta_p = (1.0 - abs_t) * (math.pi * 0.5)

    u_base = np.power(np.clip(np.sin(theta_p), 0.0, 1.0), p_len)
    if notch_depth > 0.0:
        notch = notch_depth * np.exp(-((tau / notch_width) ** 2))
        u = u_base * (1.0 - notch)
    else:
        u = u_base

    w_mag = np.power(np.clip(np.sin(math.pi * abs_t), 0.0, 1.0), p_wid)
    w_mag = w_mag * (1.0 + belly_shift * np.cos(math.pi * 0.5 * abs_t))
    w_norm = np.sign(tau) * w_mag
    max_w = float(np.max(np.abs(w_norm))) + 1e-6
    w_norm = w_norm / max_w
    return u.astype(np.float32), w_norm.astype(np.float32)


def build_botanical_petal_primitive(
    length, width, r0, z0, azimuth, open_angle,
    tip_col, mid_col, base_col, back_col, vein_col=None,
    arch_curv=0.22, cup_depth=0.18,
    p_len=0.44, p_wid=0.55, notch_depth=0.0, notch_width=0.11, belly_shift=0.20,
    ruffle_amp=0.0, ruffle_freq=4.5, ruffle_phase=0.0,
    twist=0.0, shininess=20.0, specular=0.35, translucency=0.42,
    ao_factor=1.0, layer_order=1, draw_veins=True, rot_parent=None, offset_parent=None
):
    u_outer, w_outer = compute_petal_uw_contour(
        TAU_BOUNDARY, p_len=p_len, p_wid=p_wid,
        notch_depth=notch_depth, notch_width=notch_width, belly_shift=belly_shift
    )

    surf_kwargs = dict(
        length=length, width=width, r0=r0, z0=z0, azimuth=azimuth, open_angle=open_angle,
        arch_curv=arch_curv, cup_depth=cup_depth,
        ruffle_amp=ruffle_amp, ruffle_freq=ruffle_freq, ruffle_phase=ruffle_phase,
        twist=twist, rot_parent=rot_parent, offset_parent=offset_parent
    )

    outer_pts = eval_botanical_points_3d(u_outer, w_outer, **surf_kwargs)

    shells_pts = []
    for sc in SHELL_SCALES:
        u_sh = u_outer * sc
        w_sh = w_outer * (sc ** 0.86)
        shells_pts.append(eval_botanical_points_3d(u_sh, w_sh, **surf_kwargs))

    ref_u = np.array([0.15, 0.55, 0.55, 0.88], dtype=np.float32)
    ref_w = np.array([0.00, -0.60, 0.60, 0.00], dtype=np.float32)
    ref_pts = eval_botanical_points_3d(ref_u, ref_w, **surf_kwargs)

    veins = []
    if draw_veins:
        for w_target in [-0.36, 0.0, 0.36]:
            vu = VEIN_U * (0.78 if abs(w_target) < 0.2 else 0.64)
            vw = np.sin(math.pi * 0.5 * (vu / 0.80)) * w_target
            veins.append(eval_botanical_points_3d(vu, vw, **surf_kwargs))

    return {
        'type': 'botanical_petal',
        'outer_pts': outer_pts,
        'shells_pts': shells_pts,
        'ref_pts': ref_pts,
        'veins': veins,
        'tip_col': tip_col,
        'mid_col': mid_col,
        'base_col': base_col,
        'back_col': back_col,
        'vein_col': vein_col,
        'shininess': shininess,
        'specular': specular,
        'translucency': translucency,
        'ao_factor': ao_factor,
        'layer_order': layer_order
    }


def lerp_bgr(c1, c2, t):
    return (
        c1[0] * (1.0 - t) + c2[0] * t,
        c1[1] * (1.0 - t) + c2[1] * t,
        c1[2] * (1.0 - t) + c2[2] * t
    )


def shade_color(col_bgr, k_diff, spec_add=0.0):
    return (
        min(255, max(0, int(col_bgr[0] * k_diff + spec_add))),
        min(255, max(0, int(col_bgr[1] * k_diff + spec_add))),
        min(255, max(0, int(col_bgr[2] * k_diff + spec_add)))
    )


def project_pts(pts_cam, focal_len, cx, cy):
    xs = (pts_cam[:, 0] * focal_len / pts_cam[:, 2] + cx).astype(np.int32)
    ys = (pts_cam[:, 1] * focal_len / pts_cam[:, 2] + cy).astype(np.int32)
    return np.stack([xs, ys], axis=1)


# ==============================================================================
# NATURAL 3D PARTICLE & VOLUMETRIC EFFECTS ENGINE
# ==============================================================================
PARTICLE_U_OUTER, PARTICLE_W_OUTER = compute_petal_uw_contour(
    TAU_PARTICLE, p_len=0.38, p_wid=0.50, notch_depth=0.10, notch_width=0.10, belly_shift=0.22
)
PARTICLE_U_INNER = PARTICLE_U_OUTER * 0.62
PARTICLE_W_INNER = PARTICLE_W_OUTER * 0.56


class PetalParticle3D:
    """Organic 3D curved falling petal with aerodynamic flutter, spin, and two-tone shading."""
    def __init__(self, origin_3d, tip_bgr, base_bgr):
        self.pos = np.array([
            origin_3d[0] + random.uniform(-45, 45),
            origin_3d[1] + random.uniform(-45, 45),
            origin_3d[2] + random.uniform(-35, 35)
        ], dtype=np.float32)

        speed = random.uniform(1.0, 2.8)
        angle = random.uniform(0, 2.0 * math.pi)
        self.vel = np.array([
            math.cos(angle) * speed,
            math.sin(angle) * speed - random.uniform(0.6, 1.8),
            random.uniform(-1.2, 1.2)
        ], dtype=np.float32)

        self.rot = np.array([
            random.uniform(0, 2.0 * math.pi),
            random.uniform(0, 2.0 * math.pi),
            random.uniform(0, 2.0 * math.pi)
        ], dtype=np.float32)
        self.rot_speed = np.array([
            random.uniform(-0.07, 0.07),
            random.uniform(-0.09, 0.09),
            random.uniform(-0.06, 0.06)
        ], dtype=np.float32)

        self.size = random.uniform(12.0, 20.0)
        self.life = 1.0
        self.decay = random.uniform(0.010, 0.020)
        self.tip_color = tip_bgr
        self.base_color = base_bgr
        self.phase = random.uniform(0, math.pi * 2.0)

        # Build local curved 3D petal mesh once per particle
        s = self.size
        x_out = (PARTICLE_U_OUTER - 0.45) * s
        y_out = PARTICLE_W_OUTER * (s * 0.46)
        z_out = ((PARTICLE_W_OUTER ** 2) * 0.22 + ((PARTICLE_U_OUTER - 0.5) ** 2) * 0.18) * s
        self.local_outer = np.stack([x_out, y_out, z_out], axis=-1).astype(np.float32)

        x_in = (PARTICLE_U_INNER - 0.45) * s
        y_in = PARTICLE_W_INNER * (s * 0.46)
        z_in = ((PARTICLE_W_INNER ** 2) * 0.22 + ((PARTICLE_U_INNER - 0.5) ** 2) * 0.18) * s
        self.local_inner = np.stack([x_in, y_in, z_in], axis=-1).astype(np.float32)

    def update(self):
        self.phase += 0.09
        # Aerodynamic flutter & gentle gravity
        self.vel[0] += math.sin(self.phase) * 0.28
        self.vel[1] += 0.32 + 0.12 * math.cos(self.phase * 1.3)
        self.vel[2] += math.cos(self.phase * 0.7) * 0.18
        self.vel *= 0.982

        self.pos += self.vel
        self.rot += self.rot_speed
        self.life -= self.decay

    def get_transformed_3d(self):
        cx, sx = math.cos(self.rot[0]), math.sin(self.rot[0])
        cy, sy = math.cos(self.rot[1]), math.sin(self.rot[1])
        cz, sz = math.cos(self.rot[2]), math.sin(self.rot[2])

        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        R = Rz @ Ry @ Rx

        scale = max(0.3, min(1.0, self.life * 1.25))
        outer_3d = (self.local_outer * scale) @ R.T + self.pos
        inner_3d = (self.local_inner * scale) @ R.T + self.pos
        normal_z = abs(float(R[2, 2]))
        return outer_3d, inner_3d, normal_z


class PollenParticle3D:
    """Luminous bokeh pollen spore rising in a gentle 3D spiral with soft firefly glow."""
    def __init__(self, origin_3d, color_bgr):
        self.pos = np.array(origin_3d, dtype=np.float32) + np.random.uniform(-22, 22, size=3).astype(np.float32)
        self.vel = np.array([
            random.uniform(-0.8, 0.8),
            random.uniform(-2.2, -0.7),
            random.uniform(-0.8, 0.8)
        ], dtype=np.float32)
        self.radius = random.uniform(2.2, 4.4)
        self.life = 1.0
        self.decay = random.uniform(0.016, 0.032)
        self.color = color_bgr
        self.spiral_ang = random.uniform(0, math.pi * 2.0)
        self.pulse_phase = random.uniform(0, math.pi * 2.0)

    def update(self):
        self.spiral_ang += 0.12
        self.pulse_phase += 0.22
        self.pos[0] += math.cos(self.spiral_ang) * 0.75 + self.vel[0]
        self.pos[1] += self.vel[1]
        self.pos[2] += math.sin(self.spiral_ang) * 0.75 + self.vel[2]
        self.life -= self.decay


class BloomRipple3D:
    """Harmonic 3D energy ripple expanding across the palm plane upon full bloom."""
    def __init__(self, center_3d, palm_R, color_bgr, delay_radius=0.0):
        self.center = np.array(center_3d, dtype=np.float32)
        self.R = palm_R.copy()
        self.radius = 12.0 - delay_radius
        self.max_radius = 240.0
        self.speed = 9.5
        self.life = 1.0
        self.color = color_bgr

    def update(self):
        self.radius += self.speed
        if self.radius > 0.0:
            self.life = max(0.0, 1.0 - (self.radius / self.max_radius) ** 0.85)


def render_volumetric_aura(frame, center_uv, base_radius, bloom, glow_bgr, now):
    """Renders a natural, soft Gaussian-blurred volumetric backlight behind the 3D flower."""
    if bloom < 0.04:
        return
    h, w, _ = frame.shape
    cx, cy = int(center_uv[0]), int(center_uv[1])
    pulse = 1.0 + 0.05 * math.sin(now * 3.2)
    aura_r = int(base_radius * (0.55 + 0.78 * bloom) * pulse)
    if aura_r < 12:
        return

    pad = int(aura_r * 1.45)
    x1, y1 = max(0, cx - pad), max(0, cy - pad)
    x2, y2 = min(w, cx + pad), min(h, cy + pad)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return

    roi = frame[y1:y2, x1:x2]
    glow_layer = np.zeros_like(roi)
    lcx, lcy = cx - x1, cy - y1

    cv2.circle(glow_layer, (lcx, lcy), aura_r, glow_bgr, -1)
    inner_col = tuple(min(255, int(c * 0.65 + 90)) for c in glow_bgr)
    cv2.circle(glow_layer, (lcx, lcy), int(aura_r * 0.58), inner_col, -1)
    cv2.circle(glow_layer, (lcx, lcy), int(aura_r * 0.28), (255, 255, 255), -1)

    sigma = max(5.0, aura_r * 0.36)
    glow_blurred = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=sigma, sigmaY=sigma)
    alpha = min(0.38, 0.10 + 0.26 * bloom)
    cv2.addWeighted(roi, 1.0, glow_blurred, alpha, 0, dst=roi)


# ==============================================================================
# 5 BOTANICAL 3D FLOWER GENERATORS
# ==============================================================================
def generate_sakura_botanical(radius, bloom, now=0.0):
    """Somei-Yoshino Cherry Blossom: 5 plump heart-notched petals, burgundy-green calyx & 32 curved stamens."""
    prims = []
    calyx_len = radius * 0.36
    for i in range(5):
        ang = i * (2.0 * math.pi / 5.0) + math.pi / 5.0
        prims.append(build_botanical_petal_primitive(
            length=calyx_len, width=calyx_len * 0.42,
            r0=radius * 0.04, z0=-radius * 0.04, azimuth=ang,
            open_angle=math.radians(98.0), arch_curv=0.18, cup_depth=0.10,
            p_len=0.68, p_wid=0.65, notch_depth=0.0, belly_shift=0.0,
            tip_col=(65, 132, 75), mid_col=(52, 95, 105),
            base_col=(45, 48, 125), back_col=(52, 105, 62),
            shininess=12.0, specular=0.15, translucency=0.15, ao_factor=0.88,
            layer_order=0, draw_veins=False
        ))

    open_outer = math.radians(20.0 + 56.0 * (bloom ** 0.95))
    len_outer = radius * (0.50 + 0.54 * bloom)
    wid_outer = len_outer * 0.96

    for i in range(5):
        breeze = 0.018 * math.sin(now * 2.4 + i * 1.3)
        ang = i * (2.0 * math.pi / 5.0) + breeze
        prims.append(build_botanical_petal_primitive(
            length=len_outer, width=wid_outer,
            r0=radius * 0.05, z0=radius * 0.01 * (i % 2), azimuth=ang,
            open_angle=open_outer + breeze,
            arch_curv=0.14 * bloom - 0.22 * (1.0 - bloom),
            cup_depth=0.20 * (1.0 - 0.30 * bloom),
            p_len=0.36, p_wid=0.46, notch_depth=0.14, notch_width=0.085, belly_shift=0.25,
            tip_col=(248, 240, 255), mid_col=(234, 205, 255),
            base_col=(165, 92, 242), back_col=(225, 190, 252),
            vein_col=(210, 160, 248),
            shininess=22.0, specular=0.38, translucency=0.48, ao_factor=1.0,
            layer_order=1, draw_veins=True
        ))

    prims.append({
        'type': 'sphere',
        'center': np.array([0.0, 0.0, radius * 0.03], dtype=np.float32),
        'radius': radius * 0.085,
        'color': (135, 62, 222),
        'highlight': False,
        'layer_order': 2
    })
    prims.append({
        'type': 'sphere',
        'center': np.array([0.0, 0.0, radius * 0.06], dtype=np.float32),
        'radius': radius * 0.038,
        'color': (95, 195, 145),
        'highlight': True,
        'layer_order': 3
    })

    if bloom > 0.08:
        st_bloom = (bloom - 0.08) / 0.92
        for s_i in range(32):
            ring = s_i % 2
            s_ang = s_i * (2.0 * math.pi / 32.0) + 0.018 * math.sin(now * 3.0 + s_i)
            s_len = radius * (0.28 if ring == 0 else 0.42) * (0.35 + 0.65 * st_bloom)
            s_spread = math.radians((22.0 if ring == 0 else 44.0) * st_bloom + 10.0)

            t_vals = np.linspace(0.0, 1.0, 6, dtype=np.float32)
            theta_t = s_spread * (t_vals ** 0.85) - 0.22 * (t_vals ** 2) * st_bloom
            r_t = radius * 0.04 + s_len * t_vals * np.sin(theta_t)
            z_t = radius * 0.04 + s_len * t_vals * np.cos(theta_t)
            pts_st = np.stack([r_t * np.cos(s_ang), r_t * np.sin(s_ang), z_t], axis=-1).astype(np.float32)

            prims.append({
                'type': 'curve',
                'pts': pts_st,
                'color': (235, 222, 255) if ring == 1 else (195, 142, 248),
                'thickness': 1,
                'layer_order': 3
            })
            prims.append({
                'type': 'sphere',
                'center': pts_st[-1],
                'radius': max(2.2, 3.4 * st_bloom),
                'color': (90, 225, 255) if ring == 1 else (65, 195, 252),
                'highlight': True,
                'layer_order': 4
            })

    return prims


def generate_kiku_botanical(radius, bloom, now=0.0):
    """Imperial Chrysanthemum: 79 spoon-tip ray florets across 5 golden-angle spiral whorls."""
    prims = []
    whorls = [
        (1, 22, 1.02, 0.26, 0.10, 0.00, 26.0, 82.0,  0.22, 0.24, (175, 252, 255), (85, 232, 255), (30, 162, 250), 0.95),
        (2, 18, 0.84, 0.27, 0.08, 0.03, 20.0, 68.0,  0.10, 0.26, (162, 250, 255), (75, 224, 255), (26, 152, 246), 0.97),
        (3, 16, 0.67, 0.28, 0.06, 0.06, 15.0, 54.0, -0.06, 0.28, (148, 246, 255), (64, 214, 254), (22, 140, 242), 0.98),
        (4, 13, 0.50, 0.29, 0.05, 0.09, 10.0, 38.0, -0.22, 0.30, (134, 242, 255), (52, 204, 250), (18, 128, 236), 0.99),
        (5, 10, 0.34, 0.31, 0.04, 0.12,  6.0, 24.0, -0.35, 0.32, (120, 238, 255), (42, 192, 246), (15, 118, 230), 1.00),
    ]

    for lay_ord, count, l_sc, w_sc, r0_sc, z0_sc, min_a, max_a, arch, cup, t_col, m_col, b_col, ao in whorls:
        delay = (lay_ord - 1) * 0.05
        w_bloom = max(0.0, min(1.0, (bloom - delay * 0.4) / (1.0 - delay * 0.4)))
        open_ang = math.radians(min_a + (max_a - min_a) * (w_bloom ** 1.05))
        p_len = radius * l_sc * (0.45 + 0.55 * w_bloom)
        p_wid = p_len * w_sc
        phase = lay_ord * 0.618034 * math.pi

        for k in range(count):
            breeze = 0.018 * math.sin(now * 2.2 + k * 0.9 + lay_ord)
            az = phase + k * (2.0 * math.pi / count) + breeze
            prims.append(build_botanical_petal_primitive(
                length=p_len, width=p_wid,
                r0=radius * r0_sc, z0=radius * z0_sc,
                azimuth=az, open_angle=open_ang,
                arch_curv=arch * (0.4 + 0.6 * w_bloom),
                cup_depth=cup, p_len=0.44, p_wid=0.58, notch_depth=0.0, belly_shift=0.22,
                tip_col=t_col, mid_col=m_col, base_col=b_col,
                back_col=(45, 185, 248),
                shininess=22.0, specular=0.32, translucency=0.40,
                ao_factor=ao, layer_order=lay_ord, draw_veins=False
            ))

    prims.append({
        'type': 'sphere',
        'center': np.array([0.0, 0.0, radius * 0.14], dtype=np.float32),
        'radius': radius * 0.095,
        'color': (35, 165, 220),
        'highlight': True,
        'layer_order': 6
    })
    for b_i in range(10):
        ba = b_i * (2.0 * math.pi / 10.0)
        br = radius * 0.052
        prims.append({
            'type': 'sphere',
            'center': np.array([br * math.cos(ba), br * math.sin(ba), radius * 0.18], dtype=np.float32),
            'radius': radius * 0.028,
            'color': (60, 210, 250),
            'highlight': True,
            'layer_order': 7
        })

    return prims


def generate_tsubaki_botanical(radius, bloom, now=0.0):
    """Japanese Camellia: 4 spiral imbricated crimson whorls, evergreen leaves & Chasen stamen cylinder."""
    prims = []
    for i in range(5):
        ang = i * (2.0 * math.pi / 5.0) + 0.3
        prims.append(build_botanical_petal_primitive(
            length=radius * 0.56, width=radius * 0.38,
            r0=radius * 0.10, z0=-radius * 0.05, azimuth=ang,
            open_angle=math.radians(92.0), arch_curv=0.18, cup_depth=0.12,
            p_len=0.58, p_wid=0.60, notch_depth=0.0, belly_shift=0.10,
            tip_col=(48, 118, 42), mid_col=(36, 92, 32),
            base_col=(24, 62, 20), back_col=(32, 80, 26),
            shininess=30.0, specular=0.45, translucency=0.15, ao_factor=0.88,
            layer_order=0, draw_veins=True
        ))

    whorls = [
        (1, 8, 0.98, 0.88, 0.11, 0.00, 28.0, 78.0,  0.18, 0.22, (98,  65, 246), (68, 36, 224), (36, 14, 165), 0.95),
        (2, 7, 0.82, 0.90, 0.09, 0.03, 22.0, 64.0,  0.10, 0.25, (105, 70, 250), (74, 40, 232), (40, 16, 174), 0.97),
        (3, 6, 0.66, 0.92, 0.07, 0.06, 16.0, 50.0,  0.00, 0.28, (112, 76, 253), (80, 46, 238), (44, 20, 182), 0.99),
        (4, 5, 0.48, 0.94, 0.05, 0.09, 12.0, 36.0, -0.12, 0.30, (120, 84, 255), (86, 52, 244), (48, 24, 190), 1.00),
    ]

    for lay_ord, count, l_sc, w_rat, r0_sc, z0_sc, min_a, max_a, arch, cup, t_col, m_col, b_col, ao in whorls:
        w_bloom = max(0.0, min(1.0, (bloom - (lay_ord - 1) * 0.04) / (1.0 - (lay_ord - 1) * 0.04)))
        open_ang = math.radians(min_a + (max_a - min_a) * (w_bloom ** 1.05))
        p_len = radius * l_sc * (0.45 + 0.55 * w_bloom)
        p_wid = p_len * w_rat
        phase = lay_ord * 0.54

        for k in range(count):
            breeze = 0.015 * math.sin(now * 2.0 + k + lay_ord)
            az = phase + k * (2.0 * math.pi / count) + breeze
            prims.append(build_botanical_petal_primitive(
                length=p_len, width=p_wid,
                r0=radius * r0_sc, z0=radius * z0_sc,
                azimuth=az, open_angle=open_ang,
                arch_curv=arch * w_bloom, cup_depth=cup,
                p_len=0.40, p_wid=0.52, notch_depth=0.04, notch_width=0.14, belly_shift=0.24,
                tip_col=t_col, mid_col=m_col, base_col=b_col,
                back_col=(55, 28, 195), vein_col=(62, 32, 205),
                shininess=32.0, specular=0.52, translucency=0.35,
                ao_factor=ao, layer_order=lay_ord, draw_veins=True
            ))

    cyl_r = radius * 0.13 * (0.45 + 0.55 * bloom)
    cyl_h = radius * 0.34 * (0.45 + 0.55 * bloom)
    prims.append({
        'type': 'sphere',
        'center': np.array([0.0, 0.0, radius * 0.10], dtype=np.float32),
        'radius': cyl_r * 0.85,
        'color': (175, 228, 250),
        'highlight': True,
        'layer_order': 5
    })

    for s_i in range(30):
        sa = s_i * (2.0 * math.pi / 30.0)
        r_base = cyl_r * 0.72
        r_top = cyl_r * (0.92 + 0.12 * (s_i % 2))
        h_top = cyl_h * (0.90 + 0.12 * (s_i % 2))

        pts_st = np.array([
            [r_base * math.cos(sa), r_base * math.sin(sa), radius * 0.08],
            [r_top * 0.9 * math.cos(sa), r_top * 0.9 * math.sin(sa), h_top * 0.58],
            [r_top * math.cos(sa), r_top * math.sin(sa), h_top]
        ], dtype=np.float32)

        prims.append({
            'type': 'curve',
            'pts': pts_st,
            'color': (215, 242, 255),
            'thickness': 2,
            'layer_order': 5
        })
        prims.append({
            'type': 'sphere',
            'center': pts_st[-1],
            'radius': max(2.2, 3.6 * bloom),
            'color': (40, 218, 255),
            'highlight': True,
            'layer_order': 6
        })

    return prims


def generate_ajisai_botanical(radius, bloom, now=0.0):
    """Temari Hydrangea: Lush spherical dome of 19 4-sepal florets with framing green leaves."""
    prims = []
    for i in range(4):
        ang = i * (math.pi / 2.0) + math.pi / 4.0
        prims.append(build_botanical_petal_primitive(
            length=radius * 0.62, width=radius * 0.44,
            r0=radius * 0.14, z0=-radius * 0.06, azimuth=ang,
            open_angle=math.radians(94.0), arch_curv=0.22, cup_depth=0.10,
            p_len=0.55, p_wid=0.58, notch_depth=0.0, belly_shift=0.14,
            tip_col=(62, 145, 58), mid_col=(48, 118, 44),
            base_col=(34, 82, 30), back_col=(42, 102, 38),
            shininess=18.0, specular=0.28, translucency=0.20, ao_factor=0.88,
            layer_order=0, draw_veins=True
        ))

    dome_r = radius * (0.28 + 0.48 * bloom)

    palettes = [
        ((255, 226, 165), (248, 192, 128), (225, 148, 98), (235, 175, 130), (238, 175, 115)),
        ((252, 210, 200), (242, 172, 172), (212, 125, 142), (225, 152, 162), (225, 148, 155)),
        ((248, 202, 236), (236, 158, 212), (202, 114, 178), (220, 142, 196), (218, 135, 192)),
        ((255, 246, 228), (246, 222, 184), (222, 180, 138), (235, 208, 168), (232, 198, 156)),
    ]

    floret_specs = []
    if bloom > 0.12:
        for i in range(12):
            floret_specs.append((1, math.radians(56.0), i * (2.0 * math.pi / 12.0) + 0.26, 0.35, palettes[(i + 1) % 3], i + 7))
    for i in range(6):
        floret_specs.append((2, math.radians(30.0), i * (2.0 * math.pi / 6.0), 0.38, palettes[i % 3], i + 1))
    floret_specs.append((3, 0.0, 0.0, 0.40, palettes[3], 0))

    for lay_ord, elev, az, sc, pal, f_idx in floret_specs:
        nx = math.sin(elev) * math.cos(az)
        ny = math.sin(elev) * math.sin(az)
        nz = math.cos(elev)
        norm = np.array([nx, ny, nz], dtype=np.float32)
        center = norm * dome_r

        up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(norm, up_ref))) > 0.9:
            up_ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        tx = np.cross(up_ref, norm)
        tx /= np.linalg.norm(tx)
        ty = np.cross(norm, tx)
        rot_floret = np.column_stack([tx, ty, norm])

        f_bloom = max(0.0, min(1.0, (bloom - (elev / 1.8) * 0.10) / 0.90))
        f_len = radius * sc * (0.45 + 0.55 * f_bloom)
        f_wid = f_len * 0.92
        f_open = math.radians(32.0 + 46.0 * (f_bloom ** 1.02))
        t_col, m_col, b_col, bk_col, v_col = pal

        for k in range(4):
            ang_k = k * (math.pi / 2.0) + 0.18 * (f_idx % 3) + 0.02 * math.sin(now * 2.0 + f_idx)
            prims.append(build_botanical_petal_primitive(
                length=f_len, width=f_wid,
                r0=f_len * 0.05, z0=0.0, azimuth=ang_k,
                open_angle=f_open, arch_curv=0.14 * f_bloom,
                cup_depth=0.16, p_len=0.40, p_wid=0.52, notch_depth=0.0, belly_shift=0.24,
                tip_col=t_col, mid_col=m_col, base_col=b_col,
                back_col=bk_col, vein_col=v_col,
                shininess=24.0, specular=0.42, translucency=0.45,
                ao_factor=0.95 if elev > 0.6 else 1.0,
                layer_order=lay_ord * 2, draw_veins=True,
                rot_parent=rot_floret, offset_parent=center
            ))

        prims.append({
            'type': 'sphere',
            'center': center + norm * (f_len * 0.12),
            'radius': max(2.4, 4.0 * f_bloom),
            'color': (255, 242, 228),
            'highlight': True,
            'layer_order': lay_ord * 2 + 1
        })

    return prims


def generate_higanbana_botanical(radius, bloom, now=0.0):
    """Red Spider Lily (Lycoris radiata): 24 reflexed wavy crimson tepals & 36 sweeping stamens."""
    prims = []
    num_tepals = 24
    t_len = radius * 0.86 * (0.35 + 0.65 * bloom)
    t_wid = t_len * 0.19

    for i in range(num_tepals):
        ring = i % 2
        breeze = 0.03 * math.sin(now * 2.8 + i * 0.7)
        az = i * (2.0 * math.pi / num_tepals) + breeze
        open_ang = math.radians((24.0 if ring == 0 else 42.0) + (40.0 if ring == 0 else 50.0) * bloom)
        arch_c = (0.52 if ring == 0 else 0.74) * bloom

        prims.append(build_botanical_petal_primitive(
            length=t_len * (0.90 if ring == 0 else 1.04),
            width=t_wid,
            r0=radius * 0.05, z0=radius * (0.06 if ring == 0 else 0.0),
            azimuth=az, open_angle=open_ang,
            arch_curv=arch_c, cup_depth=-0.14,
            p_len=0.65, p_wid=0.65, notch_depth=0.0, belly_shift=0.08,
            ruffle_amp=0.16 * bloom, ruffle_freq=5.0, ruffle_phase=i * 0.8,
            twist=0.12 * math.sin(i),
            tip_col=(68, 62, 255), mid_col=(40, 28, 244),
            base_col=(22, 12, 180), back_col=(32, 20, 215),
            vein_col=(25, 14, 165),
            shininess=28.0, specular=0.48, translucency=0.42,
            ao_factor=0.98, layer_order=1 if ring == 1 else 2, draw_veins=True
        ))

    prims.append({
        'type': 'sphere',
        'center': np.array([0.0, 0.0, radius * 0.06], dtype=np.float32),
        'radius': radius * 0.075,
        'color': (32, 20, 205),
        'highlight': False,
        'layer_order': 3
    })

    st_reach = radius * 1.45 * (0.28 + 0.72 * bloom)
    for s_i in range(36):
        ring = s_i % 3
        s_ang = s_i * (2.0 * math.pi / 36.0) + 0.025 * math.cos(now * 2.5 + s_i)
        reach_i = st_reach * (0.82 + 0.10 * ring)
        elev_bias = 0.26 + 0.14 * ring

        u_st = np.linspace(0.0, 1.0, 9, dtype=np.float32)
        r_st = radius * 0.04 + reach_i * (u_st ** 0.88)
        z_st = radius * 0.05 + reach_i * (
            elev_bias * np.sin(math.pi * u_st * 0.85)
            - 0.18 * (u_st ** 2)
            + 0.42 * (u_st ** 3.2)
        ) * bloom
        sway_ang = s_ang + 0.08 * np.sin(u_st * math.pi) * math.cos(s_i)

        pts_st = np.stack([r_st * np.cos(sway_ang), r_st * np.sin(sway_ang), z_st], axis=-1).astype(np.float32)

        prims.append({
            'type': 'curve',
            'pts': pts_st[:5],
            'color': (35, 26, 228),
            'thickness': 2 if bloom > 0.5 else 1,
            'layer_order': 4
        })
        prims.append({
            'type': 'curve',
            'pts': pts_st[4:],
            'color': (72, 98, 255),
            'thickness': 1,
            'layer_order': 5
        })
        if bloom > 0.18:
            prims.append({
                'type': 'sphere',
                'center': pts_st[-1],
                'radius': max(2.2, 3.3 * bloom),
                'color': (95, 205, 255),
                'highlight': True,
                'layer_order': 6
            })

    return prims


# ==============================================================================
# 3D BOTANICAL SCENE RASTERIZER (HEMISPHERICAL SHADING + DEPTH SORTING)
# ==============================================================================
def render_botanical_scene(frame, primitives, camera_pos_3d, palm_R, focal_len, light_dir):
    """Transforms 3D botanical primitives to camera space, applies translucent shading & renders."""
    h, w, _ = frame.shape
    cx, cy = w / 2.0, h / 2.0
    drawn_items = []
    view_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    half_vec = light_dir + view_vec
    half_vec /= np.linalg.norm(half_vec)

    for prim in primitives:
        p_type = prim['type']
        layer_order = prim.get('layer_order', 5)
        layer_z_bias = layer_order * 7.5

        if p_type == 'botanical_petal':
            ref_cam = (prim['ref_pts'] @ palm_R.T) + camera_pos_3d
            if np.any(ref_cam[:, 2] <= 50.0):
                continue

            z_depth = float(0.30 * ref_cam[0, 2] + 0.25 * ref_cam[1, 2] + 0.25 * ref_cam[2, 2] + 0.20 * ref_cam[3, 2])
            z_sort = z_depth - layer_z_bias

            v_long = ref_cam[3] - ref_cam[0]
            v_lat = ref_cam[2] - ref_cam[1]
            normal = np.cross(v_long, v_lat)
            n_norm = np.linalg.norm(normal)
            if n_norm < 1e-6:
                normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            else:
                normal /= n_norm

            facing = float(np.dot(normal, view_vec))
            eff_normal = normal if facing >= 0.0 else -normal

            ao = prim.get('ao_factor', 1.0)
            diffuse = max(0.0, float(np.dot(eff_normal, light_dir)))
            transl = max(0.0, float(np.dot(-eff_normal, light_dir))) * prim.get('translucency', 0.45)
            cam_fill = max(0.0, float(np.dot(eff_normal, view_vec)))
            k_diff = min(1.20, (0.64 + 0.24 * diffuse + 0.16 * cam_fill + transl * 0.35) * ao)

            spec_dot = max(0.0, float(np.dot(eff_normal, half_vec)))
            spec_val = (spec_dot ** prim.get('shininess', 20.0)) * prim.get('specular', 0.35) * 145.0

            outer_cam = (prim['outer_pts'] @ palm_R.T) + camera_pos_3d
            outer_2d = project_pts(outer_cam, focal_len, cx, cy)

            shells_2d = []
            for s_3d in prim['shells_pts']:
                sc_cam = (s_3d @ palm_R.T) + camera_pos_3d
                shells_2d.append(project_pts(sc_cam, focal_len, cx, cy))

            veins_2d = []
            for v_3d in prim.get('veins', []):
                vc = (v_3d @ palm_R.T) + camera_pos_3d
                veins_2d.append(project_pts(vc, focal_len, cx, cy))

            tip_c = prim['tip_col']
            mid_c = prim['mid_col']
            base_c = prim['base_col']

            c_outer = shade_color(tip_c, k_diff, spec_val * 0.32)
            shell_colors = [
                shade_color(lerp_bgr(tip_c, mid_c, 0.28), k_diff * 0.99, spec_val * 0.34),
                shade_color(lerp_bgr(tip_c, mid_c, 0.58), k_diff * 0.98, spec_val * 0.22),
                shade_color(lerp_bgr(tip_c, mid_c, 0.88), k_diff * 0.96, spec_val * 0.12),
                shade_color(lerp_bgr(mid_c, base_c, 0.42), k_diff * 0.93, spec_val * 0.04),
                shade_color(lerp_bgr(mid_c, base_c, 0.82), k_diff * 0.90, 0.0),
            ]
            c_rim = shade_color(lerp_bgr(tip_c, mid_c, 0.35), k_diff * 0.93, spec_val * 0.25)
            v_raw = prim['vein_col'] if prim['vein_col'] else lerp_bgr(mid_c, base_c, 0.35)
            v_col = shade_color(v_raw, k_diff * 0.94, 0.0)

            drawn_items.append((
                -z_sort, 'botanical_petal',
                outer_2d, shells_2d, veins_2d,
                c_outer, shell_colors, c_rim, v_col
            ))

        elif p_type == 'curve':
            pts_cam = (prim['pts'] @ palm_R.T) + camera_pos_3d
            if np.any(pts_cam[:, 2] <= 50.0):
                continue
            z_depth = float(np.mean(pts_cam[:, 2]))
            z_sort = z_depth - layer_z_bias
            pts_2d = project_pts(pts_cam, focal_len, cx, cy)
            drawn_items.append((-z_sort, 'curve', pts_2d, prim['color'], prim['thickness']))

        elif p_type == 'sphere':
            c_cam = (prim['center'] @ palm_R.T) + camera_pos_3d
            if c_cam[2] <= 50.0:
                continue
            z_depth = float(c_cam[2])
            z_sort = z_depth - layer_z_bias
            sx = int(c_cam[0] * focal_len / c_cam[2] + cx)
            sy = int(c_cam[1] * focal_len / c_cam[2] + cy)
            sr = max(1, int(prim['radius'] * focal_len / c_cam[2]))
            drawn_items.append((-z_sort, 'sphere', (sx, sy), sr, prim['color'], prim.get('highlight', True)))

    drawn_items.sort(key=lambda item: item[0])

    for item in drawn_items:
        itype = item[1]
        if itype == 'botanical_petal':
            _, _, outer_2d, shells_2d, veins_2d, c_outer, shell_colors, c_rim, v_col = item
            cv2.fillPoly(frame, [outer_2d], c_outer, cv2.LINE_AA)
            for sh_2d, sh_col in zip(shells_2d, shell_colors):
                cv2.fillPoly(frame, [sh_2d], sh_col, cv2.LINE_AA)
            for v2d in veins_2d:
                cv2.polylines(frame, [v2d], False, v_col, 1, cv2.LINE_AA)
            cv2.polylines(frame, [outer_2d], True, c_rim, 1, cv2.LINE_AA)

        elif itype == 'curve':
            _, _, pts_2d, col, thick = item
            cv2.polylines(frame, [pts_2d], False, col, thick, cv2.LINE_AA)

        elif itype == 'sphere':
            _, _, (sx, sy), sr, col, hi = item
            cv2.circle(frame, (sx, sy), sr, col, -1, cv2.LINE_AA)
            if hi and sr >= 2:
                hi_col = tuple(min(255, int(c * 1.30 + 45)) for c in col)
                cv2.circle(frame, (sx - max(1, sr // 3), sy - max(1, sr // 3)), max(1, sr // 2), hi_col, -1, cv2.LINE_AA)


# ==============================================================================
# PRECISE 3D PALM CENTER & CAMERA-FACING ORIENTATION TRACKING
# ==============================================================================
def compute_hand_pose_3d(landmarks, w, h, focal_len):
    """
    Computes the anatomical palm hollow center pinned dead-center on the palm,
    and constructs a right-handed 3D orthonormal basis where +Z_local points
    OUT of the palm toward the camera (-Z_cam) so flowers bloom toward the viewer.
    """
    palm_u = (
        landmarks[0].x * 0.30
        + landmarks[5].x * 0.20
        + landmarks[9].x * 0.25
        + landmarks[13].x * 0.15
        + landmarks[17].x * 0.10
    ) * w
    palm_v = (landmarks[0].y * 0.45 + landmarks[9].y * 0.55) * h

    dx_px = (landmarks[9].x - landmarks[0].x) * w
    dy_px = (landmarks[9].y - landmarks[0].y) * h
    palm_span_px = max(40.0, math.hypot(dx_px, dy_px))

    z_estimate = (focal_len * 95.0) / palm_span_px

    cx, cy = w / 2.0, h / 2.0
    cam_x = (palm_u - cx) * z_estimate / focal_len
    cam_y = (palm_v - cy) * z_estimate / focal_len
    palm_center_3d = np.array([cam_x, cam_y, z_estimate], dtype=np.float32)

    pts = np.array([[(lm.x - 0.5) * w, (lm.y - 0.5) * h, lm.z * w] for lm in landmarks], dtype=np.float32)

    v_up = pts[9] - pts[0]
    v_up[2] *= 1.35
    v_up /= max(1e-5, np.linalg.norm(v_up))

    v_across = pts[5] - pts[17]
    v_across[2] *= 1.35
    v_across /= max(1e-5, np.linalg.norm(v_across))

    normal = np.cross(v_across, v_up)
    normal /= max(1e-5, np.linalg.norm(normal))

    # In OpenCV camera space (+Z into screen), outward palm normal toward camera has normal[2] < 0
    if normal[2] > 0:
        normal = -normal

    # Local +Z axis points out of the palm toward the camera (-Z_cam)
    z_axis = normal
    x_axis = np.cross(v_up, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm > 1e-5:
        x_axis /= x_norm
    else:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(1e-5, np.linalg.norm(y_axis))

    R_palm = np.column_stack([x_axis, y_axis, z_axis])
    return palm_center_3d, R_palm, palm_span_px, (palm_u, palm_v)


def compute_hand_openness(landmarks):
    """Computes hand openness ratio normalized from 0.0 (closed bud) to 1.0 (full bloom)."""
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    palm_cx = (wrist.x + middle_mcp.x) * 0.5
    palm_cy = (wrist.y + middle_mcp.y) * 0.5

    hand_scale = math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y)
    if hand_scale < 1e-5:
        return 0.0

    fingertip_ids = [4, 8, 12, 16, 20]
    dists = [math.hypot(landmarks[t].x - palm_cx, landmarks[t].y - palm_cy) / hand_scale for t in fingertip_ids]
    avg_ratio = sum(dists) / len(dists)
    normalized = (avg_ratio - 0.70) / (1.70 - 0.70)
    return max(0.0, min(1.0, normalized))


# ==============================================================================
# MAIN REAL-TIME AR LOOP
# ==============================================================================
def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Cannot open camera! Please grant Camera permissions in macOS System Settings > Privacy & Security > Camera.")
        return

    W, H = 1280, 720
    focal_len = float(W) * 0.95
    light_dir = np.array([0.22, -0.35, -0.91], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)

    flower_idx = 0
    smooth_bloom = 0.0
    smooth_center_3d = np.array([0.0, 0.0, 650.0], dtype=np.float32)
    smooth_R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    smooth_palm_uv = (640.0, 360.0)
    rot_angle = 0.0

    index_history = []
    last_swipe_time = 0.0
    swipe_cooldown = 0.75
    swipe_banner_text = ""
    swipe_banner_until = 0.0

    petal_particles = []
    pollen_particles = []
    ripples = []
    was_full_bloom = False

    print("=======================================================")
    print(" Interactive 3D Japanese Botanical Flowers Ready!")
    print(" - Pinch hand into a fist to close bud")
    print(" - Open hand wide to bloom in 3D")
    print(" - Swipe index finger left/right to change flower species")
    print(" - Tilt hand to rotate flower in 3D space")
    print(" - Press 'q' or ESC to exit")
    print("=======================================================")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Warning: Cannot receive video frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        focal_len = float(w) * 0.95

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        now = time.time()
        rot_angle += 0.009
        current_flower = FLOWERS[flower_idx]

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0].landmark

            # 1. Compute 3D palm center and right-handed camera-facing orientation basis
            target_center_3d, target_R, palm_span_px, (raw_u, raw_v) = compute_hand_pose_3d(hand_lms, w, h, focal_len)
            smooth_center_3d = 0.70 * smooth_center_3d + 0.30 * target_center_3d
            smooth_palm_uv = (0.70 * smooth_palm_uv[0] + 0.30 * raw_u, 0.70 * smooth_palm_uv[1] + 0.30 * raw_v)

            smooth_R = 0.74 * smooth_R + 0.26 * target_R
            u_svd, _, vt_svd = np.linalg.svd(smooth_R)
            smooth_R = u_svd @ vt_svd
            if np.linalg.det(smooth_R) < 0:
                u_svd[:, -1] *= -1
                smooth_R = u_svd @ vt_svd

            # 2. Compute 3D bloom degree
            raw_bloom = compute_hand_openness(hand_lms)
            smooth_bloom = 0.80 * smooth_bloom + 0.20 * raw_bloom

            base_flower_radius = max(58.0, min(175.0, palm_span_px * 0.96))

            # 3. Full-bloom dual harmonic ripples & pollen burst
            if smooth_bloom > 0.86 and not was_full_bloom:
                ripples.append(BloomRipple3D(smooth_center_3d, smooth_R, current_flower["glow_bgr"], delay_radius=0.0))
                ripples.append(BloomRipple3D(smooth_center_3d, smooth_R, current_flower["primary_bgr"], delay_radius=35.0))
                was_full_bloom = True
                for _ in range(14):
                    pollen_particles.append(PollenParticle3D(smooth_center_3d, current_flower["anther_bgr"]))
            elif smooth_bloom < 0.68:
                was_full_bloom = False

            # Ambient rising bokeh pollen spores
            if smooth_bloom > 0.55 and random.random() < 0.42:
                pollen_particles.append(PollenParticle3D(smooth_center_3d, current_flower["anther_bgr"]))

            # Gently drifting 3D curved petals
            if smooth_bloom > 0.72 and random.random() < 0.25:
                petal_particles.append(
                    PetalParticle3D(smooth_center_3d, current_flower["primary_bgr"], current_flower["secondary_bgr"])
                )

            # 4. Swipe gesture detection with silky glowing ribbon trail
            ix, iy = hand_lms[8].x * w, hand_lms[8].y * h
            index_history.append((now, ix, iy))
            index_history = [(t, x, y) for (t, x, y) in index_history if now - t <= 0.22]

            if len(index_history) > 2:
                for k in range(1, len(index_history)):
                    pt1 = (int(index_history[k - 1][1]), int(index_history[k - 1][2]))
                    pt2 = (int(index_history[k][1]), int(index_history[k][2]))
                    thickness = max(1, (k * 4) // len(index_history))
                    cv2.line(frame, pt1, pt2, current_flower["glow_bgr"], thickness + 2, cv2.LINE_AA)
                    cv2.line(frame, pt1, pt2, (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)

            if len(index_history) >= 4 and (now - last_swipe_time) > swipe_cooldown:
                dt = index_history[-1][0] - index_history[0][0]
                dx = index_history[-1][1] - index_history[0][1]
                dy = index_history[-1][2] - index_history[0][2]

                if dt > 0.05 and abs(dx) > w * 0.15 and abs(dx) > abs(dy) * 1.6:
                    if dx > 0:
                        flower_idx = (flower_idx + 1) % len(FLOWERS)
                        swipe_banner_text = f">> Swiped Right: {FLOWERS[flower_idx]['name']}"
                    else:
                        flower_idx = (flower_idx - 1) % len(FLOWERS)
                        swipe_banner_text = f"<< Swiped Left: {FLOWERS[flower_idx]['name']}"

                    last_swipe_time = now
                    swipe_banner_until = now + 1.6
                    index_history.clear()
                    current_flower = FLOWERS[flower_idx]

                    ripples.append(BloomRipple3D(smooth_center_3d, smooth_R, current_flower["glow_bgr"], delay_radius=0.0))
                    for _ in range(14):
                        petal_particles.append(
                            PetalParticle3D(smooth_center_3d, current_flower["primary_bgr"], current_flower["secondary_bgr"])
                        )

            # 5. Combined 3D hand rotation with gentle botanical spin around local +Z
            cos_spin, sin_spin = math.cos(rot_angle), math.sin(rot_angle)
            R_spin = np.array([
                [cos_spin, -sin_spin, 0.0],
                [sin_spin,  cos_spin, 0.0],
                [0.0,       0.0,      1.0]
            ], dtype=np.float32)
            combined_R = smooth_R @ R_spin

            # 6. Soft Gaussian volumetric aura glow behind the flower
            render_volumetric_aura(
                frame, smooth_palm_uv, base_flower_radius, smooth_bloom, current_flower["glow_bgr"], now
            )

            # 7. Generate high-resolution 3D botanical flower primitives
            fid = current_flower["id"]
            if fid == "sakura":
                prims = generate_sakura_botanical(base_flower_radius, smooth_bloom, now=now)
            elif fid == "kiku":
                prims = generate_kiku_botanical(base_flower_radius, smooth_bloom, now=now)
            elif fid == "tsubaki":
                prims = generate_tsubaki_botanical(base_flower_radius, smooth_bloom, now=now)
            elif fid == "ajisai":
                prims = generate_ajisai_botanical(base_flower_radius, smooth_bloom, now=now)
            elif fid == "higanbana":
                prims = generate_higanbana_botanical(base_flower_radius, smooth_bloom, now=now)
            else:
                prims = []

            # 8. Render 3D botanical scene
            render_botanical_scene(frame, prims, smooth_center_3d, combined_R, focal_len, light_dir)

        else:
            index_history.clear()
            smooth_bloom *= 0.92

        # 9. Update & render 3D harmonic bloom ripples
        alive_ripples = []
        for rip in ripples:
            rip.update()
            if rip.life > 0.04:
                if rip.radius > 5.0:
                    angs = np.linspace(0.0, 2.0 * math.pi, 40, endpoint=False, dtype=np.float32)
                    pts_loc = np.stack([rip.radius * np.cos(angs), rip.radius * np.sin(angs), np.zeros_like(angs)], axis=-1)
                    pts_cam = (pts_loc @ rip.R.T) + rip.center
                    if np.all(pts_cam[:, 2] > 50.0):
                        ring_2d = project_pts(pts_cam, focal_len, w / 2.0, h / 2.0)
                        col_outer = shade_color(rip.color, rip.life * 0.95, 15.0 * rip.life)
                        col_inner = shade_color((255, 255, 255), rip.life * 0.85, 0.0)
                        cv2.polylines(frame, [ring_2d], True, col_outer, max(1, int(3.5 * rip.life)), cv2.LINE_AA)
                        cv2.polylines(frame, [ring_2d], True, col_inner, 1, cv2.LINE_AA)
                alive_ripples.append(rip)
        ripples = alive_ripples

        # 10. Update & render glowing bokeh pollen spores
        alive_pollens = []
        for pol in pollen_particles:
            pol.update()
            if pol.life > 0.0 and pol.pos[2] > 50.0:
                px = int(pol.pos[0] * focal_len / pol.pos[2] + w / 2.0)
                py = int(pol.pos[1] * focal_len / pol.pos[2] + h / 2.0)
                pulse = 0.82 + 0.18 * math.sin(pol.pulse_phase)
                pr = max(1, int(pol.radius * pol.life * pulse * focal_len / pol.pos[2]))
                if 0 <= px < w and 0 <= py < h:
                    halo_col = shade_color(pol.color, pol.life * 0.72, 0.0)
                    core_col = shade_color(pol.color, min(1.0, pol.life * 1.1), 55.0 * pol.life)
                    cv2.circle(frame, (px, py), pr + 2, halo_col, -1, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), max(1, pr), core_col, -1, cv2.LINE_AA)
                alive_pollens.append(pol)
        pollen_particles = alive_pollens

        # 11. Update & render 3D curved falling petals
        alive_petals = []
        for p in petal_particles:
            p.update()
            if p.life > 0.0:
                outer_3d, inner_3d, norm_z = p.get_transformed_3d()
                if np.all(outer_3d[:, 2] > 50.0):
                    outer_2d = project_pts(outer_3d, focal_len, w / 2.0, h / 2.0)
                    inner_2d = project_pts(inner_3d, focal_len, w / 2.0, h / 2.0)

                    k_lit = (0.68 + 0.32 * norm_z) * max(0.25, min(1.0, p.life * 1.15))
                    c_out = shade_color(p.tip_color, k_lit, 12.0 * norm_z)
                    c_in = shade_color(lerp_bgr(p.tip_color, p.base_color, 0.55), k_lit * 0.94, 0.0)
                    c_rim = shade_color(p.tip_color, min(1.15, k_lit * 1.06), 18.0 * norm_z)

                    cv2.fillPoly(frame, [outer_2d], c_out, cv2.LINE_AA)
                    cv2.fillPoly(frame, [inner_2d], c_in, cv2.LINE_AA)
                    cv2.polylines(frame, [outer_2d], True, c_rim, 1, cv2.LINE_AA)
                alive_petals.append(p)
        petal_particles = alive_petals

        # ==============================================================================
        # GLASSMORPHISM HUD (CLEAN ASCII ENGLISH)
        # ==============================================================================
        hud_roi = frame[20:155, 20:545]
        dark_card = np.full_like(hud_roi, (16, 15, 22))
        cv2.addWeighted(dark_card, 0.76, hud_roi, 0.24, 0, dst=hud_roi)
        cv2.rectangle(frame, (20, 20), (545, 155), current_flower["primary_bgr"], 1, cv2.LINE_AA)

        cv2.putText(
            frame, f"{current_flower['name']}",
            (35, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.62, current_flower["primary_bgr"], 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Meaning: {current_flower['meaning']}",
            (35, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 235), 1, cv2.LINE_AA
        )

        bloom_pct = int(smooth_bloom * 100)
        cv2.putText(
            frame, f"3D Bloom: {bloom_pct}%",
            (35, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA
        )

        bar_x, bar_y, bar_max_w, bar_h = 165, 107, 355, 18
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h), (48, 48, 56), -1)
        cur_w = int(bar_max_w * smooth_bloom)
        if cur_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + cur_w, bar_y + bar_h), current_flower["primary_bgr"], -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + cur_w, bar_y + bar_h), (255, 255, 255), 1, cv2.LINE_AA)

        if now < swipe_banner_until:
            cv2.putText(
                frame, swipe_banner_text,
                (35, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (130, 255, 195), 2, cv2.LINE_AA
            )

        cv2.putText(
            frame, "Pinch/Open Hand: 3D Bloom | Swipe Index: Change Flower | Tilt Hand: 3D Rotate | Press Q: Exit",
            (25, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA
        )

        cv2.imshow("Japanese 3D Botanical Flowers - Computer Vision Project", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()