import io
import json
import math
import zipfile
from dataclasses import asdict, dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# IO helpers
# ---------------------------
def read_npz_phi_k(file_obj) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(file_obj)
    if "phi" not in data or "k" not in data:
        raise KeyError("NPZ must contain arrays with keys: 'phi' and 'k'")
    phi = np.array(data["phi"], dtype=np.float32)
    k = np.array(data["k"], dtype=np.float32)
    if phi.ndim != 2 or k.ndim != 2:
        raise ValueError("phi and k must be 2D arrays")
    if phi.shape != k.shape:
        raise ValueError(f"phi and k must have the same shape. Got {phi.shape} vs {k.shape}")
    return phi, k


def read_schedule_csv(file_obj) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(file_obj)
    cols = {c.lower(): c for c in df.columns}
    if "t" not in cols or "q" not in cols:
        raise KeyError("Schedule CSV must contain columns: 't' and 'q'")
    t = df[cols["t"]].to_numpy(dtype=np.float32)
    q = df[cols["q"]].to_numpy(dtype=np.float32)
    if len(t) < 2:
        raise ValueError("Schedule must have at least 2 rows.")
    order = np.argsort(t)
    return t[order], q[order]


# ---------------------------
# Numerics
# ---------------------------
def geom_mean(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x)
    return float(np.exp(np.mean(np.log(np.maximum(x, eps)))))


def _pad_edge(a: np.ndarray) -> np.ndarray:
    return np.pad(a, 1, mode="edge")


def grad_center(p: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    pp = _pad_edge(p)
    gx = (pp[2:, 1:-1] - pp[:-2, 1:-1]) / (2.0 * dx)
    gy = (pp[1:-1, 2:] - pp[1:-1, :-2]) / (2.0 * dy)
    return gx.astype(np.float32), gy.astype(np.float32)


def div_alpha_grad(p: np.ndarray, alpha: np.ndarray, dx: float, dy: float) -> np.ndarray:
    ppad = _pad_edge(p)
    apad = _pad_edge(alpha)

    a_e = 0.5 * (apad[1:-1, 1:-1] + apad[2:, 1:-1])
    a_w = 0.5 * (apad[1:-1, 1:-1] + apad[:-2, 1:-1])
    a_n = 0.5 * (apad[1:-1, 1:-1] + apad[1:-1, 2:])
    a_s = 0.5 * (apad[1:-1, 1:-1] + apad[1:-1, :-2])

    dp_e = (ppad[2:, 1:-1] - ppad[1:-1, 1:-1]) / dx
    dp_w = (ppad[1:-1, 1:-1] - ppad[:-2, 1:-1]) / dx
    dp_n = (ppad[1:-1, 2:] - ppad[1:-1, 1:-1]) / dy
    dp_s = (ppad[1:-1, 1:-1] - ppad[1:-1, :-2]) / dy

    fx = (a_e * dp_e - a_w * dp_w) / dx
    fy = (a_n * dp_n - a_s * dp_s) / dy
    return (fx + fy).astype(np.float32)


def upwind_advect(S: np.ndarray, ux: np.ndarray, uy: np.ndarray, dx: float, dy: float, dt: float) -> np.ndarray:
    Sp = _pad_edge(S)
    uxp = _pad_edge(ux)
    uyp = _pad_edge(uy)

    u = uxp[1:-1, 1:-1]
    dSdx_pos = (Sp[1:-1, 1:-1] - Sp[:-2, 1:-1]) / dx
    dSdx_neg = (Sp[2:, 1:-1] - Sp[1:-1, 1:-1]) / dx
    dSdx = np.where(u >= 0.0, dSdx_pos, dSdx_neg)

    v = uyp[1:-1, 1:-1]
    dSdy_pos = (Sp[1:-1, 1:-1] - Sp[1:-1, :-2]) / dy
    dSdy_neg = (Sp[1:-1, 2:] - Sp[1:-1, 1:-1]) / dy
    dSdy = np.where(v >= 0.0, dSdy_pos, dSdy_neg)

    return (S - dt * (u * dSdx + v * dSdy)).astype(np.float32)


def diffuse_aniso(S: np.ndarray, D: np.ndarray, anisD: float, dx: float, dy: float, dt: float) -> np.ndarray:
    Sp = _pad_edge(S)
    Dc = D.astype(np.float32)
    Dx = Dc
    Dy = Dc * float(anisD)

    lapx = (Sp[2:, 1:-1] - 2.0 * Sp[1:-1, 1:-1] + Sp[:-2, 1:-1]) / (dx * dx)
    lapy = (Sp[1:-1, 2:] - 2.0 * Sp[1:-1, 1:-1] + Sp[1:-1, :-2]) / (dy * dy)

    return (S + dt * (Dx * lapx + Dy * lapy)).astype(np.float32)


def gaussian_source(nx: int, ny: int, wi: int, wj: int, rad: float) -> np.ndarray:
    y = np.arange(ny, dtype=np.float32)[None, :]
    x = np.arange(nx, dtype=np.float32)[:, None]
    r2 = (x - wi) ** 2 + (y - wj) ** 2
    g = np.exp(-0.5 * r2 / max(rad * rad, 1e-6)).astype(np.float32)
    g /= (g.sum() + 1e-12)
    return g


def land_trap_floor(Sg_max: np.ndarray, Sgr_max: float, C_L: float) -> np.ndarray:
    denom = Sg_max + float(C_L) + 1e-8
    Sgr = float(Sgr_max) * (Sg_max / denom)
    return np.clip(Sgr, 0.0, float(Sgr_max)).astype(np.float32)


# ---------------------------
# Forward model parameters
# ---------------------------
@dataclass
class ForwardParams:
    # grid/time
    dx: float = 1.0
    dy: float = 1.0
    dt: float = 1.0
    Nt: int = 80

    # pressure driver
    mu: float = 1.0
    ct: float = 1e-5
    p_substeps: int = 2

    # transport knobs
    D0: float = 0.10
    anisD: float = 1.0
    vel_scale: float = 1.0

    # saturation source/sink
    src_amp: float = 0.08
    rad_w: float = 2.5
    prod_frac: float = 0.0

    # bounds + trapping
    Swr: float = 0.10
    Sgr_max: float = 0.20
    C_L: float = 0.15

    # shaping
    mob_exp: float = 1.2
    hc: float = 0.10


def run_forward(phi: np.ndarray, k: np.ndarray,
                t_sched: np.ndarray, q_sched: np.ndarray,
                well_ij: Tuple[int, int],
                prm: ForwardParams,
                thr: float = 0.05) -> Dict[str, np.ndarray]:
    nx, ny = phi.shape
    wi, wj = int(well_ij[0]), int(well_ij[1])
    wi = int(np.clip(wi, 0, nx - 1))
    wj = int(np.clip(wj, 0, ny - 1))

    # normalize k for heterogeneity coupling
    k_eff = geom_mean(k)
    k_norm = (k / (k_eff + 1e-12)).astype(np.float32)

    # pressure diffusivity
    alpha = (k / (np.maximum(phi, 1e-6) * prm.mu * prm.ct)).astype(np.float32)

    # time grid and interpolated schedule
    t_grid = (np.arange(prm.Nt, dtype=np.float32) * float(prm.dt)).astype(np.float32)
    q_grid = np.interp(t_grid, t_sched, q_sched).astype(np.float32)

    G = gaussian_source(nx, ny, wi, wj, prm.rad_w)

    p = np.zeros((nx, ny), dtype=np.float32)
    Sg = np.zeros((nx, ny), dtype=np.float32)
    Sg_max = np.zeros((nx, ny), dtype=np.float32)

    sg_list = np.zeros((prm.Nt, nx, ny), dtype=np.float32)
    p_list = np.zeros((prm.Nt, nx, ny), dtype=np.float32)

    area_ts = np.zeros(prm.Nt, dtype=np.float32)
    r_eq_ts = np.zeros(prm.Nt, dtype=np.float32)

    sub = max(1, int(prm.p_substeps))
    dtp = float(prm.dt) / sub

    for n in range(prm.Nt):
        qn = float(q_grid[n])

        # source for dp/dt
        src_p = (qn * G).astype(np.float32)

        for _ in range(sub):
            p = (p + dtp * (div_alpha_grad(p, alpha, prm.dx, prm.dy) + src_p)).astype(np.float32)

        gx, gy = grad_center(p, prm.dx, prm.dy)
        ux = (-(k / prm.mu) * gx * prm.vel_scale).astype(np.float32)
        uy = (-(k / prm.mu) * gy * prm.vel_scale).astype(np.float32)

        # advect
        Sg_adv = upwind_advect(Sg, ux, uy, prm.dx, prm.dy, prm.dt)

        # diffusion / spreading
        mob = np.power(np.clip(Sg, 0.0, 1.0) + 1e-6, float(prm.mob_exp)).astype(np.float32)
        D = (float(prm.D0) * (1.0 + float(prm.hc) * k_norm) * (0.25 + mob)).astype(np.float32)
        Sg_dif = diffuse_aniso(Sg_adv, D, prm.anisD, prm.dx, prm.dy, prm.dt)

        # inject / produce effect on gas saturation
        if qn > 0:
            Sg_src = Sg_dif + float(prm.src_amp) * qn * G
        elif qn < 0 and prm.prod_frac > 0:
            Sg_src = Sg_dif - float(prm.prod_frac) * (-qn) * G
        else:
            Sg_src = Sg_dif

        Sg_src = np.clip(Sg_src, 0.0, 1.0).astype(np.float32)
        Sg_max = np.maximum(Sg_max, Sg_src)

        # residual trapping floor
        Sgr = land_trap_floor(Sg_max, prm.Sgr_max, prm.C_L)
        Sg = np.maximum(Sg_src, Sgr)
        Sg = np.clip(Sg, 0.0, 1.0 - float(prm.Swr)).astype(np.float32)

        sg_list[n] = Sg
        p_list[n] = p

        mask = (Sg >= float(thr))
        area = float(mask.sum())
        area_ts[n] = area
        r_eq_ts[n] = float(math.sqrt(area / math.pi)) if area > 0 else 0.0

    return {
        "sg_list": sg_list,
        "p_list": p_list,
        "t_grid": t_grid,
        "q_grid": q_grid,
        "area_ts": area_ts,
        "r_eq_ts": r_eq_ts,
        "well_ij": np.array([wi, wj], dtype=np.int32),
    }


# ---------------------------
# Plot + export
# ---------------------------
def fig_imshow(arr: np.ndarray, title: str = "", vmin=None, vmax=None):
    fig, ax = plt.subplots()
    im = ax.imshow(arr, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def fig_line(x: np.ndarray, y: np.ndarray, xlabel: str = "", ylabel: str = "", title: str = ""):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    return fig


def make_zip_bytes(outputs: Dict[str, np.ndarray],
                   prm: ForwardParams,
                   thr: float,
                   include_snapshots: bool = True,
                   snap_ids: List[int] = None) -> bytes:
    if snap_ids is None:
        snap_ids = [0, outputs["sg_list"].shape[0] - 1]

    sg_list = outputs["sg_list"]
    t = outputs["t_grid"]
    q = outputs["q_grid"]
    area = outputs["area_ts"]
    r_eq = outputs["r_eq_ts"]

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        params_dict = asdict(prm)
        params_dict["thr"] = float(thr)
        params_dict["well_ij"] = outputs["well_ij"].tolist()
        z.writestr("params.json", json.dumps(params_dict, indent=2))

        df = pd.DataFrame({"t": t, "q": q, "area": area, "r_eq": r_eq})
        z.writestr("timeseries_plume.csv", df.to_csv(index=False))

        if include_snapshots:
            for tidx in snap_ids:
                tidx = int(np.clip(tidx, 0, sg_list.shape[0] - 1))
                fig = fig_imshow(sg_list[tidx], title=f"Sg predicted | tidx={tidx}", vmin=0, vmax=1)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                z.writestr(f"snapshots/sg_t{tidx:04d}.png", buf.getvalue())

    mem.seek(0)
    return mem.getvalue()
