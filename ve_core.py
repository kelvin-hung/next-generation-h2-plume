# ve_core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np


@dataclass
class ForwardResult:
    t: np.ndarray
    q: np.ndarray
    sg_list: List[np.ndarray]
    p_list: Optional[List[np.ndarray]]
    area: np.ndarray
    r_eq: np.ndarray
    meta: Dict[str, float]


DEFAULT_PARAMS: Dict[str, float] = {
    # Land-ish residualization (simple)
    "Swr": 0.2,
    "Sgr_max": 0.35,
    "C_L": 0.6,
    "prod_frac": 0.5,

    # Smooth VE-ish plume model controls
    "gauss_r0": 1.5,
    "gauss_beta": 0.35,
    "gauss_w": 2.2,              # front thickness
    "gauss_gamma": 0.75,         # k-channel exponent
    "gauss_smooth_sigma": 1.6,   # smooth log10(k)

    # Pressure surrogate (visual only)
    "p_width": 2.5,

    # Post smoothing of Sg (very helpful on blocky grids)
    "sg_smooth_sigma": 1.2,

    # Guardrails
    "sg_clip_logk": 3.0,
}


def _nanfill(a: np.ndarray, fill: float) -> np.ndarray:
    b = a.copy()
    b[~np.isfinite(b)] = fill
    return b


def _gaussian_blur2d(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return a
    r = int(max(1, round(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2 * sigma * sigma)).astype(np.float32)
    k /= np.sum(k)

    def conv1d_axis(arr, axis):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        ap = np.pad(arr, pad, mode="edge")
        out = np.zeros_like(arr, dtype=np.float32)
        if axis == 0:
            for i in range(arr.shape[0]):
                window = ap[i:i + 2 * r + 1, :]
                out[i, :] = np.sum(window * k[:, None], axis=0)
        else:
            for j in range(arr.shape[1]):
                window = ap[:, j:j + 2 * r + 1]
                out[:, j] = np.sum(window * k[None, :], axis=1)
        return out

    a32 = a.astype(np.float32)
    a1 = conv1d_axis(a32, axis=0)
    a2 = conv1d_axis(a1, axis=1)
    return a2


def _apply_masked_smooth(S: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return S
    S0 = np.nan_to_num(S, nan=0.0).astype(np.float32)
    M0 = mask.astype(np.float32)
    num = _gaussian_blur2d(S0 * M0, sigma)
    den = _gaussian_blur2d(M0, sigma)
    out = np.where(den > 1e-8, num / den, 0.0).astype(np.float32)
    out[~mask] = np.nan
    return out


def prepare_phi_k(phi: np.ndarray, k: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns: phi_clean, k_norm(0..1), mask(active)."""
    phi = np.asarray(phi, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)

    if mask is None:
        mask = np.isfinite(phi) & np.isfinite(k) & (phi > 0) & (k > 0)
    else:
        mask = mask.astype(bool) & np.isfinite(phi) & np.isfinite(k)

    if int(mask.sum()) < 10:
        raise ValueError("Too few active cells. Check ACTNUM/mask and phi/k validity.")

    phi_med = float(np.nanmedian(phi[mask]))
    k_med = float(np.nanmedian(k[mask]))
    phi_c = _nanfill(phi, phi_med)
    k_c = _nanfill(k, k_med)

    logk = np.log10(np.clip(k_c, 1e-20, None))
    lo = float(np.nanpercentile(logk[mask], 5))
    hi = float(np.nanpercentile(logk[mask], 95))
    if hi <= lo:
        hi = lo + 1.0
    k01 = np.clip((logk - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    return phi_c.astype(np.float32), k01, mask.astype(bool)


def choose_well_ij(k_norm: np.ndarray, mask: np.ndarray, mode: str, ij: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    nx, ny = k_norm.shape

    def nearest_active(i0, j0):
        ii, jj = np.where(mask)
        d = (ii - i0) ** 2 + (jj - j0) ** 2
        k = int(np.argmin(d))
        return int(ii[k]), int(jj[k])

    if mode == "manual":
        if ij is None:
            raise ValueError("manual mode requires ij")
        i, j = int(np.clip(int(ij[0]), 0, nx - 1)), int(np.clip(int(ij[1]), 0, ny - 1))
        return (i, j) if mask[i, j] else nearest_active(i, j)

    if mode == "center":
        i0, j0 = nx // 2, ny // 2
        return (i0, j0) if mask[i0, j0] else nearest_active(i0, j0)

    if mode == "max_k":
        kk = k_norm.copy()
        kk[~mask] = -1.0
        idx = int(np.argmax(kk))
        i, j = np.unravel_index(idx, kk.shape)
        return int(i), int(j)

    raise ValueError("well_mode must be one of: max_k, center, manual")


def _model_smooth_ve(phi, k_raw, t, q, params, wi, wj, mask):
    logk = np.log10(np.clip(k_raw, 1e-20, None)).astype(np.float32)
    med = float(np.nanmedian(logk[mask]))
    clip = float(params.get("sg_clip_logk", 3.0))
    logk = np.clip(logk, med - clip, med + clip)

    logk_s = _gaussian_blur2d(logk, sigma=float(params.get("gauss_smooth_sigma", 1.6)))

    lo = float(np.nanpercentile(logk_s[mask], 5))
    hi = float(np.nanpercentile(logk_s[mask], 95))
    if hi <= lo:
        hi = lo + 1.0
    k01 = np.clip((logk_s - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    nx, ny = phi.shape
    ii, jj = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32), indexing="ij")
    di = ii - float(wi)
    dj = jj - float(wj)
    dist = np.sqrt(di * di + dj * dj).astype(np.float32)

    r0 = float(params.get("gauss_r0", 1.5))
    beta = float(params.get("gauss_beta", 0.35))
    w = float(params.get("gauss_w", 2.2))
    gamma = float(params.get("gauss_gamma", 0.75))

    prod_frac = float(params.get("prod_frac", 0.5))
    C_L = float(params.get("C_L", 0.6))
    Sgr_max = float(params.get("Sgr_max", 0.35))
    sg_smooth_sigma = float(params.get("sg_smooth_sigma", 1.2))

    t = np.asarray(t, dtype=np.float32)
    dt = np.diff(t, prepend=t[0]).astype(np.float32)
    dt[0] = dt[1] if len(dt) > 1 else 1.0
    dt = np.clip(dt, 1e-6, None)

    inj = np.maximum(q, 0.0).astype(np.float32)
    prd = np.maximum(-q, 0.0).astype(np.float32)

    V = np.cumsum(inj * dt).astype(np.float32)
    Veff = V - prod_frac * np.cumsum(prd * dt).astype(np.float32)
    Veff = np.maximum(Veff, 0.0)

    sg_list, p_list = [], []
    s_res = np.zeros((nx, ny), dtype=np.float32)
    s_max = np.zeros((nx, ny), dtype=np.float32)

    p_width = float(params.get("p_width", 2.5))

    for n in range(len(t)):
        r = r0 + beta * np.sqrt(float(Veff[n]) + 1e-6)

        front = 1.0 / (1.0 + np.exp((dist - r) / max(w, 1e-3))).astype(np.float32)
        chan = np.power(np.clip(k01, 0.0, 1.0), gamma).astype(np.float32)
        sg = front * chan

        m = float(np.nanmax(sg[mask])) if mask.any() else 1.0
        if m > 1e-6:
            sg = (sg / m).astype(np.float32)

        s_max = np.maximum(s_max, sg)
        s_res = np.minimum(Sgr_max, C_L * s_max)

        if q[n] < 0:
            sg = np.maximum(s_res, sg * (1.0 - prod_frac)).astype(np.float32)
        else:
            sg = np.maximum(s_res, sg).astype(np.float32)

        sg[~mask] = np.nan
        sg = _apply_masked_smooth(sg, mask, sg_smooth_sigma)
        sg_list.append(sg.astype(np.float32))

        # pressure surrogate (just for plotting)
        p = np.exp(-(dist * dist) / (2.0 * (p_width**2))).astype(np.float32)
        p *= float(np.maximum(0.0, inj[n] + 0.25))
        p[~mask] = np.nan
        p_list.append(p.astype(np.float32))

    return sg_list, p_list


def run_forward(
    phi: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    q: np.ndarray,
    params: Dict[str, float],
    well_mode: str = "max_k",
    well_ij: Optional[Tuple[int, int]] = None,
    mask: Optional[np.ndarray] = None,
    return_pressure: bool = True,
    thr_area: float = 0.05,
    dx: float = 1.0,
    dy: float = 1.0,
) -> ForwardResult:
    phi_c, k_norm_for_well, mask2 = prepare_phi_k(phi, k, mask=mask)

    if well_mode == "manual":
        wi, wj = choose_well_ij(k_norm_for_well, mask2, "manual", ij=well_ij or (0, 0))
    else:
        wi, wj = choose_well_ij(k_norm_for_well, mask2, well_mode)

    t = np.asarray(t, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)

    sg_list, p_list = _model_smooth_ve(phi_c, k.astype(np.float32), t, q, params, wi, wj, mask2)

    area = np.zeros(len(sg_list), dtype=np.float32)
    r_eq = np.zeros(len(sg_list), dtype=np.float32)

    cell_area = float(dx) * float(dy)

    for i, sg in enumerate(sg_list):
        finite = np.isfinite(sg)
        a_cells = np.sum((sg > thr_area) & finite)
        area[i] = float(a_cells) * cell_area
        r_eq[i] = float(np.sqrt(max(area[i], 0.0) / np.pi))

    if not return_pressure:
        p_list = None

    meta = {"dx": float(dx), "dy": float(dy), "cell_area": float(cell_area), "well_i": float(wi), "well_j": float(wj)}
    return ForwardResult(t=t, q=q, sg_list=sg_list, p_list=p_list, area=area, r_eq=r_eq, meta=meta)
