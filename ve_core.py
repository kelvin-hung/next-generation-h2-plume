"""
VE + Darcy + Land forward model (paper version)
- Takes phi/k 2D maps (np.ndarray)
- Takes injection schedule q(t) (CSV or array)
- Predicts Sg(t,i,j) sequence using:
    * pressure surrogate (diffusion + source) -> Darcy velocity
    * VE thickness h evolution (advection + k-spreading + diffusion + source/sink)
    * capillary-fringe mapping (hc, mob_exp) -> mobile Sg
    * Land hysteresis -> residual Sg
    * total Sg = max(mobile, residual)

This file is meant to be imported by Streamlit (app.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# -------------------------
# Parameters (from your paper script)
# -------------------------

DEFAULT_PARAMS: Dict[str, float] = {
    # Core VE parameters (paper-calibrated defaults)
    "D0": 0.27802127542157334,
    "alpha_p": 0.3035718747536404,
    "src_amp": 0.20917119946812723,
    "prod_frac": 0.5,
    "Swr": 0.2,
    "Sgr_max": 0.35,
    "C_L": 2.0,
    "hc": 1.0,
    "mob_exp": 1.0,
    "anisD": 1.0,
    "eps_h": 1e-6,
    "nu": 0.0,
    "m_spread": 1.0,
    "ap_diff": 1.0,
    "qp_amp": 1.0,
    "dx": 1.0,
    "dy": 1.0,
}


@dataclass
class ForwardResult:
    sg_list: List[np.ndarray]            # length Nt, each (nx,ny) masked (inactive -> NaN)
    p_list: Optional[List[np.ndarray]]   # length Nt, each (nx,ny)
    area: np.ndarray                     # (Nt,)
    r_eq: np.ndarray                     # (Nt,)
    q: np.ndarray                        # (Nt,)
    t: np.ndarray                        # (Nt,)


# -------------------------
# Numeric utilities
# -------------------------

def _nan_to_num_inplace(a: np.ndarray, val: float = 0.0) -> np.ndarray:
    """Return a float32 array with NaNs/Infs replaced."""
    out = np.array(a, dtype=np.float32, copy=True)
    out[~np.isfinite(out)] = np.float32(val)
    return out


def prepare_phi_k(phi: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare porosity and permeability maps:
      - converts to float32
      - builds active mask from finite values and positive phi,k
      - returns (phi_clean, k_norm, mask)
    """
    phi = np.array(phi, dtype=np.float32, copy=True)
    k = np.array(k, dtype=np.float32, copy=True)

    mask = np.isfinite(phi) & np.isfinite(k) & (phi > 0) & (k > 0)
    if mask.sum() == 0:
        raise ValueError("No active cells found (phi/k all invalid or non-positive).")

    # Clean phi
    phi_clean = phi.copy()
    phi_clean[~mask] = np.nan

    # Normalize k using log10 scaling on active cells
    k_act = k[mask]
    klog = np.log10(np.maximum(k_act, 1e-30))
    kmin, kmax = float(np.min(klog)), float(np.max(klog))
    denom = (kmax - kmin) if (kmax > kmin) else 1.0
    k_norm = np.full_like(k, np.nan, dtype=np.float32)
    k_norm[mask] = ((klog - kmin) / denom).astype(np.float32)

    return phi_clean, k_norm, mask


def choose_well_ij(
    k_norm: np.ndarray,
    mask: np.ndarray,
    mode: str = "max_k",
    ij: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    """
    Choose injector location:
      - max_k: cell with highest k_norm among active
      - center: nearest active to center
      - manual: user-provided ij (must be active)
    """
    nx, ny = k_norm.shape

    if mode == "manual":
        if ij is None:
            raise ValueError("manual mode requires ij=(i,j).")
        i, j = int(ij[0]), int(ij[1])
        if not (0 <= i < nx and 0 <= j < ny):
            raise ValueError("Manual ij out of bounds.")
        if not bool(mask[i, j]):
            raise ValueError("Manual ij is not an active cell.")
        return i, j

    if mode == "max_k":
        kk = np.where(mask, k_norm, -np.inf)
        flat = int(np.argmax(kk))
        i, j = np.unravel_index(flat, kk.shape)
        return int(i), int(j)

    if mode == "center":
        ci, cj = nx // 2, ny // 2
        if mask[ci, cj]:
            return int(ci), int(cj)

        # find nearest active cell
        act = np.argwhere(mask)
        d2 = (act[:, 0] - ci) ** 2 + (act[:, 1] - cj) ** 2
        idx = int(np.argmin(d2))
        return int(act[idx, 0]), int(act[idx, 1])

    raise ValueError("well mode must be one of: max_k, center, manual")


def laplace_aniso(a: np.ndarray, dx: float, dy: float, anis: float) -> np.ndarray:
    """2D Laplacian with anisotropy scaling in y direction."""
    a = np.array(a, dtype=np.float32, copy=False)
    axp = np.roll(a, -1, axis=0)
    axm = np.roll(a,  1, axis=0)
    ayp = np.roll(a, -1, axis=1)
    aym = np.roll(a,  1, axis=1)
    d2x = (axp - 2*a + axm) / (dx*dx)
    d2y = (ayp - 2*a + aym) / (dy*dy)
    return (d2x + anis * d2y).astype(np.float32)


def grad_central(a: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """Central gradient (da/dx, da/dy) with periodic-ish roll boundary."""
    a = np.array(a, dtype=np.float32, copy=False)
    axp = np.roll(a, -1, axis=0)
    axm = np.roll(a,  1, axis=0)
    ayp = np.roll(a, -1, axis=1)
    aym = np.roll(a,  1, axis=1)
    dax = (axp - axm) / (2*dx)
    day = (ayp - aym) / (2*dy)
    return dax.astype(np.float32), day.astype(np.float32)


def upwind_advect(h: np.ndarray, ux: np.ndarray, uy: np.ndarray, dx: float, dy: float, dt: float) -> np.ndarray:
    """Simple 2D upwind advection for a scalar h."""
    h = np.array(h, dtype=np.float32, copy=False)

    h_up = np.roll(h, 1, axis=0)
    h_dn = np.roll(h, -1, axis=0)
    dhdx = np.where(ux > 0, h - h_up, h_dn - h) / dx

    h_lt = np.roll(h, 1, axis=1)
    h_rt = np.roll(h, -1, axis=1)
    dhdy = np.where(uy > 0, h - h_lt, h_rt - h) / dy

    return (h - dt * (ux * dhdx + uy * dhdy)).astype(np.float32)


def gaussian_source(nx: int, ny: int, wi: int, wj: int, sigma: float = 2.5) -> np.ndarray:
    """Normalized Gaussian source centered at (wi,wj)."""
    ii = np.arange(nx, dtype=np.float32)[:, None]
    jj = np.arange(ny, dtype=np.float32)[None, :]
    d2 = (ii - wi) ** 2 + (jj - wj) ** 2
    g = np.exp(-0.5 * d2 / (sigma * sigma)).astype(np.float32)
    s = float(np.sum(g))
    if s > 0:
        g /= np.float32(s)
    return g


def simulate_pressure(
    k_norm: np.ndarray,
    well_ij: Tuple[int, int],
    q: np.ndarray,
    params: Dict[str, float],
) -> List[np.ndarray]:
    """
    Pressure surrogate: p_{t+1} = p_t + alpha_p * Laplace(p_t) + qp_amp*q[t]*source
    (masked/normalized via k_norm only through source placement and later Darcy).
    """
    nx, ny = k_norm.shape
    wi, wj = well_ij

    alpha_p = float(params["alpha_p"])
    qp_amp = float(params["qp_amp"])
    anisD = float(params["anisD"])
    dx = float(params.get("dx", 1.0))
    dy = float(params.get("dy", 1.0))

    src = gaussian_source(nx, ny, wi, wj, sigma=2.5)

    p = np.zeros((nx, ny), dtype=np.float32)
    out: List[np.ndarray] = []

    for tt in range(len(q)):
        out.append(p.copy())
        p = p + alpha_p * laplace_aniso(p, dx, dy, anisD) + (qp_amp * float(q[tt])) * src

    return out


def land_residual_from_max(Sg_max: np.ndarray, C_L: float, Sgr_max: float) -> np.ndarray:
    """Land model-like residual saturation from max historical mobile saturation."""
    # A safe monotone mapping: Sgr = (C_L * Sg_max) / (1 + C_L * Sg_max)
    Sgr = (C_L * Sg_max) / (1.0 + C_L * Sg_max)
    return np.minimum(Sgr, Sgr_max).astype(np.float32)


def thickness_to_mobile_sg(h: np.ndarray, hc: float, mob_exp: float) -> np.ndarray:
    """Map VE thickness to mobile saturation proxy."""
    h = np.maximum(h, 0.0).astype(np.float32)
    x = h / float(hc + 1e-12)
    # smooth saturating curve; exp-power
    sg = 1.0 - np.exp(-np.power(x, float(mob_exp)))
    return np.clip(sg, 0.0, 1.0).astype(np.float32)


def simulate_ve_from_pressure(
    p_list: List[np.ndarray],
    k_norm: np.ndarray,
    well_ij: Tuple[int, int],
    q: np.ndarray,
    params: Dict[str, float],
) -> List[np.ndarray]:
    """
    VE thickness evolution driven by Darcy velocity from pressure:
      - u = -k_norm * grad(p)
      - h_{t+1} = advect(h) + D0*Laplace(h) + src_amp*q[t]*source - prod_frac*|q<0|*source
      - mobile Sg from thickness; residual from Land using historical max
      - total Sg = max(mobile, residual)
    """
    nx, ny = k_norm.shape
    wi, wj = well_ij

    D0 = float(params["D0"])
    src_amp = float(params["src_amp"])
    prod_frac = float(params["prod_frac"])
    hc = float(params["hc"])
    mob_exp = float(params["mob_exp"])
    C_L = float(params["C_L"])
    Sgr_max = float(params["Sgr_max"])
    anisD = float(params["anisD"])
    eps_h = float(params["eps_h"])
    m_spread = float(params["m_spread"])
    nu = float(params["nu"])
    dx = float(params.get("dx", 1.0))
    dy = float(params.get("dy", 1.0))

    src = gaussian_source(nx, ny, wi, wj, sigma=2.5)

    h = np.zeros((nx, ny), dtype=np.float32)
    sg_max = np.zeros((nx, ny), dtype=np.float32)

    out: List[np.ndarray] = []

    # dt=1 in index-time units
    dt = 1.0

    for tt in range(len(q)):
        p = p_list[tt]

        dpx, dpy = grad_central(p, dx, dy)
        ux = -(k_norm * dpx)
        uy = -(k_norm * dpy)

        # optional viscosity-like damping
        if nu != 0.0:
            ux = ux / (1.0 + nu * np.abs(ux))
            uy = uy / (1.0 + nu * np.abs(uy))

        # advect
        h_adv = upwind_advect(h, ux, uy, dx, dy, dt)

        # diffusion + spreading
        h_diff = D0 * laplace_aniso(h_adv, dx, dy, anisD)
        h_spread = m_spread * laplace_aniso((k_norm * h_adv), dx, dy, anisD)

        # source/sink
        qt = float(q[tt])
        inj = max(qt, 0.0)
        prod = -min(qt, 0.0)

        h_new = h_adv + dt * (h_diff + h_spread + (src_amp * inj) * src - (prod_frac * prod) * src)

        # stabilize
        h_new = np.where(np.isfinite(h_new), h_new, 0.0).astype(np.float32)
        h_new = np.maximum(h_new, 0.0).astype(np.float32)

        # thickness -> mobile sg
        sg_m = thickness_to_mobile_sg(h_new + eps_h, hc=hc, mob_exp=mob_exp)

        # update max and residual
        sg_max = np.maximum(sg_max, sg_m)
        sg_r = land_residual_from_max(sg_max, C_L=C_L, Sgr_max=Sgr_max)

        sg_tot = np.maximum(sg_m, sg_r).astype(np.float32)
        out.append(sg_tot)

        h = h_new

    return out


def run_forward(
    phi: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    q: np.ndarray,
    params: Optional[Dict[str, float]] = None,
    well_mode: str = "max_k",
    well_ij: Optional[Tuple[int, int]] = None,
    return_pressure: bool = True,
    thr_area: float = 0.05,
) -> ForwardResult:
    """
    Public API used by Streamlit.
    """
    if params is None:
        params = dict(DEFAULT_PARAMS)
    else:
        tmp = dict(DEFAULT_PARAMS)
        tmp.update(params)
        params = tmp

    # sanitize
    t = np.array(t, dtype=np.float32)
    q = np.array(q, dtype=np.float32)
    if len(t) != len(q):
        raise ValueError(f"t and q must have same length. Got len(t)={len(t)} len(q)={len(q)}")

    phi_clean, k_norm, mask = prepare_phi_k(phi, k)

    if well_mode == "manual":
        if well_ij is None:
            raise ValueError("manual well_mode requires well_ij=(i,j)")
        wi, wj = choose_well_ij(k_norm, mask, "manual", ij=well_ij)
    else:
        wi, wj = choose_well_ij(k_norm, mask, well_mode)

    p_list = simulate_pressure(
        k_norm=k_norm,
        well_ij=(wi, wj),
        q=q,
        params=dict(
            alpha_p=float(params.get("alpha_p", DEFAULT_PARAMS["alpha_p"])),
            anisD=float(params.get("anisD", DEFAULT_PARAMS["anisD"])),
            ap_diff=float(params.get("ap_diff", DEFAULT_PARAMS["ap_diff"])),
            qp_amp=float(params.get("qp_amp", DEFAULT_PARAMS["qp_amp"])),
            dx=float(params.get("dx", 1.0)),
            dy=float(params.get("dy", 1.0)),
        ),
    )

    sg_list = simulate_ve_from_pressure(
        p_list=p_list,
        k_norm=k_norm,
        well_ij=(wi, wj),
        q=q,
        params=params,
    )

    sg_masked = [np.where(mask, s, np.nan).astype(np.float32) for s in sg_list]

    area = np.array([float(np.nansum(s > thr_area)) for s in sg_masked], dtype=np.float32)
    r_eq = np.sqrt(area / np.pi).astype(np.float32)

    return ForwardResult(
        sg_list=sg_masked,
        p_list=p_list if return_pressure else None,
        area=area,
        r_eq=r_eq,
        q=q.astype(np.float32),
        t=t.astype(np.float32),
    )
