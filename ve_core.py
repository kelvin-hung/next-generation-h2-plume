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
    "src_amp": 289.9128164502606,
    "prod_frac": 0.21370220515124894,
    "Swr": 0.32291144862162,
    "Sgr_max": 0.07441779469199929,
    "C_L": 0.5969969351350207,
    "eps_h": 0.08728309474730092,
    "nu": 0.0010582605526077622,
    "m_spread": 2.731778941305009,
    "rad_w": 1,
    "dx": 1.0,
    "dy": 1.0,

    # Capillary fringe + anisotropy
    "hc": 0.10,
    "mob_exp": 1.20,
    "anisD": 1.00,

    # Pressure surrogate (dimensionless calibration)
    "ap_diff": 0.03,
    "qp_amp": 1.0,

    # Numerical stabilizers / switches
    "pressure_use_div_k_grad": 1.0,   # 1.0=True, 0.0=False (fallback to k*laplacian)
    "advection_nonperiodic": 1.0,     # 1.0=True, 0.0=False (fallback to np.roll)
    "smooth_well_kernel": 1.0,        # 1.0=True, 0.0=False (flat square)
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
    out = a.astype(np.float32, copy=True)
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = np.float32(val)
    return out


def laplacian(a: np.ndarray) -> np.ndarray:
    """5-point Laplacian with Neumann-like edge handling via padding."""
    a = a.astype(np.float32, copy=False)
    ap = np.pad(a, ((1, 1), (1, 1)), mode="edge")
    return (ap[1:-1, 2:] + ap[1:-1, :-2] + ap[2:, 1:-1] + ap[:-2, 1:-1] - 4.0 * ap[1:-1, 1:-1]).astype(np.float32)


def central_grad(a: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Central gradient (∂/∂x along i, ∂/∂y along j) with edge padding."""
    a = a.astype(np.float32, copy=False)
    ap = np.pad(a, ((1, 1), (1, 1)), mode="edge")
    gx = (ap[2:, 1:-1] - ap[:-2, 1:-1]) / (2.0 * float(dx))
    gy = (ap[1:-1, 2:] - ap[1:-1, :-2]) / (2.0 * float(dy))
    return gx.astype(np.float32), gy.astype(np.float32)


def div_k_grad(p: np.ndarray, k: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """
    Conservative operator: ∇·(k ∇p) using face (harmonic) transmissibilities.
    p,k: (nx,ny)
    """
    p = p.astype(np.float32, copy=False)
    k = k.astype(np.float32, copy=False)

    pp = np.pad(p, ((1, 1), (1, 1)), mode="edge")
    kp = np.pad(k, ((1, 1), (1, 1)), mode="edge")

    pc = pp[1:-1, 1:-1]
    pe = pp[2:, 1:-1]
    pw = pp[:-2, 1:-1]
    pn = pp[1:-1, 2:]
    ps = pp[1:-1, :-2]

    kc = kp[1:-1, 1:-1]
    ke = kp[2:, 1:-1]
    kw = kp[:-2, 1:-1]
    kn = kp[1:-1, 2:]
    ks = kp[1:-1, :-2]

    # harmonic mean at faces (stable for high-contrast k)
    kE = 2.0 * kc * ke / (kc + ke + 1e-12)
    kW = 2.0 * kc * kw / (kc + kw + 1e-12)
    kN = 2.0 * kc * kn / (kc + kn + 1e-12)
    kS = 2.0 * kc * ks / (kc + ks + 1e-12)

    dx2 = float(dx) ** 2
    dy2 = float(dy) ** 2

    fluxE = kE * (pe - pc) / dx2
    fluxW = kW * (pc - pw) / dx2
    fluxN = kN * (pn - pc) / dy2
    fluxS = kS * (pc - ps) / dy2

    return (fluxE - fluxW + fluxN - fluxS).astype(np.float32)


def upwind_advect_nonperiodic(h: np.ndarray, ux: np.ndarray, uy: np.ndarray, dt: float, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """
    First-order upwind advection with edge padding (NO periodic wrap).
    """
    h = h.astype(np.float32, copy=False)
    ux = ux.astype(np.float32, copy=False)
    uy = uy.astype(np.float32, copy=False)

    hp = np.pad(h, ((1, 1), (1, 1)), mode="edge")
    hc = hp[1:-1, 1:-1]
    hE = hp[2:, 1:-1]
    hW = hp[:-2, 1:-1]
    hN = hp[1:-1, 2:]
    hS = hp[1:-1, :-2]

    dhdx = np.where(ux > 0, (hc - hW) / float(dx), (hE - hc) / float(dx))
    dhdy = np.where(uy > 0, (hc - hS) / float(dy), (hN - hc) / float(dy))

    return (hc - dt * (ux * dhdx + uy * dhdy)).astype(np.float32)


def upwind_advect_periodic_roll(h: np.ndarray, ux: np.ndarray, uy: np.ndarray, dt: float) -> np.ndarray:
    """
    Old behavior (periodic wrap). Kept only for debugging.
    """
    h = h.astype(np.float32, copy=False)
    ux = ux.astype(np.float32, copy=False)
    uy = uy.astype(np.float32, copy=False)

    h_up = np.roll(h, 1, axis=0)
    h_dn = np.roll(h, -1, axis=0)
    dhdx = np.where(ux > 0, h - h_up, h_dn - h)

    h_lt = np.roll(h, 1, axis=1)
    h_rt = np.roll(h, -1, axis=1)
    dhdy = np.where(uy > 0, h - h_lt, h_rt - h)

    return (h - dt * (ux * dhdx + uy * dhdy)).astype(np.float32)


def k_spreading_power_aniso(
    h: np.ndarray,
    k_norm: np.ndarray,
    D0x: float,
    D0y: float,
    eps_h: float,
    m_spread: float,
    dt: float,
) -> np.ndarray:
    """
    Nonlinear k-controlled spreading term (paper):
      flux ~ D0 * k * (h+eps)^m * grad(h)
    Implemented in a conservative finite-volume-ish way.
    """
    h = h.astype(np.float32, copy=False)
    k_norm = k_norm.astype(np.float32, copy=False)

    hp = np.pad(h, ((1, 1), (1, 1)), mode="edge")
    kp = np.pad(k_norm, ((1, 1), (1, 1)), mode="edge")

    h_c = hp[1:-1, 1:-1]
    h_e = hp[2:, 1:-1]
    h_w = hp[:-2, 1:-1]
    h_n = hp[1:-1, 2:]
    h_s = hp[1:-1, :-2]

    k_c = kp[1:-1, 1:-1]
    k_e = kp[2:, 1:-1]
    k_w = kp[:-2, 1:-1]
    k_n = kp[1:-1, 2:]
    k_s = kp[1:-1, :-2]

    # harmonic-ish mean for k at faces
    ke = 2.0 * k_c * k_e / (k_c + k_e + 1e-12)
    kw = 2.0 * k_c * k_w / (k_c + k_w + 1e-12)
    kn = 2.0 * k_c * k_n / (k_c + k_n + 1e-12)
    ks = 2.0 * k_c * k_s / (k_c + k_s + 1e-12)

    he = 0.5 * (h_c + h_e)
    hw = 0.5 * (h_c + h_w)
    hn = 0.5 * (h_c + h_n)
    hs = 0.5 * (h_c + h_s)

    Ce = D0x * ke * np.power(np.maximum(he + eps_h, 0.0), m_spread)
    Cw = D0x * kw * np.power(np.maximum(hw + eps_h, 0.0), m_spread)
    Cn = D0y * kn * np.power(np.maximum(hn + eps_h, 0.0), m_spread)
    Cs = D0y * ks * np.power(np.maximum(hs + eps_h, 0.0), m_spread)

    Fe = Ce * (h_e - h_c)
    Fw = Cw * (h_c - h_w)
    Fn = Cn * (h_c - h_n)
    Fs = Cs * (h_s - h_c)

    divF = (Fe - Fw) + (Fs - Fn)
    return (h + dt * divF).astype(np.float32)


def apply_well_source_sink(
    h: np.ndarray,
    q_sign: float,
    q_w: float,
    src_amp: float,
    prod_frac: float,
    wi: int,
    wj: int,
    rad_w: int,
    dt: float,
    smooth_kernel: bool = True,
) -> np.ndarray:
    """
    Apply injection (+) or production (-) around (wi,wj).
    Smooth kernel prevents ring/checkerboard artifacts near well.
    """
    if q_sign == 0.0:
        return h

    h = h.astype(np.float32, copy=False)
    nx, ny = h.shape
    r = int(max(0, rad_w))
    i0, i1 = max(0, wi - r), min(nx, wi + r + 1)
    j0, j1 = max(0, wj - r), min(ny, wj + r + 1)

    if q_sign > 0:
        if smooth_kernel:
            sig2 = 2.0 * max(1.0, float(r)) ** 2
            for ii in range(i0, i1):
                for jj in range(j0, j1):
                    di = float(ii - wi)
                    dj = float(jj - wj)
                    w = float(np.exp(-(di * di + dj * dj) / (sig2 + 1e-12)))
                    h[ii, jj] += np.float32(src_amp * q_w * dt * w)
        else:
            h[i0:i1, j0:j1] += np.float32(src_amp * q_w * dt)
    else:
        # production removes a fraction of h
        h[i0:i1, j0:j1] *= np.float32(max(0.0, 1.0 - prod_frac * q_w * dt))

    return h


# -------------------------
# VE saturation + hysteresis
# -------------------------

def ve_mobile_sg_from_h(h: np.ndarray, Swr: float, hc: float, mob_exp: float) -> np.ndarray:
    """
    Capillary/fringe mapping:
      eff = clip((h - hc)/(1-hc), 0, 1)
      Sg_mob = (1-Swr) * eff^mob_exp
    """
    hc = float(np.clip(hc, 0.0, 0.9))
    eff = np.clip((h - hc) / (1.0 - hc + 1e-12), 0.0, 1.0)
    eff = np.power(eff, max(0.25, float(mob_exp)))
    return ((1.0 - float(Swr)) * eff).astype(np.float32)


def land_residual(Sg_max: np.ndarray, Sgr_max: float, C_L: float) -> np.ndarray:
    return (float(Sgr_max) * (Sg_max / (Sg_max + float(C_L) + 1e-12))).astype(np.float32)


# -------------------------
# Input prep
# -------------------------

def prepare_phi_k(phi: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      phi_clean (float32, with inactive set to 1)
      k_norm (float32, inactive set to 0)
      mask (bool) active cells
    """
    phi = _nan_to_num_inplace(phi, val=np.nan)
    k = _nan_to_num_inplace(k, val=np.nan)

    mask = np.isfinite(phi) & np.isfinite(k) & (phi > 0)
    if mask.sum() == 0:
        raise ValueError("No active cells found in phi/k (all NaN or non-positive).")

    # clean fields for simulation
    phi_clean = np.where(mask, phi, 1.0).astype(np.float32)
    k_clean = np.where(mask, k, 0.0).astype(np.float32)

    # k normalization: geometric mean over active cells (robust)
    k_pos = k_clean[mask]
    k_pos = k_pos[k_pos > 0]
    if k_pos.size == 0:
        k_eff = 1.0
    else:
        k_eff = float(np.exp(np.mean(np.log(k_pos + 1e-30))))
    k_norm = (k_clean / (k_eff + 1e-12)).astype(np.float32)

    return phi_clean, k_norm, mask


def choose_well_ij(k_norm: np.ndarray, mask: np.ndarray, mode: str, ij: Tuple[int, int] | None = None) -> Tuple[int, int]:
    nx, ny = k_norm.shape
    if mode == "max_k":
        tmp = np.where(mask, k_norm, -np.inf)
        wi, wj = np.unravel_index(int(np.argmax(tmp)), tmp.shape)
        return int(wi), int(wj)
    if mode == "center":
        ii, jj = np.where(mask)
        wi = int(np.round(np.mean(ii)))
        wj = int(np.round(np.mean(jj)))
        return wi, wj
    if mode == "manual":
        if ij is None:
            raise ValueError("manual mode requires ij.")
        wi, wj = int(ij[0]), int(ij[1])
        wi = int(np.clip(wi, 0, nx - 1))
        wj = int(np.clip(wj, 0, ny - 1))
        return wi, wj
    raise ValueError("mode must be one of: max_k, center, manual")


# -------------------------
# Pressure surrogate (paper)
# -------------------------

def simulate_pressure_from_schedule(
    k_norm: np.ndarray,
    well_ij: Tuple[int, int],
    q: np.ndarray,
    ap_diff: float,
    qp_amp: float,
    dx: float,
    dy: float,
    use_div_k_grad: bool = True,
) -> List[np.ndarray]:
    """
    Paper-compatible pressure surrogate:

      p_{t+1} = p_t + ap_diff * ( ∇·(k_norm ∇p_t) ) + source(q_t)

    If use_div_k_grad=False, falls back to legacy:
      p_{t+1} = p_t + ap_diff * (k_norm * laplacian(p_t)) + source(q_t)
    """
    wi, wj = well_ij
    Nt = int(q.size)
    p = np.zeros_like(k_norm, dtype=np.float32)
    out: List[np.ndarray] = []

    maxk = float(np.nanmax(k_norm))
    coefmax = (abs(float(ap_diff)) * maxk + 1e-12)
    dt_stable = 0.18 * (min(float(dx), float(dy)) ** 2) / coefmax
    nsub = int(np.clip(np.ceil(1.0 / max(dt_stable, 1e-6)), 1, 60))
    dt = 1.0 / nsub

    rad = 1

    for t in range(Nt):
        qt = float(q[t])
        for _ in range(nsub):
            if use_div_k_grad:
                div = div_k_grad(p, k_norm, dx=dx, dy=dy)
                p = p + (float(ap_diff) * dt) * div
            else:
                div = laplacian(p)
                p = p + (float(ap_diff) * dt) * (k_norm * div)

            if qt != 0.0:
                i0, i1 = max(0, wi - rad), min(p.shape[0], wi + rad + 1)
                j0, j1 = max(0, wj - rad), min(p.shape[1], wj + rad + 1)
                p[i0:i1, j0:j1] += np.float32(float(qp_amp) * qt * dt)

        out.append(p.astype(np.float32, copy=True))

    return out


# -------------------------
# VE forward solve (paper)
# -------------------------

def simulate_ve_from_pressure(
    p_list: List[np.ndarray],
    k_norm: np.ndarray,
    well_ij: Tuple[int, int],
    q: np.ndarray,
    params: Dict[str, float],
) -> List[np.ndarray]:
    wi, wj = well_ij
    Nt = len(p_list)

    dx = float(params.get("dx", 1.0))
    dy = float(params.get("dy", 1.0))

    D0        = float(params["D0"])
    alpha_p   = float(params["alpha_p"])
    src_amp   = float(params["src_amp"])
    prod_frac = float(params["prod_frac"])
    Swr       = float(params["Swr"])
    Sgr_max   = float(params["Sgr_max"])
    C_L       = float(params["C_L"])
    eps_h     = float(params["eps_h"])
    nu        = float(params["nu"])
    m_spread  = float(params["m_spread"])
    rad_w     = int(params.get("rad_w", 1))
    hc        = float(params.get("hc", 0.10))
    mob_exp   = float(params.get("mob_exp", 1.2))
    anisD     = float(params.get("anisD", 1.0))

    nonperiodic = bool(float(params.get("advection_nonperiodic", 1.0)) >= 0.5)
    smooth_well = bool(float(params.get("smooth_well_kernel", 1.0)) >= 0.5)

    maxk = float(np.nanmax(k_norm))
    D0x = D0 * anisD
    D0y = D0 / (anisD + 1e-12)
    coefmax = (max(D0x, D0y) * maxk * (1.0 + eps_h) ** max(1.0, m_spread) + nu)
    dt_stable = 0.18 * (min(dx, dy) ** 2) / (coefmax + 1e-12)
    nsub = int(np.clip(np.ceil(1.0 / max(dt_stable, 1e-6)), 1, 80))
    dt = 1.0 / nsub

    h = np.zeros_like(p_list[0], dtype=np.float32)
    Sg_max_hist = np.zeros_like(h, dtype=np.float32)

    sg_pred: List[np.ndarray] = []

    for t in range(Nt):
        gx, gy = central_grad(p_list[t], dx=dx, dy=dy)
        ux = (-alpha_p * k_norm * gx).astype(np.float32)
        uy = (-alpha_p * k_norm * gy).astype(np.float32)

        qt = float(q[t])
        q_sign = 0.0
        q_w = 0.0
        if qt > 0:
            q_sign, q_w = 1.0, abs(qt)
        elif qt < 0:
            q_sign, q_w = -1.0, abs(qt)

        for _ in range(nsub):
            if nonperiodic:
                h = upwind_advect_nonperiodic(h, ux, uy, dt=dt, dx=dx, dy=dy)
            else:
                h = upwind_advect_periodic_roll(h, ux, uy, dt=dt)

            h = k_spreading_power_aniso(h, k_norm, D0x=D0x, D0y=D0y, eps_h=eps_h, m_spread=m_spread, dt=dt)
            h = h + (nu * dt) * laplacian(h)
            h = apply_well_source_sink(h, q_sign, q_w, src_amp, prod_frac, wi, wj, rad_w, dt=dt, smooth_kernel=smooth_well)
            h = np.clip(h, 0.0, 1.0).astype(np.float32)

        sg_mob = ve_mobile_sg_from_h(h, Swr=Swr, hc=hc, mob_exp=mob_exp).astype(np.float32)
        Sg_max_hist = np.maximum(Sg_max_hist, sg_mob)
        sg_res = land_residual(Sg_max_hist, Sgr_max=Sgr_max, C_L=C_L).astype(np.float32)
        sg_tot = np.maximum(sg_mob, sg_res)
        sg_tot = np.clip(sg_tot, 0.0, 1.0).astype(np.float32)

        sg_pred.append(sg_tot)

    return sg_pred


# -------------------------
# Public forward API
# -------------------------

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
    Forward prediction without sg_obs:
      (phi,k) + schedule -> pressure surrogate -> VE -> sg(t)

    Inputs:
      t,q: arrays length Nt. (t can be 0..Nt-1)
    """
    if params is None:
        params = dict(DEFAULT_PARAMS)
    else:
        merged = dict(DEFAULT_PARAMS)
        merged.update(params)
        params = merged

    _, k_norm, mask = prepare_phi_k(phi, k)
    wi, wj = choose_well_ij(k_norm, mask, mode=well_mode, ij=well_ij)

    q = np.asarray(q, dtype=np.float32).reshape(-1)
    t = np.asarray(t, dtype=np.float32).reshape(-1)
    if q.size != t.size:
        raise ValueError(f"t and q must have same length, got {t.size} vs {q.size}.")

    use_div = bool(float(params.get("pressure_use_div_k_grad", 1.0)) >= 0.5)

    p_list = simulate_pressure_from_schedule(
        k_norm=k_norm,
        well_ij=(wi, wj),
        q=q,
        ap_diff=float(params.get("ap_diff", DEFAULT_PARAMS["ap_diff"])),
        qp_amp=float(params.get("qp_amp", DEFAULT_PARAMS["qp_amp"])),
        dx=float(params.get("dx", 1.0)),
        dy=float(params.get("dy", 1.0)),
        use_div_k_grad=use_div,
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
