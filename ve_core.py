"""
VE + Darcy + Land (forward-only) core.

Self-contained (numpy only) and designed to be imported by Streamlit.
Includes the forward paper model + small helpers for preprocessing and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np

# -------------------------
# Paper defaults
# -------------------------
START_PARAMS_DEFAULT = {
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

    # NEW (capillary fringe + anisotropy)
    "hc": 0.10,         # thickness cutoff before mobile gas appears
    "mob_exp": 1.20,    # mapping exponent (front sharpness)
    "anisD": 1.00,      # D_x = D0*anisD, D_y = D0/anisD

    # OPTIONAL pressure model params (dimensionless calibration)
    "ap_diff": 0.03,    # pressure diffusivity strength
    "qp_amp": 1.0,      # pressure source amplitude
}

DEFAULT_PARAMS: Dict[str, float] = dict(START_PARAMS_DEFAULT)

# -------------------------
# Utilities for app use
# -------------------------
def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def prepare_phi_k(phi: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare phi and k for the paper model.

    Returns:
        phi01 : phi normalized to [0,1] on active cells
        k01   : log10(k) normalized to [0,1] on active cells
        mask  : active boolean mask (finite phi & finite k & phi>0 & k>0)
    """
    phi = np.asarray(phi, dtype=np.float32)
    k   = np.asarray(k, dtype=np.float32)

    mask = np.isfinite(phi) & np.isfinite(k) & (phi > 0) & (k > 0)
    if mask.sum() < 10:
        raise ValueError("Too few active cells in phi/k (check NPZ keys, NaNs, or units).")

    p = phi.copy()
    p[~mask] = np.nan
    pmin = np.nanmin(p); pmax = np.nanmax(p)
    phi01 = (p - pmin) / (pmax - pmin + 1e-12)
    phi01 = np.clip(phi01, 0.0, 1.0).astype(np.float32)

    kk = k.copy()
    kk[~mask] = np.nan
    logk = np.log10(np.maximum(kk, 1e-30))
    lkmin = np.nanpercentile(logk, 1)
    lkmax = np.nanpercentile(logk, 99)
    logk = np.clip(logk, lkmin, lkmax)
    k01 = (logk - lkmin) / (lkmax - lkmin + 1e-12)
    k01 = np.clip(k01, 0.0, 1.0).astype(np.float32)

    phi01[~mask] = np.nan
    k01[~mask]   = np.nan
    return phi01, k01, mask

def choose_well_ij(k01: np.ndarray, mask: np.ndarray, mode: str = "max_k", ij: Optional[Tuple[int,int]] = None) -> Tuple[int,int]:
    """
    Choose well (i,j) on the active mask.
    - max_k : choose argmax(k01) on active
    - center: choose closest active cell to geometric center
    - manual: use ij (validated, snapped to nearest active if needed)
    """
    mode = (mode or "max_k").lower()
    nx, ny = k01.shape

    if mode == "manual":
        if ij is None:
            raise ValueError("manual mode requires ij")
        i, j = int(ij[0]), int(ij[1])
        if not (0 <= i < nx and 0 <= j < ny):
            raise ValueError("manual well (i,j) out of bounds")
        if not bool(mask[i, j]):
            ii, jj = np.where(mask)
            d2 = (ii - i)**2 + (jj - j)**2
            k = int(np.argmin(d2))
            return int(ii[k]), int(jj[k])
        return i, j

    if mode == "max_k":
        kk = np.where(mask, k01, -np.inf)
        idx = int(np.nanargmax(kk))
        i, j = np.unravel_index(idx, kk.shape)
        return int(i), int(j)

    if mode == "center":
        ii, jj = np.where(mask)
        ci = 0.5*(nx-1); cj = 0.5*(ny-1)
        d2 = (ii - ci)**2 + (jj - cj)**2
        k = int(np.argmin(d2))
        return int(ii[k]), int(jj[k])

    raise ValueError("mode must be one of: max_k, center, manual")

def _area_radius(sg: np.ndarray, mask: np.ndarray, thr: float) -> Tuple[float,float]:
    plume = np.isfinite(sg) & mask & (sg > thr)
    area = float(plume.sum())
    r_eq = float(np.sqrt(area / np.pi)) if area > 0 else 0.0
    return area, r_eq

@dataclass
class ForwardResult:
    sg_list: List[np.ndarray]
    p_list: Optional[List[np.ndarray]]
    t: np.ndarray
    q: np.ndarray
    area: np.ndarray
    r_eq: np.ndarray
    well_ij: Tuple[int,int]
    mask: np.ndarray
    params: Dict[str,float]

def run_forward(
    phi: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    q: np.ndarray,
    params: Optional[Dict[str,float]] = None,
    well_mode: str = "max_k",
    well_ij: Optional[Tuple[int,int]] = None,
    return_pressure: bool = True,
    thr_area: float = 0.05,
) -> ForwardResult:
    """
    Run the VE+Darcy+Land paper model forward without observations.
    """
    t = np.asarray(t, dtype=np.float32).reshape(-1)
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    if t.size != q.size:
        raise ValueError(f"Schedule mismatch: len(t)={t.size} but len(q)={q.size}")

    phi01, k01, mask = prepare_phi_k(phi, k)
    wi, wj = choose_well_ij(k01, mask, mode=well_mode, ij=well_ij)

    p0 = np.zeros_like(phi01, dtype=np.float32)

    pp = dict(DEFAULT_PARAMS)
    if params:
        # merge, coercing to float
        for key, val in params.items():
            if key in pp:
                pp[key] = _safe_float(val, pp[key])

    # no-observation pressure list
    p_obs_list = [np.zeros_like(p0, dtype=np.float32) for _ in range(int(t.size))]

    sg_list, p_list = simulate_ve(
        phi=phi01,
        k=k01,
        p0=p0,
        q=q,
        well_ij=(wi, wj),
        params=pp,
        p_obs_list=p_obs_list,
        return_pressure=bool(return_pressure),
    )

    area = np.zeros((t.size,), dtype=np.float32)
    r_eq = np.zeros((t.size,), dtype=np.float32)
    for n in range(int(t.size)):
        a, r = _area_radius(sg_list[n], mask, float(thr_area))
        area[n] = a
        r_eq[n] = r

    return ForwardResult(
        sg_list=sg_list,
        p_list=p_list,
        t=t,
        q=q,
        area=area,
        r_eq=r_eq,
        well_ij=(wi, wj),
        mask=mask,
        params=pp,
    )

# -------------------------
# Paper model implementation (copied from your paper script)
# -------------------------
def geom_mean(a: np.ndarray, eps=1e-30) -> float:
    a = np.asarray(a, dtype=float)
    a = np.clip(a, eps, None)
    return float(np.exp(np.mean(np.log(a))))

def central_grad(a: np.ndarray, dx=1.0, dy=1.0):
    ap = np.pad(a, ((1,1),(1,1)), mode="edge")
    dax = (ap[1:-1, 2:] - ap[1:-1, :-2]) * (0.5 / dx)
    day = (ap[2:, 1:-1] - ap[:-2, 1:-1]) * (0.5 / dy)
    return dax, day

def laplacian(a: np.ndarray):
    ap = np.pad(a, ((1,1),(1,1)), mode="edge")
    return (ap[1:-1, 2:] + ap[1:-1, :-2] + ap[2:, 1:-1] + ap[:-2, 1:-1] - 4.0*ap[1:-1, 1:-1])

def upwind_advect(h: np.ndarray, ux: np.ndarray, uy: np.ndarray, dt: float):
    hp = np.pad(h, ((1,1),(1,1)), mode="edge")
    hx_f = hp[1:-1, 2:] - hp[1:-1, 1:-1]
    hx_b = hp[1:-1, 1:-1] - hp[1:-1, :-2]
    hy_f = hp[2:, 1:-1] - hp[1:-1, 1:-1]
    hy_b = hp[1:-1, 1:-1] - hp[:-2, 1:-1]
    dhdx = np.where(ux >= 0, hx_b, hx_f)
    dhdy = np.where(uy >= 0, hy_b, hy_f)
    return h - dt * (ux*dhdx + uy*dhdy)

def k_spreading_power_aniso(h: np.ndarray, k_norm: np.ndarray, D0x: float, D0y: float, eps_h: float, m_spread: float, dt: float):
    """
    ∂h/∂t = ∂/∂x( D0x*k*(h+eps)^m * ∂h/∂x ) + ∂/∂y( D0y*k*(h+eps)^m * ∂h/∂y )
    """
    hp = np.pad(h, ((1,1),(1,1)), mode="edge")
    kp = np.pad(k_norm, ((1,1),(1,1)), mode="edge")

    h_c = hp[1:-1, 1:-1]
    h_e = hp[1:-1, 2:]
    h_w = hp[1:-1, :-2]
    h_n = hp[:-2, 1:-1]
    h_s = hp[2:, 1:-1]

    k_c = kp[1:-1, 1:-1]
    k_e = kp[1:-1, 2:]
    k_w = kp[1:-1, :-2]
    k_n = kp[:-2, 1:-1]
    k_s = kp[2:, 1:-1]

    def havg(a, b):
        return 2*a*b/(a+b+1e-12)

    ke = havg(k_c, k_e)
    kw = havg(k_c, k_w)
    kn = havg(k_c, k_n)
    ks = havg(k_c, k_s)

    he = 0.5*(h_c + h_e)
    hw = 0.5*(h_c + h_w)
    hn = 0.5*(h_c + h_n)
    hs = 0.5*(h_c + h_s)

    Ce = D0x * ke * np.power(np.maximum(he + eps_h, 0.0), m_spread)
    Cw = D0x * kw * np.power(np.maximum(hw + eps_h, 0.0), m_spread)
    Cn = D0y * kn * np.power(np.maximum(hn + eps_h, 0.0), m_spread)
    Cs = D0y * ks * np.power(np.maximum(hs + eps_h, 0.0), m_spread)

    Fe = Ce * (h_e - h_c)
    Fw = Cw * (h_c - h_w)
    Fn = Cn * (h_c - h_n)
    Fs = Cs * (h_s - h_c)

    divF = (Fe - Fw) + (Fs - Fn)
    return h + dt * divF

def ve_mobile_sg_from_h(h: np.ndarray, Swr: float, hc: float, mob_exp: float):
    """
    Capillary/fringe mapping:
      effective thickness = clip((h - hc)/(1-hc), 0, 1)
      Sg_mob = (1-Swr) * eff^mob_exp
    """
    hc = float(np.clip(hc, 0.0, 0.9))
    eff = np.clip((h - hc) / (1.0 - hc + 1e-12), 0.0, 1.0)
    eff = np.power(eff, max(0.25, float(mob_exp)))
    return (1.0 - Swr) * eff

def land_residual(Sg_max: np.ndarray, Sgr_max: float, C_L: float):
    return Sgr_max * (Sg_max / (Sg_max + C_L + 1e-12))

def plume_mask(sg, thr):
    return sg > thr

def infer_q_sign_and_weight(p_list, wi, wj):
    pw = np.array([p[wi, wj] for p in p_list], dtype=float)
    dp = np.diff(pw, prepend=pw[0])

    dead = np.std(dp) * 0.25 + 1e-6
    sign = np.where(dp > dead,  1.0, np.where(dp < -dead, -1.0, 0.0))

    dp_pos = np.clip(dp, 0.0, None)
    dp_neg = np.clip(-dp, 0.0, None)

    pos_ref = np.mean(dp_pos[dp_pos > dead]) if np.any(dp_pos > dead) else 1.0
    neg_ref = np.mean(dp_neg[dp_neg > dead]) if np.any(dp_neg > dead) else 1.0

    w = np.where(sign > 0, dp_pos / (pos_ref + 1e-12),
                 np.where(sign < 0, dp_neg / (neg_ref + 1e-12), 0.0))
    w = np.clip(w, 0.0, 3.0)
    return sign.astype(np.float32), w.astype(np.float32)

def apply_well_source_sink(h, q_sign, q_w, src_amp, prod_frac, wi, wj, rad_w, dt):
    h2 = h.copy()
    i0, i1 = max(0, wi-rad_w), min(h.shape[0], wi+rad_w+1)
    j0, j1 = max(0, wj-rad_w), min(h.shape[1], wj+rad_w+1)
    n = (i1-i0)*(j1-j0)
    if q_sign > 0:
        h2[i0:i1, j0:j1] += (src_amp * q_w * dt) / max(1, n)
    elif q_sign < 0:
        h2[i0:i1, j0:j1] -= (prod_frac * src_amp * q_w * dt) / max(1, n)
    return h2

def simulate_pressure(p_obs_list, k_norm, well_ij, q_sign, q_w, ap_diff, qp_amp, dx, dy):
    """
    Simple variable-coefficient pressure diffusion for Δp:
      ∂p/∂t = ap_diff * ∇·(k_norm ∇p) + qp_amp*q(t)*I_well
    """
    wi, wj = well_ij
    Nt = len(p_obs_list)
    p = p_obs_list[0].copy().astype(np.float32)
    p0 = p.copy()
    out = []

    # stability
    maxk = float(np.max(k_norm))
    dt_stable = 0.18 * (min(dx, dy)**2) / (ap_diff * maxk + 1e-12)
    nsub = int(np.clip(np.ceil(1.0 / max(dt_stable, 1e-6)), 1, 60))
    dt = 1.0 / nsub

    rad = 1
    for t in range(Nt):
        for _ in range(nsub):
            gx, gy = central_grad(p, dx=dx, dy=dy)
            # flux = k * grad(p)
            fx = (k_norm * gx).astype(np.float32)
            fy = (k_norm * gy).astype(np.float32)
            # div(k grad p) ~ laplacian(p) weighted approx
            div = laplacian(p)  # cheap fallback
            p = p + (ap_diff * dt) * (k_norm * div)

            # source term (sign + magnitude)
            if q_sign[t] != 0.0:
                i0, i1 = max(0, wi-rad), min(p.shape[0], wi+rad+1)
                j0, j1 = max(0, wj-rad), min(p.shape[1], wj+rad+1)
                n = (i1-i0)*(j1-j0)
                p[i0:i1, j0:j1] += (qp_amp * q_sign[t] * q_w[t] * dt) / max(1, n)

        out.append(p.copy())

    # shift baseline to match p0 exactly at t=0
    out[0] = p0
    return out

def simulate_ve(p_obs_list, k_norm, well_ij, params, fit_pressure=False, use_pred_p_for_vel=False):
    wi, wj = well_ij
    Nt = len(p_obs_list)
    dx = float(params.get("dx", DX_DEFAULT))
    dy = float(params.get("dy", DY_DEFAULT))

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
    rad_w     = int(params["rad_w"])
    hc        = float(params.get("hc", 0.0))
    mob_exp   = float(params.get("mob_exp", 1.0))
    anisD     = float(params.get("anisD", 1.0))

    q_sign, q_w = infer_q_sign_and_weight(p_obs_list, wi, wj)

    # optional predicted pressure
    p_pred_list = None
    if fit_pressure or use_pred_p_for_vel:
        ap_diff = float(params.get("ap_diff", 0.03))
        qp_amp  = float(params.get("qp_amp", 1.0))
        p_pred_list = simulate_pressure(p_obs_list, k_norm, well_ij, q_sign, q_w, ap_diff, qp_amp, dx, dy)

    # substepping for VE stability
    maxk = float(np.max(k_norm))
    D0x = D0 * anisD
    D0y = D0 / (anisD + 1e-12)
    coefmax = (max(D0x, D0y) * maxk * (1.0 + eps_h)**max(1.0, m_spread) + nu)
    dt_stable = 0.18 * (min(dx, dy)**2) / (coefmax + 1e-12)
    nsub = int(np.clip(np.ceil(1.0 / max(dt_stable, 1e-6)), 1, 40))
    dt = 1.0 / nsub

    h = np.zeros_like(p_obs_list[0], dtype=np.float32)
    Sg_max_hist = np.zeros_like(h, dtype=np.float32)

    sg_pred = []

    for t in range(Nt):
        p_field = p_pred_list[t] if (use_pred_p_for_vel and p_pred_list is not None) else p_obs_list[t]
        gx, gy = central_grad(p_field, dx=dx, dy=dy)
        ux = (-alpha_p * k_norm * gx).astype(np.float32)
        uy = (-alpha_p * k_norm * gy).astype(np.float32)

        for _ in range(nsub):
            h = upwind_advect(h, ux, uy, dt=dt)
            h = k_spreading_power_aniso(h, k_norm, D0x=D0x, D0y=D0y, eps_h=eps_h, m_spread=m_spread, dt=dt)
            h = h + (nu * dt) * laplacian(h)
            h = apply_well_source_sink(h, q_sign[t], q_w[t], src_amp, prod_frac, wi, wj, rad_w, dt=dt)
            h = np.clip(h, 0.0, 1.0).astype(np.float32)

        sg_mob = ve_mobile_sg_from_h(h, Swr=Swr, hc=hc, mob_exp=mob_exp).astype(np.float32)
        Sg_max_hist = np.maximum(Sg_max_hist, sg_mob)
        sg_res = land_residual(Sg_max_hist, Sgr_max=Sgr_max, C_L=C_L).astype(np.float32)
        sg_tot = np.maximum(sg_mob, sg_res)
        sg_tot = np.clip(sg_tot, 0.0, 1.0).astype(np.float32)

        sg_pred.append(sg_tot)

    return sg_pred, p_pred_list
