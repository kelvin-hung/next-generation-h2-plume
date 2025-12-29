# ve_core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np


# -------------------------
# Results container
# -------------------------
@dataclass
class ForwardResult:
    t: np.ndarray
    q: np.ndarray
    sg_list: List[np.ndarray]
    p_list: Optional[List[np.ndarray]]
    area: np.ndarray
    r_eq: np.ndarray


# -------------------------
# Default params (tune in app)
# -------------------------
DEFAULT_PARAMS: Dict[str, float] = {
    # shared
    "Swr": 0.2,
    "Sgr_max": 0.35,
    "C_L": 0.6,           # Land-like residualization strength
    "prod_frac": 0.5,     # production removes this fraction of mobile gas
    "eps": 1e-12,

    # gaussian_k model
    "gauss_r0": 1.5,      # base radius (cells)
    "gauss_beta": 0.35,   # growth with sqrt(injected volume proxy)
    "gauss_w": 1.2,       # front thickness (cells)
    "gauss_gamma": 0.75,  # how strongly k channels modulate saturation
    "gauss_smooth_sigma": 1.2,  # smooth log10(k) before using it

    # advdiff model
    "p_D0": 0.15,         # pressure diffusivity scale
    "p_src_amp": 2.0,     # pressure source amplitude
    "sg_D": 0.08,         # saturation diffusion
    "sg_adv": 0.6,        # advection strength
    "sg_nl": 1.0,         # nonlinearity for flux limiter (>=1)
    "sg_clip_logk": 3.0,  # clip log10(k) range around median (+/-)
}


# -------------------------
# Utilities
# -------------------------
def _nanfill(a: np.ndarray, fill: float) -> np.ndarray:
    b = a.copy()
    b[~np.isfinite(b)] = fill
    return b

def _gaussian_blur2d(a: np.ndarray, sigma: float) -> np.ndarray:
    # very small, dependency-free Gaussian blur using separable 1D conv
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

def prepare_phi_k(phi: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns: phi_clean, k_norm (0..1), mask(active)"""
    phi = np.asarray(phi, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)

    mask = np.isfinite(phi) & np.isfinite(k) & (phi > 0) & (k > 0)
    if mask.sum() < 10:
        raise ValueError("Too few active cells in phi/k (check ACTNUM/masking).")

    phi_med = float(np.nanmedian(phi[mask]))
    k_med = float(np.nanmedian(k[mask]))
    phi_c = _nanfill(phi, phi_med)
    k_c = _nanfill(k, k_med)

    # normalize k to [0,1] in log-space (robust)
    logk = np.log10(np.clip(k_c, 1e-20, None))
    lo = float(np.nanpercentile(logk[mask], 5))
    hi = float(np.nanpercentile(logk[mask], 95))
    if hi <= lo:
        hi = lo + 1.0
    logk_n = (logk - lo) / (hi - lo)
    logk_n = np.clip(logk_n, 0.0, 1.0).astype(np.float32)

    return phi_c.astype(np.float32), logk_n, mask.astype(bool)

def choose_well_ij(k_norm: np.ndarray, mask: np.ndarray, mode: str, ij: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    nx, ny = k_norm.shape
    if mode == "manual":
        if ij is None:
            raise ValueError("manual mode requires ij.")
        i, j = int(ij[0]), int(ij[1])
        i = int(np.clip(i, 0, nx - 1))
        j = int(np.clip(j, 0, ny - 1))
        if not mask[i, j]:
            # fallback to nearest active
            ii, jj = np.where(mask)
            d = (ii - i) ** 2 + (jj - j) ** 2
            k = int(np.argmin(d))
            return int(ii[k]), int(jj[k])
        return i, j

    if mode == "center":
        i0, j0 = nx // 2, ny // 2
        if mask[i0, j0]:
            return i0, j0
        ii, jj = np.where(mask)
        d = (ii - i0) ** 2 + (jj - j0) ** 2
        k = int(np.argmin(d))
        return int(ii[k]), int(jj[k])

    if mode == "max_k":
        kk = k_norm.copy()
        kk[~mask] = -1.0
        idx = int(np.argmax(kk))
        i, j = np.unravel_index(idx, kk.shape)
        return int(i), int(j)

    raise ValueError("well_mode must be one of: max_k, center, manual")


# -------------------------
# Model A: gaussian_k (robust smooth plume)
# -------------------------
def _model_gaussian_k(phi, k_raw, t, q, params, wi, wj, mask) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # Use smoothed log10(k) to avoid Norne checkerboard
    logk = np.log10(np.clip(k_raw, 1e-20, None)).astype(np.float32)
    med = float(np.nanmedian(logk[mask]))
    clip = float(params.get("sg_clip_logk", 3.0))
    logk = np.clip(logk, med - clip, med + clip)

    sigma = float(params.get("gauss_smooth_sigma", 1.2))
    logk_s = _gaussian_blur2d(logk, sigma=sigma)

    # Normalize smoothed logk to [0,1]
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
    w = float(params.get("gauss_w", 1.2))
    gamma = float(params.get("gauss_gamma", 0.75))
    prod_frac = float(params.get("prod_frac", 0.5))
    C_L = float(params.get("C_L", 0.6))
    Sgr_max = float(params.get("Sgr_max", 0.35))

    # dt from t (works even if t is [0,10,20,...])
    t = np.asarray(t, dtype=np.float32)
    dt = np.diff(t, prepend=t[0]).astype(np.float32)
    dt[0] = dt[1] if len(dt) > 1 else 1.0
    dt = np.clip(dt, 1e-6, None)

    inj = np.maximum(q, 0.0).astype(np.float32)
    prd = np.maximum(-q, 0.0).astype(np.float32)

    # injected volume proxy
    V = np.cumsum(inj * dt).astype(np.float32)
    # production reduces mobile, not residual
    Veff = V - prod_frac * np.cumsum(prd * dt).astype(np.float32)
    Veff = np.maximum(Veff, 0.0)

    sg_list = []
    p_list = []

    # residual saturation (Land-ish) tracked as history of max mobile
    s_res = np.zeros((nx, ny), dtype=np.float32)
    s_max = np.zeros((nx, ny), dtype=np.float32)

    for n in range(len(t)):
        r = r0 + beta * np.sqrt(float(Veff[n]) + 1e-6)
        # smooth front (sigmoid)
        front = 1.0 / (1.0 + np.exp((dist - r) / max(w, 1e-3))).astype(np.float32)

        # channel preference via k01
        chan = np.power(np.clip(k01, 0.0, 1.0), gamma).astype(np.float32)
        sg = front * chan

        # normalize peak to ~1 in active region
        m = float(np.nanmax(sg[mask])) if mask.any() else 1.0
        if m > 1e-6:
            sg = (sg / m).astype(np.float32)

        # update Land-like residualization
        s_max = np.maximum(s_max, sg)
        s_res = np.minimum(Sgr_max, C_L * s_max)

        # apply production by reducing mobile only (keep residual)
        if q[n] < 0:
            sg = np.maximum(s_res, sg * (1.0 - prod_frac)).astype(np.float32)
        else:
            sg = np.maximum(s_res, sg).astype(np.float32)

        sg[~mask] = np.nan
        sg_list.append(sg.astype(np.float32))

        # pressure surrogate: simple Gaussian blob (for visualization)
        p = np.exp(-(dist * dist) / (2.0 * (r0 + 2.0) ** 2)).astype(np.float32)
        p *= float(np.maximum(0.0, inj[n] + 0.25))  # show stronger when injecting
        p[~mask] = np.nan
        p_list.append(p.astype(np.float32))

    return sg_list, p_list


# -------------------------
# Model B: advdiff (pressure diffusion + sg transport)
# -------------------------
def _laplace(a: np.ndarray) -> np.ndarray:
    # 5-point Laplacian with Neumann-ish edges (edge replicate)
    ap = np.pad(a, ((1, 1), (1, 1)), mode="edge")
    c = ap[1:-1, 1:-1]
    up = ap[:-2, 1:-1]
    dn = ap[2:, 1:-1]
    lt = ap[1:-1, :-2]
    rt = ap[1:-1, 2:]
    return (up + dn + lt + rt - 4.0 * c).astype(np.float32)

def _grad(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ap = np.pad(a, ((1, 1), (1, 1)), mode="edge")
    c = ap[1:-1, 1:-1]
    di = (ap[2:, 1:-1] - ap[:-2, 1:-1]) * 0.5
    dj = (ap[1:-1, 2:] - ap[1:-1, :-2]) * 0.5
    return di.astype(np.float32), dj.astype(np.float32)

def _advect_upwind(S, ui, uj) -> np.ndarray:
    # simple first-order upwind divergence of (u*S)
    # fluxes on faces
    nx, ny = S.shape
    Sp = np.pad(S, ((1, 1), (1, 1)), mode="edge")
    uip = np.pad(ui, ((1, 1), (1, 1)), mode="edge")
    ujp = np.pad(uj, ((1, 1), (1, 1)), mode="edge")

    # i-faces
    u_i_plus = uip[2:, 1:-1]
    u_i_minus = uip[1:-1, 1:-1]
    S_i_plus = np.where(u_i_plus >= 0, Sp[1:-1, 1:-1], Sp[2:, 1:-1])
    S_i_minus = np.where(u_i_minus >= 0, Sp[:-2, 1:-1], Sp[1:-1, 1:-1])

    # j-faces
    u_j_plus = ujp[1:-1, 2:]
    u_j_minus = ujp[1:-1, 1:-1]
    S_j_plus = np.where(u_j_plus >= 0, Sp[1:-1, 1:-1], Sp[1:-1, 2:])
    S_j_minus = np.where(u_j_minus >= 0, Sp[1:-1, :-2], Sp[1:-1, 1:-1])

    div = (u_i_plus * S_i_plus - u_i_minus * S_i_minus) + (u_j_plus * S_j_plus - u_j_minus * S_j_minus)
    return div.astype(np.float32)

def _model_advdiff(phi, k_raw, t, q, params, wi, wj, mask) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    nx, ny = phi.shape
    eps = float(params.get("eps", 1e-12))

    # stabilize Norne k by clipping log10(k)
    k = np.clip(k_raw, 1e-20, None).astype(np.float32)
    logk = np.log10(k).astype(np.float32)
    med = float(np.nanmedian(logk[mask]))
    clip = float(params.get("sg_clip_logk", 3.0))
    logk = np.clip(logk, med - clip, med + clip)
    k_stab = np.power(10.0, logk).astype(np.float32)

    # normalize for numerics
    k_ref = float(np.nanmedian(k_stab[mask]))
    if not np.isfinite(k_ref) or k_ref <= 0:
        k_ref = 1.0
    kN = (k_stab / k_ref).astype(np.float32)

    # dt
    t = np.asarray(t, dtype=np.float32)
    dt = np.diff(t, prepend=t[0]).astype(np.float32)
    dt[0] = dt[1] if len(dt) > 1 else 1.0
    dt = np.clip(dt, 1e-6, None)

    pD0 = float(params.get("p_D0", 0.15))
    p_src = float(params.get("p_src_amp", 2.0))
    sgD = float(params.get("sg_D", 0.08))
    sgA = float(params.get("sg_adv", 0.6))
    sg_nl = float(params.get("sg_nl", 1.0))

    prod_frac = float(params.get("prod_frac", 0.5))
    C_L = float(params.get("C_L", 0.6))
    Sgr_max = float(params.get("Sgr_max", 0.35))

    # fields
    p = np.zeros((nx, ny), dtype=np.float32)
    S = np.zeros((nx, ny), dtype=np.float32)
    Smax = np.zeros((nx, ny), dtype=np.float32)
    Sres = np.zeros((nx, ny), dtype=np.float32)

    sg_list = []
    p_list = []

    for n in range(len(t)):
        # pressure diffusion with source at well
        lap_p = _laplace(p)
        p = p + dt[n] * (pD0 * (kN * lap_p))

        # add source when injecting / sink when producing
        src = np.zeros_like(p, dtype=np.float32)
        src[wi, wj] = float(q[n]) * p_src
        p = p + dt[n] * src

        # velocity ~ -k grad(p)
        dpi, dpj = _grad(p)
        ui = -(kN * dpi).astype(np.float32)
        uj = -(kN * dpj).astype(np.float32)

        # nonlinear mobility-ish factor to avoid checkerboard
        mob = np.power(np.clip(S, 0.0, 1.0) + 1e-3, sg_nl).astype(np.float32)
        div_adv = _advect_upwind(S, ui * mob, uj * mob)
        lap_S = _laplace(S)

        # inject adds gas at well cell (scaled by phi)
        inj = max(float(q[n]), 0.0)
        prd = max(float(-q[n]), 0.0)
        add = np.zeros_like(S, dtype=np.float32)
        add[wi, wj] = inj / max(float(phi[wi, wj]), 1e-3)

        S = S + dt[n] * (-sgA * div_adv + sgD * lap_S + add)
        S = np.clip(S, 0.0, 1.0).astype(np.float32)

        # residualization + production removes mobile
        Smax = np.maximum(Smax, S)
        Sres = np.minimum(Sgr_max, C_L * Smax)

        if prd > 0:
            S = np.maximum(Sres, S * (1.0 - prod_frac)).astype(np.float32)
        else:
            S = np.maximum(Sres, S).astype(np.float32)

        S[~mask] = np.nan
        p_out = p.copy()
        p_out[~mask] = np.nan

        sg_list.append(S.astype(np.float32))
        p_list.append(p_out.astype(np.float32))

    return sg_list, p_list


# -------------------------
# Public API used by app.py
# -------------------------
def run_forward(
    phi: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    q: np.ndarray,
    params: Dict[str, float],
    well_mode: str = "max_k",
    well_ij: Optional[Tuple[int, int]] = None,
    return_pressure: bool = True,
    thr_area: float = 0.05,
    model: str = "advdiff",   # <--- choose: "paper"(alias), "gaussian_k", "advdiff"
) -> ForwardResult:
    """
    Forward prediction without observations.
    model:
      - "gaussian_k": fast smooth plume, channel-aware
      - "advdiff": pressure diffusion + advection-diffusion saturation (recommended for Norne)
      - "paper": alias to gaussian_k (keep your UI text consistent)
    """
    phi_c, k_norm_for_well, mask = prepare_phi_k(phi, k)

    if well_mode == "manual":
        if well_ij is None:
            well_ij = (0, 0)
        wi, wj = choose_well_ij(k_norm_for_well, mask, "manual", ij=well_ij)
    else:
        wi, wj = choose_well_ij(k_norm_for_well, mask, well_mode)

    t = np.asarray(t, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)

    m = model.lower().strip()
    if m == "paper":
        m = "gaussian_k"

    if m == "gaussian_k":
        sg_list, p_list = _model_gaussian_k(phi_c, k.astype(np.float32), t, q, params, wi, wj, mask)
    elif m == "advdiff":
        sg_list, p_list = _model_advdiff(phi_c, k.astype(np.float32), t, q, params, wi, wj, mask)
    else:
        raise ValueError("Unknown model. Use: paper, gaussian_k, advdiff")

    # metrics
    area = np.zeros(len(sg_list), dtype=np.float32)
    r_eq = np.zeros(len(sg_list), dtype=np.float32)
    for i, sg in enumerate(sg_list):
        finite = np.isfinite(sg)
        a = np.sum((sg > thr_area) & finite)
        area[i] = float(a)
        r_eq[i] = float(np.sqrt(max(a, 0.0) / np.pi))

    if not return_pressure:
        p_list = None

    return ForwardResult(
        t=t,
        q=q,
        sg_list=sg_list,
        p_list=p_list,
        area=area,
        r_eq=r_eq,
    )
