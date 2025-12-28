from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

# -------------------------
# Parameters (add 2 scale knobs)
# -------------------------
DEFAULT_PARAMS: Dict[str, float] = {
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
    "dx": 1.0,
    "dy": 1.0,
    # NEW (helps match paper scaling)
    "k_vel_scale": 1.0,   # multiplies Darcy velocity
    "p_diff_scale": 1.0,  # multiplies div(k grad p)
    "src_sigma": 2.5,     # source width in grid cells
}

@dataclass
class ForwardResult:
    sg_list: List[np.ndarray]
    p_list: Optional[List[np.ndarray]]
    area: np.ndarray
    r_eq: np.ndarray
    q: np.ndarray
    t: np.ndarray
    well_ij: Tuple[int, int]
    max_sg: np.ndarray
    max_p: np.ndarray


# -------------------------
# Mask-aware utilities (NO wrap, no-flow at inactive)
# -------------------------
def prepare_phi_k(phi: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.array(phi, dtype=np.float32, copy=True)
    k = np.array(k, dtype=np.float32, copy=True)

    mask = np.isfinite(phi) & np.isfinite(k) & (phi > 0) & (k > 0)
    if mask.sum() == 0:
        raise ValueError("No active cells found (phi/k invalid or non-positive).")

    phi_clean = phi.copy()
    phi_clean[~mask] = np.nan

    # log-normalize permeability on active cells
    k_act = k[mask]
    klog = np.log10(np.maximum(k_act, 1e-30)).astype(np.float32)
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
    nx, ny = k_norm.shape

    if mode == "manual":
        if ij is None:
            raise ValueError("manual mode requires ij=(i,j)")
        i, j = int(ij[0]), int(ij[1])
        if not (0 <= i < nx and 0 <= j < ny):
            raise ValueError("Manual ij out of bounds.")
        if not bool(mask[i, j]):
            raise ValueError("Manual ij is not an active cell.")
        return i, j

    if mode == "max_k":
        kk = np.where(mask, k_norm, -np.inf)
        flat = int(np.argmax(kk))
        return tuple(map(int, np.unravel_index(flat, kk.shape)))

    if mode == "center":
        ci, cj = nx // 2, ny // 2
        if mask[ci, cj]:
            return int(ci), int(cj)
        act = np.argwhere(mask)
        d2 = (act[:, 0] - ci) ** 2 + (act[:, 1] - cj) ** 2
        idx = int(np.argmin(d2))
        return int(act[idx, 0]), int(act[idx, 1])

    raise ValueError("mode must be one of: max_k, center, manual")


def gaussian_source(nx: int, ny: int, wi: int, wj: int, sigma: float) -> np.ndarray:
    ii = np.arange(nx, dtype=np.float32)[:, None]
    jj = np.arange(ny, dtype=np.float32)[None, :]
    d2 = (ii - wi) ** 2 + (jj - wj) ** 2
    g = np.exp(-0.5 * d2 / (sigma * sigma)).astype(np.float32)
    s = float(np.sum(g))
    if s > 0:
        g /= np.float32(s)
    return g


def _nbr(a: np.ndarray, di: int, dj: int) -> np.ndarray:
    """Neighbor lookup WITHOUT wrap: out-of-bounds uses self (Neumann)."""
    nx, ny = a.shape
    out = a.copy()
    ii = np.arange(nx)[:, None]
    jj = np.arange(ny)[None, :]
    ni = ii + di
    nj = jj + dj
    ok = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny)
    out[ok] = a[ni[ok], nj[ok]]
    return out


def div_k_grad(p: np.ndarray, k: np.ndarray, mask: np.ndarray, dx: float, dy: float, anis: float) -> np.ndarray:
    """
    Compute divergence of flux: div( k * grad(p) )
    - no-flow at boundaries
    - no-flow across inactive neighbors (mask)
    """
    p = np.array(p, dtype=np.float32, copy=False)
    k = np.array(k, dtype=np.float32, copy=False)

    pxp = _nbr(p, +1, 0); pxm = _nbr(p, -1, 0)
    pyp = _nbr(p, 0, +1); pym = _nbr(p, 0, -1)

    kxp = _nbr(k, +1, 0); kxm = _nbr(k, -1, 0)
    kyp = _nbr(k, 0, +1); kym = _nbr(k, 0, -1)

    mxp = _nbr(mask.astype(np.float32), +1, 0) > 0.5
    mxm = _nbr(mask.astype(np.float32), -1, 0) > 0.5
    myp = _nbr(mask.astype(np.float32), 0, +1) > 0.5
    mym = _nbr(mask.astype(np.float32), 0, -1) > 0.5

    # face transmissibilities (average), zero if neighbor inactive
    kfxp = 0.5 * (k + kxp) * (mask & mxp)
    kfxm = 0.5 * (k + kxm) * (mask & mxm)
    kfyp = 0.5 * (k + kyp) * (mask & myp)
    kfym = 0.5 * (k + kym) * (mask & mym)

    fxp = kfxp * (pxp - p) / dx
    fxm = kfxm * (p - pxm) / dx
    fyp = kfyp * (pyp - p) / dy
    fym = kfym * (p - pym) / dy

    divx = (fxp - fxm) / dx
    divy = (fyp - fym) / dy

    out = (divx + anis * divy).astype(np.float32)
    out[~mask] = 0.0
    return out


def grad_center(p: np.ndarray, mask: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """Central grad with Neumann boundary; inactive treated as wall."""
    p = np.array(p, dtype=np.float32, copy=False)
    pxp = _nbr(p, +1, 0); pxm = _nbr(p, -1, 0)
    pyp = _nbr(p, 0, +1); pym = _nbr(p, 0, -1)

    mxp = _nbr(mask.astype(np.float32), +1, 0) > 0.5
    mxm = _nbr(mask.astype(np.float32), -1, 0) > 0.5
    myp = _nbr(mask.astype(np.float32), 0, +1) > 0.5
    mym = _nbr(mask.astype(np.float32), 0, -1) > 0.5

    # if neighbor inactive => use self (zero gradient)
    pxp = np.where(mxp, pxp, p)
    pxm = np.where(mxm, pxm, p)
    pyp = np.where(myp, pyp, p)
    pym = np.where(mym, pym, p)

    dpx = (pxp - pxm) / (2 * dx)
    dpy = (pyp - pym) / (2 * dy)

    dpx[~mask] = 0.0
    dpy[~mask] = 0.0
    return dpx.astype(np.float32), dpy.astype(np.float32)


def upwind_advect(h: np.ndarray, ux: np.ndarray, uy: np.ndarray, mask: np.ndarray, dx: float, dy: float, dt: float) -> np.ndarray:
    """
    Simple upwind without wrap + walls at inactive cells.
    """
    h = np.array(h, dtype=np.float32, copy=False)

    hxp = _nbr(h, +1, 0); hxm = _nbr(h, -1, 0)
    hyp = _nbr(h, 0, +1); hym = _nbr(h, 0, -1)

    mxp = _nbr(mask.astype(np.float32), +1, 0) > 0.5
    mxm = _nbr(mask.astype(np.float32), -1, 0) > 0.5
    myp = _nbr(mask.astype(np.float32), 0, +1) > 0.5
    mym = _nbr(mask.astype(np.float32), 0, -1) > 0.5

    # if upstream neighbor inactive => use self (zero gradient)
    hxp = np.where(mxp, hxp, h)
    hxm = np.where(mxm, hxm, h)
    hyp = np.where(myp, hyp, h)
    hym = np.where(mym, hym, h)

    # upwind derivatives
    dhdx = np.where(ux > 0, (h - hxm) / dx, (hxp - h) / dx)
    dhdy = np.where(uy > 0, (h - hym) / dy, (hyp - h) / dy)

    out = h - dt * (ux * dhdx + uy * dhdy)
    out[~mask] = 0.0
    return out.astype(np.float32)


def laplace_masked(a: np.ndarray, mask: np.ndarray, dx: float, dy: float, anis: float) -> np.ndarray:
    a = np.array(a, dtype=np.float32, copy=False)
    axp = _nbr(a, +1, 0); axm = _nbr(a, -1, 0)
    ayp = _nbr(a, 0, +1); aym = _nbr(a, 0, -1)

    mxp = _nbr(mask.astype(np.float32), +1, 0) > 0.5
    mxm = _nbr(mask.astype(np.float32), -1, 0) > 0.5
    myp = _nbr(mask.astype(np.float32), 0, +1) > 0.5
    mym = _nbr(mask.astype(np.float32), 0, -1) > 0.5

    axp = np.where(mxp, axp, a)
    axm = np.where(mxm, axm, a)
    ayp = np.where(myp, ayp, a)
    aym = np.where(mym, aym, a)

    d2x = (axp - 2 * a + axm) / (dx * dx)
    d2y = (ayp - 2 * a + aym) / (dy * dy)
    out = (d2x + anis * d2y).astype(np.float32)
    out[~mask] = 0.0
    return out


# -------------------------
# Physics blocks
# -------------------------
def simulate_pressure(k_norm: np.ndarray, mask: np.ndarray, well_ij: Tuple[int, int], q: np.ndarray, params: Dict[str, float]) -> List[np.ndarray]:
    nx, ny = k_norm.shape
    wi, wj = well_ij

    alpha_p = float(params["alpha_p"])
    anisD = float(params["anisD"])
    dx = float(params["dx"]); dy = float(params["dy"])
    p_diff_scale = float(params.get("p_diff_scale", 1.0))
    qp_amp = float(params.get("qp_amp", 1.0))
    sigma = float(params.get("src_sigma", 2.5))

    src = gaussian_source(nx, ny, wi, wj, sigma=sigma).astype(np.float32)
    src = np.where(mask, src, 0.0).astype(np.float32)

    p = np.zeros((nx, ny), dtype=np.float32)
    out: List[np.ndarray] = []

    for tt in range(len(q)):
        out.append(np.where(mask, p, np.nan).astype(np.float32))
        p = p + alpha_p * p_diff_scale * div_k_grad(p, np.where(mask, k_norm, 0.0), mask, dx, dy, anisD) \
              + (qp_amp * float(q[tt])) * src
        p[~mask] = 0.0

    return out


def thickness_to_mobile_sg(h: np.ndarray, hc: float, mob_exp: float) -> np.ndarray:
    h = np.maximum(h, 0.0).astype(np.float32)
    x = h / float(hc + 1e-12)
    sg = 1.0 - np.exp(-np.power(x, float(mob_exp)))
    return np.clip(sg, 0.0, 1.0).astype(np.float32)


def land_residual_from_max(Sg_max: np.ndarray, C_L: float, Sgr_max: float) -> np.ndarray:
    Sgr = (C_L * Sg_max) / (1.0 + C_L * Sg_max)
    return np.minimum(Sgr, Sgr_max).astype(np.float32)


def simulate_sg(p_list: List[np.ndarray], k_norm: np.ndarray, mask: np.ndarray, well_ij: Tuple[int, int], q: np.ndarray, params: Dict[str, float]) -> List[np.ndarray]:
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
    dx = float(params["dx"]); dy = float(params["dy"])
    nu = float(params["nu"])
    k_vel_scale = float(params.get("k_vel_scale", 1.0))
    sigma = float(params.get("src_sigma", 2.5))

    src = gaussian_source(nx, ny, wi, wj, sigma=sigma).astype(np.float32)
    src = np.where(mask, src, 0.0).astype(np.float32)

    h = np.zeros((nx, ny), dtype=np.float32)
    sg_max = np.zeros((nx, ny), dtype=np.float32)
    out: List[np.ndarray] = []

   dt_outer = 1.0

for tt in range(len(q)):
    p = np.where(np.isfinite(p_list[tt]), p_list[tt], 0.0).astype(np.float32)

    dpx, dpy = grad_center(p, mask, dx, dy)
    ux = -k_vel_scale * (k_eff * dpx)
    uy = -k_vel_scale * (k_eff * dpy)

    if nu != 0.0:
        ux = ux / (1.0 + nu * np.abs(ux))
        uy = uy / (1.0 + nu * np.abs(uy))

    # ---- stability / CFL ----
    umax = float(np.nanmax(np.abs(ux[mask]))) if mask.any() else 0.0
    vmax = float(np.nanmax(np.abs(uy[mask]))) if mask.any() else 0.0
    max_u = max(umax, vmax)

    # effective diffusion strength (very conservative but stable)
    kmax = float(np.nanmax(k_eff[mask])) if mask.any() else 0.0
    D_eff = float(D0 + abs(m_spread) * kmax)  # conservative

    dt_adv  = 0.45 * min(dx, dy) / (max_u + 1e-8)            # advection CFL
    dt_diff = 0.20 * min(dx*dx, dy*dy) / (D_eff + 1e-12)     # explicit diffusion stability

    dt_sub = min(dt_outer, dt_adv, dt_diff)
    nsub = int(np.ceil(dt_outer / dt_sub))
    nsub = max(1, nsub)
    dt = dt_outer / nsub

    qt = float(q[tt])
    inj = max(qt, 0.0)
    prod = -min(qt, 0.0)

    for _ in range(nsub):
        h_adv = upwind_advect(h, ux, uy, mask, dx, dy, dt)

        h_diff = D0 * laplace_masked(h_adv, mask, dx, dy, anisD)

        # your spread term (kept), but now stable due to substeps
        h_spread = m_spread * laplace_masked(k_eff * h_adv, mask, dx, dy, anisD)

        # sources scaled by dt
        h_new = h_adv + dt * (
            h_diff
            + h_spread
            + (src_amp * inj) * src
            - (prod_frac * prod) * src
        )

        h = np.where(mask, np.maximum(h_new, 0.0), 0.0).astype(np.float32)

    # convert thickness -> Sg and apply Land residual
    sg_m = thickness_to_mobile_sg(h + eps_h, hc=hc, mob_exp=mob_exp)
    sg_max = np.maximum(sg_max, sg_m)
    sg_r = land_residual_from_max(sg_max, C_L=C_L, Sgr_max=Sgr_max)
    sg_tot = np.maximum(sg_m, sg_r)

    out.append(np.where(mask, sg_tot, np.nan).astype(np.float32))


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
    if params is None:
        params = dict(DEFAULT_PARAMS)
    else:
        tmp = dict(DEFAULT_PARAMS)
        tmp.update(params)
        params = tmp

    t = np.array(t, dtype=np.float32)
    q = np.array(q, dtype=np.float32)
    if len(t) != len(q):
        raise ValueError(f"t and q must have same length. Got len(t)={len(t)} len(q)={len(q)}")

    _, k_norm, mask = prepare_phi_k(phi, k)

    if well_mode == "manual":
        if well_ij is None:
            raise ValueError("manual well_mode requires well_ij=(i,j)")
        wi, wj = choose_well_ij(k_norm, mask, "manual", ij=well_ij)
    else:
        wi, wj = choose_well_ij(k_norm, mask, well_mode)

    p_list = simulate_pressure(k_norm, mask, (wi, wj), q, params)

    sg_list = simulate_sg(p_list, k_norm, mask, (wi, wj), q, params)

    area = np.array([float(np.nansum(s > thr_area)) for s in sg_list], dtype=np.float32)
    r_eq = np.sqrt(area / np.pi).astype(np.float32)

    max_sg = np.array([float(np.nanmax(s)) if np.isfinite(s).any() else 0.0 for s in sg_list], dtype=np.float32)
    max_p = np.array([float(np.nanmax(p)) if np.isfinite(p).any() else 0.0 for p in p_list], dtype=np.float32)

    return ForwardResult(
        sg_list=sg_list,
        p_list=p_list if return_pressure else None,
        area=area,
        r_eq=r_eq,
        q=q,
        t=t,
        well_ij=(wi, wj),
        max_sg=max_sg,
        max_p=max_p,
    )

