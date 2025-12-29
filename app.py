# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from ve_core import DEFAULT_PARAMS, run_forward, choose_well_ij, prepare_phi_k

st.set_page_config(page_title="VE Forward Predictor (Diverse Data)", layout="wide")
st.title("VE + Darcy + Land (robust input): Forward plume predictor")


st.markdown(
    """
This version supports **diverse data**:

### NPZ
- Keys can be: `phi/k`, `poro/permx`, `PORO/PERMX`, `porosity/permeability`, etc.
- Arrays can be **2D** or **3D** (choose layer / mean / max).

Optional keys:
- `actnum` / `ACTNUM` mask
- `dx`, `dy` (cell size) to get radius/area in physical units

### Schedule CSV
Accepts either:
1) **Per-step**: columns `t`, `q`
2) **Segments**: columns `t_start`, `t_end`, `q` (auto-expanded & resampled)

You can also **resample to uniform dt** (recommended for smooth results).
"""
)

# -------------------------
# Helpers (diverse NPZ)
# -------------------------
PHI_KEYS = ["phi", "poro", "porosity", "PORO"]
K_KEYS = ["k", "permx", "perm", "permeability", "PERMX"]
ACT_KEYS = ["actnum", "ACTNUM", "mask", "MASK"]
DX_KEYS = ["dx", "DX"]
DY_KEYS = ["dy", "DY"]


def _find_key(data, candidates):
    for k in candidates:
        if k in data.files:
            return k
    # try case-insensitive
    lower_map = {kk.lower(): kk for kk in data.files}
    for k in candidates:
        if k.lower() in lower_map:
            return lower_map[k.lower()]
    return None


def load_npz_diverse(uploaded, mode_3d="layer", layer=0):
    data = np.load(uploaded, allow_pickle=True)

    kphi = _find_key(data, PHI_KEYS)
    kk = _find_key(data, K_KEYS)
    if kphi is None or kk is None:
        raise ValueError(f"NPZ must contain porosity + permeability keys. Found keys: {list(data.files)}")

    phi = np.array(data[kphi])
    perm = np.array(data[kk])

    # optional mask
    km = _find_key(data, ACT_KEYS)
    mask = None
    if km is not None:
        m = np.array(data[km])
        mask = (m > 0)

    # optional dx/dy
    dx = float(data[_find_key(data, DX_KEYS)]) if _find_key(data, DX_KEYS) else 1.0
    dy = float(data[_find_key(data, DY_KEYS)]) if _find_key(data, DY_KEYS) else 1.0

    # handle 3D -> 2D
    def to2d(a):
        if a.ndim == 2:
            return a
        if a.ndim != 3:
            raise ValueError(f"Array must be 2D or 3D, got shape={a.shape}")
        if mode_3d == "layer":
            k = int(np.clip(layer, 0, a.shape[0] - 1))
            return a[k, :, :]
        if mode_3d == "mean":
            return np.nanmean(a, axis=0)
        if mode_3d == "max":
            return np.nanmax(a, axis=0)
        raise ValueError("mode_3d must be layer/mean/max")

    phi2 = to2d(phi).astype(np.float32)
    k2 = to2d(perm).astype(np.float32)

    mask2 = None
    if mask is not None:
        mask2 = to2d(mask).astype(bool)

    meta = {"phi_key": kphi, "k_key": kk, "dx": dx, "dy": dy, "orig_phi_shape": tuple(phi.shape), "orig_k_shape": tuple(perm.shape)}
    return phi2, k2, mask2, meta


# -------------------------
# Helpers (diverse schedule)
# -------------------------
def load_schedule_diverse(uploaded):
    df = pd.read_csv(uploaded)
    df.columns = [c.lower().strip() for c in df.columns]

    if ("t" in df.columns) and ("q" in df.columns):
        t = df["t"].to_numpy(dtype=np.float32)
        q = df["q"].to_numpy(dtype=np.float32)
        return t, q

    # segment format
    need = {"t_start", "t_end", "q"}
    if need.issubset(set(df.columns)):
        seg = df[["t_start", "t_end", "q"]].copy()
        seg["t_start"] = seg["t_start"].astype(np.float32)
        seg["t_end"] = seg["t_end"].astype(np.float32)
        seg["q"] = seg["q"].astype(np.float32)
        return seg, None

    raise ValueError("Schedule CSV must have either (t,q) OR (t_start,t_end,q).")


def expand_segments_to_steps(seg_df, dt=1.0):
    tmin = float(seg_df["t_start"].min())
    tmax = float(seg_df["t_end"].max())
    t = np.arange(tmin, tmax + 1e-6, dt, dtype=np.float32)
    q = np.zeros_like(t, dtype=np.float32)

    for _, r in seg_df.iterrows():
        m = (t >= float(r["t_start"])) & (t < float(r["t_end"]))
        q[m] = float(r["q"])
    return t, q


def resample_steps(t, q, dt=1.0):
    t = np.asarray(t, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)

    t0, t1 = float(t.min()), float(t.max())
    tg = np.arange(t0, t1 + 1e-6, dt, dtype=np.float32)
    # piecewise constant hold-last
    qg = np.interp(tg, t, q, left=q[0], right=q[-1]).astype(np.float32)
    return tg, qg


def fig_imshow(arr, title, vmin=None, vmax=None):
    fig = plt.figure(figsize=(7.2, 4.6))
    plt.title(title)
    if vmin is None: vmin = float(np.nanmin(arr))
    if vmax is None: vmax = float(np.nanmax(arr))
    im = plt.imshow(arr, origin="lower", vmin=vmin, vmax=vmax)
    plt.xlabel("j")
    plt.ylabel("i")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def fig_schedule(t, q, tidx=None):
    fig = plt.figure(figsize=(7.2, 4.6))
    plt.title("Schedule q(t)")
    plt.plot(t, q)
    if tidx is not None and 0 <= tidx < len(t):
        plt.axvline(t[tidx], linestyle="--")
    plt.xlabel("t")
    plt.ylabel("q")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def fig_timeseries(t, y, ylab, title):
    fig = plt.figure(figsize=(7.2, 4.6))
    plt.title(title)
    plt.plot(t, y)
    plt.xlabel("t")
    plt.ylabel(ylab)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("1) Upload")
    up_npz = st.file_uploader("phi/k NPZ (2D or 3D)", type=["npz"])
    up_csv = st.file_uploader("schedule CSV", type=["csv"])

    st.divider()
    st.header("2) 3D handling (if NPZ is 3D)")
    mode_3d = st.selectbox("3D -> 2D reduction", ["layer", "mean", "max"], index=0)
    layer = st.number_input("Layer index (if mode=layer)", value=0, step=1)

    st.divider()
    st.header("3) Well")
    well_mode = st.selectbox("Well placement", ["max_k", "center", "manual"], index=1)
    manual_i = st.number_input("manual i", value=0, step=1)
    manual_j = st.number_input("manual j", value=0, step=1)

    st.divider()
    st.header("4) Schedule cleanup (recommended)")
    q_scale = st.number_input("q scale factor", value=1.0)
    normalize_q = st.checkbox("Normalize max(|q|)=1", value=False)

    schedule_mode = st.selectbox("Schedule mode", ["auto", "force_resample_dt", "force_expand_segments_dt"], index=0)
    dt_uniform = st.number_input("Uniform dt (for resample/expand)", value=1.0, min_value=0.01)

    st.divider()
    st.header("5) Smoothness + metrics")
    thr_area = st.slider("Area threshold (Sg > thr)", 0.0, 0.5, 0.05, 0.01)
    sg_smooth_sigma = st.slider("Output smoothing sigma", 0.0, 4.0, 1.2, 0.1)
    logk_smooth_sigma = st.slider("Smooth log10(k) sigma", 0.0, 4.0, 1.6, 0.1)
    front_w = st.slider("Front thickness w", 0.5, 6.0, 2.2, 0.1)

    st.divider()
    run_btn = st.button("Run forward prediction", type="primary")


if up_npz is None or up_csv is None:
    st.info("Upload a phi/k NPZ and a schedule CSV to begin.")
    st.stop()

# -------------------------
# Load NPZ
# -------------------------
try:
    phi, k, act_mask, meta_npz = load_npz_diverse(up_npz, mode_3d=mode_3d, layer=int(layer))
except Exception as e:
    st.error(f"NPZ load failed: {e}")
    st.stop()

# -------------------------
# Load schedule
# -------------------------
try:
    sched = load_schedule_diverse(up_csv)
    if isinstance(sched[0], pd.DataFrame):  # segments
        seg_df = sched[0]
        if schedule_mode == "force_expand_segments_dt":
            t, q = expand_segments_to_steps(seg_df, dt=float(dt_uniform))
        else:
            # auto-expand segments with dt=1 by default
            t, q = expand_segments_to_steps(seg_df, dt=1.0)
    else:
        t, q = sched
        if schedule_mode == "force_resample_dt":
            t, q = resample_steps(t, q, dt=float(dt_uniform))
except Exception as e:
    st.error(f"Schedule load failed: {e}")
    st.stop()

q = q.astype(np.float32) * np.float32(q_scale)
if normalize_q:
    m = float(np.max(np.abs(q))) if q.size else 1.0
    if m > 0:
        q = (q / m).astype(np.float32)

# -------------------------
# Preview
# -------------------------
st.caption(f"NPZ keys used: phi={meta_npz['phi_key']} | k={meta_npz['k_key']} | orig phi shape={meta_npz['orig_phi_shape']} | orig k shape={meta_npz['orig_k_shape']}")
st.caption(f"dx={meta_npz['dx']}, dy={meta_npz['dy']} (used for area/radius units)")

colA, colB = st.columns(2)
with colA:
    st.subheader("phi (porosity)")
    st.pyplot(fig_imshow(phi, f"phi | {phi.shape}"))
with colB:
    st.subheader("k (permeability)")
    st.pyplot(fig_imshow(k, f"k | {k.shape}"))

# Select well for display
try:
    _, k_norm, mask = prepare_phi_k(phi, k, mask=act_mask)
    well_ij = (int(manual_i), int(manual_j)) if well_mode == "manual" else None
    wi, wj = choose_well_ij(k_norm, mask, well_mode, ij=well_ij)
    st.caption(f"Selected well (i,j)=({wi},{wj}) | active cells={int(mask.sum())}")
except Exception as e:
    st.error(f"Well selection failed: {e}")
    st.stop()

st.subheader("Schedule")
st.pyplot(fig_schedule(t, q, tidx=0))

if not run_btn:
    st.stop()

# -------------------------
# Run model
# -------------------------
params = dict(DEFAULT_PARAMS)
params["sg_smooth_sigma"] = float(sg_smooth_sigma)
params["gauss_smooth_sigma"] = float(logk_smooth_sigma)
params["gauss_w"] = float(front_w)

with st.spinner("Running forward model..."):
    try:
        res = run_forward(
            phi=phi,
            k=k,
            t=t,
            q=q,
            params=params,
            well_mode=well_mode,
            well_ij=(int(manual_i), int(manual_j)) if well_mode == "manual" else None,
            mask=act_mask,
            return_pressure=True,
            thr_area=float(thr_area),
            dx=float(meta_npz["dx"]),
            dy=float(meta_npz["dy"]),
        )
    except Exception as e:
        st.error(f"Run failed: {e}")
        st.stop()

st.success("Done.")

Nt = len(res.sg_list)
tidx = st.slider("Select timestep (tidx)", 0, max(0, Nt - 1), 0)

left, right = st.columns(2)
with left:
    st.subheader(f"Sg predicted | tidx={tidx}")
    st.pyplot(fig_imshow(res.sg_list[tidx], f"Sg predicted | tidx={tidx}", vmin=0.0, vmax=1.0))
with right:
    st.subheader(f"Pressure surrogate | tidx={tidx}")
    p = res.p_list[tidx] if res.p_list is not None else None
    st.pyplot(fig_imshow(p, f"p predicted | tidx={tidx}"))


col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("q(t)")
    st.pyplot(fig_schedule(res.t, res.q, tidx=tidx))
with col2:
    st.subheader("Plume area")
    st.pyplot(fig_timeseries(res.t, res.area, "area (dx*dy units)", "Area time series"))
with col3:
    st.subheader("Equivalent radius")
    st.pyplot(fig_timeseries(res.t, res.r_eq, "r_eq (same length unit as dx/dy)", "Equivalent radius"))


st.subheader("Download results")
out_npz = io.BytesIO()
sg_stack = np.stack([np.nan_to_num(s, nan=0.0) for s in res.sg_list], axis=0).astype(np.float32)
np.savez_compressed(out_npz, sg=sg_stack, t=res.t, q=res.q, area=res.area, r_eq=res.r_eq, dx=res.meta["dx"], dy=res.meta["dy"])
st.download_button("Download predicted Sg (NPZ)", data=out_npz.getvalue(), file_name="sg_predicted.npz")

out_csv = pd.DataFrame({"t": res.t, "q": res.q, "area": res.area, "r_eq": res.r_eq}).to_csv(index=False).encode("utf-8")
st.download_button("Download time series (CSV)", data=out_csv, file_name="plume_timeseries.csv")
