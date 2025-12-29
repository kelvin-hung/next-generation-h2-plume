import io
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from ve_core import DEFAULT_PARAMS, run_forward, choose_well_ij, prepare_phi_k

st.set_page_config(page_title="VE+Darcy+Land Forward Predictor", layout="wide")
st.title("VE + Darcy + Land: Forward plume prediction (paper model)")

st.markdown(
    """
Upload **phi/k NPZ** and an **injection schedule CSV**, then run the paper model forward **without sg_obs**.

**Expected NPZ keys**
- `phi` : 2D array (nx, ny)
- `k`   : 2D array (nx, ny)  (PERMX recommended)

**Expected schedule CSV columns**
- `t` : timestep index or time (monotonic)
- `q` : signed rate per step (positive=injection, negative=production)

**If results look "too small" on Norne**
1) Turn on **Normalize q** (default).
2) Use well placement **max_k**.
3) If still tiny, increase `qp_amp` and `src_amp` (try 2×, 5×, 10×).
"""
)

# -------------------------
# Helpers
# -------------------------
def load_npz(uploaded) -> Tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(uploaded, allow_pickle=True)
    if "phi" not in data or "k" not in data:
        raise ValueError("NPZ must contain keys: phi and k")
    meta = {k: data[k] for k in data.files if k not in ("phi", "k")}
    phi = np.asarray(data["phi"], dtype=np.float32)
    k = np.asarray(data["k"], dtype=np.float32)
    if phi.ndim != 2 or k.ndim != 2:
        raise ValueError(f"phi and k must be 2D arrays, got phi.ndim={phi.ndim}, k.ndim={k.ndim}")
    if phi.shape != k.shape:
        raise ValueError(f"phi and k must have same shape, got {phi.shape} vs {k.shape}")
    return phi, k, meta

def load_schedule_csv(uploaded) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(uploaded)
    df.columns = [c.lower().strip() for c in df.columns]
    if "t" not in df.columns or "q" not in df.columns:
        raise ValueError("CSV must have columns: t, q")
    t = df["t"].to_numpy(dtype=np.float32)
    q = df["q"].to_numpy(dtype=np.float32)
    m = np.isfinite(t) & np.isfinite(q)
    t, q = t[m], q[m]
    if t.size < 2:
        raise ValueError("Schedule too short after cleaning.")
    s = np.argsort(t)
    return t[s], q[s]

def fig_imshow(arr, title, vmin=None, vmax=None):
    fig = plt.figure(figsize=(7.2, 4.6))
    plt.title(title)
    if vmin is None:
        vmin = float(np.nanmin(arr))
    if vmax is None:
        vmax = float(np.nanmax(arr))
    im = plt.imshow(arr, origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
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
    st.header("1) Upload inputs")
    up_npz = st.file_uploader("phi/k NPZ", type=["npz"])
    up_csv = st.file_uploader("schedule CSV", type=["csv"])

    st.divider()
    st.header("2) Well location")
    well_mode = st.selectbox("Well placement", ["max_k", "center", "manual"], index=0)
    manual_i = st.number_input("manual i", value=0, step=1)
    manual_j = st.number_input("manual j", value=0, step=1)

    st.divider()
    st.header("3) Schedule scaling")
    q_scale = st.number_input("Schedule scale factor (multiplies q)", value=1.0)
    normalize_q = st.checkbox("Normalize q to max(|q|)=1", value=True)

    st.divider()
    st.header("4) Output / metrics")
    thr_area = st.slider("Area threshold (Sg > thr)", 0.0, 0.5, 0.05, 0.01)

    st.divider()
    st.header("5) Paper parameters")
    use_defaults = st.checkbox("Use paper-calibrated defaults", value=True)

    params = dict(DEFAULT_PARAMS)
    if not use_defaults:
        st.caption("For Norne, try increasing qp_amp + src_amp (2×–10×).")
        params["qp_amp"] = st.number_input("qp_amp", value=float(params["qp_amp"]))
        params["src_amp"] = st.number_input("src_amp", value=float(params["src_amp"]))
        # keep others available if you want
        with st.expander("Other parameters"):
            for key in sorted(params.keys()):
                if key in ("qp_amp","src_amp"):
                    continue
                params[key] = st.number_input(key, value=float(params[key]))

    st.divider()
    run_btn = st.button("Run forward prediction", type="primary")

# -------------------------
# Main
# -------------------------
if up_npz is None or up_csv is None:
    st.info("Upload a phi/k NPZ and a schedule CSV to begin.")
    st.stop()

try:
    phi, k, meta = load_npz(up_npz)
    t, q = load_schedule_csv(up_csv)

    q = q.astype(np.float32) * np.float32(q_scale)
    if normalize_q:
        m = float(np.max(np.abs(q))) if q.size else 1.0
        if m > 0:
            q = (q / m).astype(np.float32)

except Exception as e:
    st.error(f"Failed to read inputs: {e}")
    st.stop()

try:
    phi01, k01, mask = prepare_phi_k(phi, k)
except Exception as e:
    st.error(f"Input fields invalid: {e}")
    st.stop()

colA, colB = st.columns(2)
with colA:
    st.subheader("Input: porosity (phi)")
    st.pyplot(fig_imshow(phi, f"phi | shape={phi.shape}"))
with colB:
    st.subheader("Input: permeability (k)")
    st.pyplot(fig_imshow(k, f"k | shape={k.shape}"))

colC, colD = st.columns(2)
with colC:
    st.subheader("Prepared: phi01")
    st.pyplot(fig_imshow(phi01, "phi01 (inactive=NaN)", vmin=0.0, vmax=1.0))
with colD:
    st.subheader("Prepared: k01")
    st.pyplot(fig_imshow(k01, "k01 (inactive=NaN)", vmin=0.0, vmax=1.0))

try:
    user_ij = (int(manual_i), int(manual_j)) if well_mode == "manual" else None
    wi, wj = choose_well_ij(k01, mask, well_mode, ij=user_ij)
    st.caption(f"Selected well (i,j)=({wi},{wj}) | active cells={int(mask.sum())}")
except Exception as e:
    st.error(f"Well selection failed: {e}")
    st.stop()

st.subheader("Injection schedule")
st.pyplot(fig_schedule(t, q, tidx=0))

if not run_btn:
    st.stop()

with st.spinner("Running VE+Darcy+Land forward model..."):
    try:
        res = run_forward(
            phi=phi,
            k=k,
            t=t,
            q=q,
            params=None if use_defaults else params,
            well_mode=well_mode,
            well_ij=user_ij,
            return_pressure=True,
            thr_area=float(thr_area),
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
    if res.p_list is None:
        st.write("Pressure output disabled.")
    else:
        st.pyplot(fig_imshow(res.p_list[tidx], f"p predicted | tidx={tidx}"))

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("q(t)")
    st.pyplot(fig_schedule(res.t, res.q, tidx=tidx))
with col2:
    st.subheader("Plume area")
    st.pyplot(fig_timeseries(res.t, res.area, "area (cells)", "Area time series"))
with col3:
    st.subheader("Equivalent radius")
    st.pyplot(fig_timeseries(res.t, res.r_eq, "r_eq (cells)", "Equivalent radius time series"))

st.subheader("Download results")
out_npz = io.BytesIO()
sg_stack = np.stack([np.nan_to_num(s, nan=0.0) for s in res.sg_list], axis=0).astype(np.float32)
np.savez_compressed(out_npz, sg=sg_stack, t=res.t, q=res.q, area=res.area, r_eq=res.r_eq, well_ij=np.array(res.well_ij))
st.download_button("Download predicted Sg (NPZ)", data=out_npz.getvalue(), file_name="sg_predicted.npz")

out_csv = pd.DataFrame({"t": res.t, "q": res.q, "area": res.area, "r_eq": res.r_eq}).to_csv(index=False).encode("utf-8")
st.download_button("Download time series (CSV)", data=out_csv, file_name="plume_timeseries.csv")
