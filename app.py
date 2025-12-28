import io
import os
import sys
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Make sure repo root is importable ---
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# --- Import ve_core safely (show full error in Streamlit if it fails) ---
try:
    import ve_core
    DEFAULT_PARAMS = ve_core.DEFAULT_PARAMS
    run_forward = ve_core.run_forward
    choose_well_ij = ve_core.choose_well_ij
    prepare_phi_k = ve_core.prepare_phi_k
except Exception:
    st.error("Failed to import ve_core.py. Full traceback:")
    st.code(traceback.format_exc())
    st.stop()


st.set_page_config(page_title="VE+Darcy+Land Forward Predictor", layout="wide")
st.title("VE + Darcy + Land: Forward plume prediction (paper model)")

st.markdown(
    """
Upload **phi/k NPZ** and an **injection schedule CSV**, then run the paper model forward **without sg_obs**.

**Expected NPZ keys**
- `phi` : 2D array (nx, ny)
- `k`   : 2D array (nx, ny)  (PERMX recommended)

**Expected schedule CSV columns**
- `t` : timestep index (0..Nt-1) or time
- `q` : signed rate per timestep (positive=injection, negative=production)
"""
)

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def load_npz_bytes(b: bytes):
    bio = io.BytesIO(b)
    data = np.load(bio)
    if "phi" not in data or "k" not in data:
        raise ValueError("NPZ must contain keys: phi and k")
    phi = data["phi"]
    k = data["k"]
    meta = {key: data[key] for key in data.files if key not in ("phi", "k")}
    return phi, k, meta

@st.cache_data(show_spinner=False)
def load_schedule_csv_bytes(b: bytes):
    bio = io.BytesIO(b)
    df = pd.read_csv(bio)
    df.columns = [c.lower().strip() for c in df.columns]
    if "t" not in df.columns or "q" not in df.columns:
        raise ValueError("CSV must have columns: t, q")
    t = df["t"].to_numpy(dtype=np.float32)
    q = df["q"].to_numpy(dtype=np.float32)
    return t, q

def fig_imshow(arr, title, vmin=None, vmax=None):
    fig = plt.figure(figsize=(7.2, 4.6))
    plt.title(title)

    a = np.array(arr, copy=False)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")  # NaN/inactive

    if vmin is None:
        vmin = float(np.nanmin(a)) if np.isfinite(a).any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(a)) if np.isfinite(a).any() else 1.0

    im = plt.imshow(a, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
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

def make_sample_phi_k(nx=110, ny=45):
    phi = np.full((nx, ny), 0.25, dtype=np.float32)
    k = np.full((nx, ny), 100.0, dtype=np.float32)
    # add smooth heterogeneity
    ii = np.linspace(-1, 1, nx, dtype=np.float32)[:, None]
    jj = np.linspace(-1, 1, ny, dtype=np.float32)[None, :]
    blob = np.exp(-3.0 * (ii**2 + (1.5*jj)**2)).astype(np.float32)
    k = k * (1.0 + 3.0 * blob)
    # inactive border
    phi[:8, :] = np.nan
    phi[-6:, :] = np.nan
    phi[:, :6] = np.nan
    phi[:, -6:] = np.nan
    k[~np.isfinite(phi)] = np.nan
    return phi, k

def make_sample_schedule(Nt=80):
    t = np.arange(Nt, dtype=np.float32)
    q = np.zeros(Nt, dtype=np.float32)
    q[:20] = 1.0
    q[20:30] = np.linspace(1.0, 0.0, 10, dtype=np.float32)
    q[30:40] = np.linspace(0.0, -0.5, 10, dtype=np.float32)
    q[40:60] = np.linspace(-0.5, 0.0, 20, dtype=np.float32)
    return t, q


# -------------------------
# Sidebar inputs
# -------------------------
with st.sidebar:
    st.header("0) Quick test (optional)")
    if st.button("Use built-in sample inputs"):
        phi_s, k_s = make_sample_phi_k()
        t_s, q_s = make_sample_schedule()
        buf = io.BytesIO()
        np.savez_compressed(buf, phi=phi_s, k=k_s)
        st.session_state["_npz_bytes"] = buf.getvalue()
        st.session_state["_csv_bytes"] = pd.DataFrame({"t": t_s, "q": q_s}).to_csv(index=False).encode("utf-8")

    st.divider()
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
    normalize_q = st.checkbox("Normalize q to max(|q|)=1", value=False)

    st.divider()
    st.header("4) Output / metrics")
    thr_area = st.slider("Area threshold (Sg > thr)", 0.0, 0.5, 0.05, 0.01)

    st.divider()
    st.header("5) Paper parameters")
    use_defaults = st.checkbox("Use paper-calibrated defaults", value=True)

    params = dict(DEFAULT_PARAMS)
    if not use_defaults:
        for key in list(params.keys()):
            params[key] = st.number_input(key, value=float(params[key]))

    st.divider()
    run_btn = st.button("Run forward prediction", type="primary")


# -------------------------
# Read inputs
# -------------------------
npz_bytes = None
csv_bytes = None

if up_npz is not None:
    npz_bytes = up_npz.getvalue()
elif "_npz_bytes" in st.session_state:
    npz_bytes = st.session_state["_npz_bytes"]

if up_csv is not None:
    csv_bytes = up_csv.getvalue()
elif "_csv_bytes" in st.session_state:
    csv_bytes = st.session_state["_csv_bytes"]

if npz_bytes is None or csv_bytes is None:
    st.info("Upload a phi/k NPZ and a schedule CSV (or click 'Use built-in sample inputs').")
    st.stop()

try:
    phi, k, meta = load_npz_bytes(npz_bytes)
    t, q = load_schedule_csv_bytes(csv_bytes)

    q = q.astype(np.float32) * np.float32(q_scale)
    if normalize_q:
        m = float(np.max(np.abs(q))) if q.size else 1.0
        if m > 0:
            q = (q / m).astype(np.float32)

except Exception as e:
    st.error(f"Failed to read inputs: {e}")
    st.stop()


# -------------------------
# Preview inputs
# -------------------------
colA, colB = st.columns(2)
with colA:
    st.subheader("Input: porosity (phi)")
    st.pyplot(fig_imshow(phi, f"phi | shape={phi.shape}"))
with colB:
    st.subheader("Input: permeability (k)")
    # show log10(k) for nicer viewing if needed
    k_show = np.where(np.isfinite(k) & (k > 0), np.log10(k), np.nan).astype(np.float32)
    st.pyplot(fig_imshow(k_show, f"log10(k) | shape={k.shape}"))

# compute well coordinate for display
try:
    _, k_norm, mask = prepare_phi_k(phi, k)
    if well_mode == "manual":
        well_ij = (int(manual_i), int(manual_j))
    else:
        well_ij = None
    wi, wj = choose_well_ij(k_norm, mask, well_mode, ij=well_ij)
    st.caption(f"Selected well (i,j)=({wi},{wj}) | active cells={int(mask.sum())}")
except Exception as e:
    st.error(f"Input fields invalid: {e}")
    st.stop()

st.subheader("Injection schedule")
st.pyplot(fig_schedule(t, q, tidx=0))

if not run_btn:
    st.stop()


# -------------------------
# Run model
# -------------------------
with st.spinner("Running VE+Darcy+Land forward model..."):
    try:
        res = run_forward(
            phi=phi,
            k=k,
            t=t,
            q=q,
            params=params,
            well_mode=well_mode,
            well_ij=(int(manual_i), int(manual_j)) if well_mode == "manual" else None,
            return_pressure=True,
            thr_area=float(thr_area),
        )
    except Exception as e:
        st.error("Run failed. Full traceback:")
        st.code(traceback.format_exc())
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
np.savez_compressed(out_npz, sg=sg_stack, t=res.t, q=res.q, area=res.area, r_eq=res.r_eq)
st.download_button("Download predicted Sg (NPZ)", data=out_npz.getvalue(), file_name="sg_predicted.npz")

out_csv = pd.DataFrame({"t": res.t, "q": res.q, "area": res.area, "r_eq": res.r_eq}).to_csv(index=False).encode("utf-8")
st.download_button("Download time series (CSV)", data=out_csv, file_name="plume_timeseries.csv")
