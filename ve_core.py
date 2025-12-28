import io
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
- `t` : timestep index (0..Nt-1) or time
- `q` : signed rate per timestep (positive=injection, negative=production)
"""
)

# -------------------------
# Sample generator (for testing)
# -------------------------
def make_sample_phi_k(nx=80, ny=120, seed=7):
    rng = np.random.default_rng(seed)
    ii = np.linspace(0, 1, nx)[:, None]
    jj = np.linspace(0, 1, ny)[None, :]
    phi = 0.22 + 0.06*np.sin(2*np.pi*ii) * np.cos(2*np.pi*jj)
    phi += 0.01*rng.standard_normal((nx, ny))
    phi = np.clip(phi, 0.05, 0.35).astype(np.float32)

    logk = 2.2 + 0.5*np.sin(2*np.pi*jj) + 0.3*np.cos(2*np.pi*ii)
    logk += 0.08*rng.standard_normal((nx, ny))
    k = (10**logk).astype(np.float32)

    # inactive mask (a "cut-out" region)
    mask = np.ones((nx, ny), dtype=bool)
    mask[:6, :10] = False
    mask[-8:, -12:] = False
    phi[~mask] = np.nan
    k[~mask] = np.nan
    return phi, k

def make_sample_schedule(Nt=80):
    t = np.arange(Nt, dtype=np.float32)
    q = np.zeros(Nt, dtype=np.float32)
    q[:20] = 1.0
    q[20:30] = np.linspace(1.0, 0.0, 10, dtype=np.float32)
    q[30:40] = np.linspace(0.0, -0.5, 10, dtype=np.float32)
    q[40:60] = np.linspace(-0.5, 0.0, 20, dtype=np.float32)
    q[60:] = 0.0
    return t, q

# -------------------------
# Helpers
# -------------------------
def load_npz(uploaded):
    data = np.load(uploaded)
    if "phi" not in data or "k" not in data:
        raise ValueError("NPZ must contain keys: phi and k")
    meta = {k: data[k] for k in data.files if k not in ("phi", "k")}
    return data["phi"], data["k"], meta

def load_schedule_csv(uploaded):
    df = pd.read_csv(uploaded)
    df.columns = [c.lower().strip() for c in df.columns]
    if "t" not in df.columns or "q" not in df.columns:
        raise ValueError("CSV must have columns: t, q")
    t = df["t"].to_numpy(dtype=np.float32)
    q = df["q"].to_numpy(dtype=np.float32)
    return t, q

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
# Sidebar inputs
# -------------------------
with st.sidebar:
    st.header("Quick test (no upload)")
    if st.button("Use built-in sample inputs"):
        phi_s, k_s = make_sample_phi_k()
        t_s, q_s = make_sample_schedule()
        sample_npz = io.BytesIO()
        np.savez_compressed(sample_npz, phi=phi_s, k=k_s)
        st.download_button("Download sample_phi_k.npz", sample_npz.getvalue(), file_name="sample_phi_k.npz")
        sample_csv = pd.DataFrame({"t": t_s, "q": q_s}).to_csv(index=False).encode("utf-8")
        st.download_button("Download sample_schedule.csv", sample_csv, file_name="sample_schedule.csv")

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
        for kkey in list(params.keys()):
            # keep UI sane: show only “core” params, not the whole dict if you want
            params[kkey] = st.number_input(kkey, value=float(params[kkey]))

    st.divider()
    run_btn = st.button("Run forward prediction", type="primary")

# -------------------------
# Main logic
# -------------------------
if up_npz is None or up_csv is None:
    st.info("Upload a phi/k NPZ and a schedule CSV to begin (or use the sample downloads in the sidebar).")
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

# quick preview
colA, colB = st.columns(2)
with colA:
    st.subheader("Input: porosity (phi)")
    st.pyplot(fig_imshow(phi, f"phi | shape={phi.shape}"))
with colB:
    st.subheader("Input: permeability (k)")
    st.pyplot(fig_imshow(k, f"k | shape={k.shape}"))

with st.expander("Debug: NPZ meta + masks", expanded=False):
    st.write("NPZ meta keys:", list(meta.keys()))
    try:
        _, k_norm, mask = prepare_phi_k(phi, k)
        st.write("Active cells:", int(mask.sum()), " / ", int(mask.size))
        st.pyplot(fig_imshow(mask.astype(np.float32), "active mask (1=active)", vmin=0.0, vmax=1.0))
        st.pyplot(fig_imshow(np.log10(np.where(mask, np.maximum(k_norm, 1e-12), np.nan)), "log10(k_norm) (masked)"))
    except Exception as e:
        st.warning(f"prepare_phi_k failed: {e}")

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
    if p is None:
        st.write("Pressure output disabled.")
    else:
        st.pyplot(fig_imshow(p, f"p predicted | tidx={tidx}"))

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
