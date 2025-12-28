import os, sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import streamlit as st

from ve_core import (
    read_npz_phi_k,
    read_schedule_csv,
    ForwardParams,
    run_forward,
    fig_imshow,
    fig_line,
    make_zip_bytes,
)

st.set_page_config(page_title="VE Forward Predictor", layout="wide")
st.title("VE Forward Predictor (Upload phi/k NPZ + schedule CSV → Sg plume)")

st.sidebar.header("Upload")
npz_file = st.sidebar.file_uploader("NPZ file with arrays: phi, k", type=["npz"])
csv_file = st.sidebar.file_uploader("Schedule CSV with columns: t, q", type=["csv"])

st.sidebar.header("Run settings")
Nt = st.sidebar.slider("Nt (timesteps)", 20, 400, 80, 10)
dt = st.sidebar.number_input("dt", value=1.0, format="%.6f")
dx = st.sidebar.number_input("dx", value=1.0, format="%.6f")
dy = st.sidebar.number_input("dy", value=1.0, format="%.6f")
thr = st.sidebar.slider("Plume threshold (Sg ≥ thr)", 0.0, 0.5, 0.05, 0.01)

st.sidebar.header("Pressure driver")
mu = st.sidebar.number_input("mu", value=1.0, format="%.6f")
ct = st.sidebar.number_input("ct", value=1e-5, format="%.8f")
p_substeps = st.sidebar.slider("pressure substeps", 1, 10, 2, 1)

st.sidebar.header("VE knobs")
D0 = st.sidebar.number_input("D0", value=0.10, format="%.4f")
anisD = st.sidebar.slider("anisD", 0.5, 2.0, 1.0, 0.05)
vel_scale = st.sidebar.number_input("vel_scale", value=1.0, format="%.4f")

src_amp = st.sidebar.number_input("src_amp", value=0.08, format="%.4f")
rad_w = st.sidebar.slider("rad_w (cells)", 0.5, 10.0, 2.5, 0.1)
prod_frac = st.sidebar.slider("prod_frac (if q<0)", 0.0, 1.0, 0.0, 0.01)

Swr = st.sidebar.slider("Swr", 0.0, 0.6, 0.10, 0.01)
Sgr_max = st.sidebar.slider("Sgr_max", 0.0, 0.8, 0.20, 0.01)
C_L = st.sidebar.slider("C_L (Land)", 0.01, 1.0, 0.15, 0.01)

mob_exp = st.sidebar.slider("mob_exp", 0.5, 3.0, 1.2, 0.05)
hc = st.sidebar.slider("hc (k coupling)", 0.0, 0.5, 0.10, 0.01)

run_btn = st.sidebar.button("🚀 Run forward simulation")

st.sidebar.markdown("---")
include_snaps = st.sidebar.checkbox("Include snapshot PNGs in ZIP", value=True)
snap_count = st.sidebar.slider("snapshot PNG count", 1, 20, 6, 1)

if npz_file is None or csv_file is None:
    st.info("Upload NPZ (phi,k) and schedule CSV (t,q) to start.")
    st.stop()

# Load inputs
try:
    out = read_npz_phi_k(npz_file)
if isinstance(out, tuple) and len(out) == 3:
    phi, k, actmask = out
elif isinstance(out, tuple) and len(out) == 2:
    phi, k = out
    actmask = np.isfinite(phi) & np.isfinite(k)
    # sanitize for solver safety
    phi = np.where(actmask, phi, 1.0).astype(np.float32)
    k   = np.where(actmask, k, 0.0).astype(np.float32)
else:
    raise RuntimeError("read_npz_phi_k must return (phi,k) or (phi,k,actmask)")
    t_sched, q_sched = read_schedule_csv(csv_file)
except Exception as e:
    st.error(str(e))
    st.stop()

nx, ny = phi.shape
active_pct = float(actmask.mean() * 100.0)

st.write(f"Loaded `phi`/`k` shape **{phi.shape}** | schedule points: **{len(t_sched)}**")
st.write(f"Active cells: **{active_pct:.1f}%** (inactive are handled safely).")

# Well location
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    wi = st.number_input("well i (row)", 0, nx - 1, nx // 2, 1)
with col2:
    wj = st.number_input("well j (col)", 0, ny - 1, ny // 2, 1)
with col3:
    st.caption("Well is the source/sink location for q(t).")

# Quicklook maps
with st.expander("Quicklook: phi and k maps", expanded=False):
    # show inactive as NaN for display
    phi_plot = np.where(actmask, phi, np.nan).astype(np.float32)
    k_plot = np.where(actmask, k, np.nan).astype(np.float32)

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(fig_imshow(phi_plot, title="phi (masked inactive)"))
    with c2:
        k_disp = np.log10(np.maximum(k_plot, 1e-12))
        st.pyplot(fig_imshow(k_disp, title="log10(k) (masked inactive)"))

prm = ForwardParams(
    dx=float(dx), dy=float(dy), dt=float(dt), Nt=int(Nt),
    mu=float(mu), ct=float(ct), p_substeps=int(p_substeps),
    D0=float(D0), anisD=float(anisD), vel_scale=float(vel_scale),
    src_amp=float(src_amp), rad_w=float(rad_w), prod_frac=float(prod_frac),
    Swr=float(Swr), Sgr_max=float(Sgr_max), C_L=float(C_L),
    mob_exp=float(mob_exp), hc=float(hc),
)

if "outputs" not in st.session_state:
    st.session_state["outputs"] = None

if run_btn:
    with st.spinner("Running forward simulation..."):
        outputs = run_forward(phi, k, t_sched, q_sched, (int(wi), int(wj)), prm, thr=float(thr))
    st.session_state["outputs"] = outputs
    st.success("Done!")

outputs = st.session_state["outputs"]
if outputs is None:
    st.warning("Click **Run forward simulation** in the sidebar.")
    st.stop()

sg_list = outputs["sg_list"]
t_grid = outputs["t_grid"]
q_grid = outputs["q_grid"]

st.subheader("Predicted plume")
tidx = st.slider("tidx", 0, sg_list.shape[0] - 1, min(10, sg_list.shape[0] - 1), 1)

cA, cB = st.columns([1, 1])
with cA:
    st.pyplot(fig_imshow(sg_list[tidx], title=f"Sg predicted | tidx={tidx}", vmin=0, vmax=1))
with cB:
    fig = fig_line(t_grid, q_grid, xlabel="t", ylabel="q", title="Schedule q(t)")
    ax = fig.axes[0]
    ax.axvline(float(t_grid[tidx]), linestyle="--")
    st.pyplot(fig)

st.subheader("Time series")
c1, c2 = st.columns(2)
with c1:
    st.pyplot(fig_line(t_grid, outputs["area_ts"], xlabel="t", ylabel="area (cells)", title=f"area(t) for Sg ≥ {thr:.2f}"))
with c2:
    st.pyplot(fig_line(t_grid, outputs["r_eq_ts"], xlabel="t", ylabel="r_eq (cells)", title="Equivalent radius r_eq(t)"))

# Export ZIP
snap_ids = np.linspace(0, sg_list.shape[0] - 1, snap_count).round().astype(int).tolist()
zip_bytes = make_zip_bytes(outputs, prm, float(thr), include_snapshots=bool(include_snaps), snap_ids=snap_ids)

st.download_button(
    "⬇️ Download ZIP (timeseries_plume.csv + params.json + optional snapshots)",
    data=zip_bytes,
    file_name="ve_forward_outputs.zip",
    mime="application/zip",
)


