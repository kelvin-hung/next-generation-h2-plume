import os
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
VE_CORE_PATH = os.path.join(HERE, "ve_core.py")

spec = importlib.util.spec_from_file_location("ve_core", VE_CORE_PATH)
ve_core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ve_core)

DEFAULT_PARAMS = ve_core.DEFAULT_PARAMS
run_forward     = ve_core.run_forward
choose_well_ij  = ve_core.choose_well_ij
prepare_phi_k   = ve_core.prepare_phi_k



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

⚠️ Important: the solver requires **linear, strictly-positive k**.  
If your NPZ stores **log10(k)**, you must enable conversion.
"""
)

# -------------------------
# Helpers
# -------------------------
def load_npz(uploaded):
    data = np.load(uploaded)
    if "phi" not in data or "k" not in data:
        raise ValueError("NPZ must contain keys: phi and k")
    meta = {}
    for key in data.files:
        if key not in ("phi", "k"):
            try:
                meta[key] = data[key]
            except Exception:
                pass
    return data["phi"], data["k"], meta

def load_schedule_csv(uploaded):
    df = pd.read_csv(uploaded)
    df.columns = [c.lower().strip() for c in df.columns]
    if "t" not in df.columns or "q" not in df.columns:
        raise ValueError("CSV must have columns: t, q")
    t = df["t"].to_numpy(dtype=np.float32)
    q = df["q"].to_numpy(dtype=np.float32)
    return t, q

def stats_line(name, arr):
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    frac = float(finite.mean()) * 100.0
    if finite.any():
        mn = float(np.nanmin(arr))
        mx = float(np.nanmax(arr))
        return f"{name}: min={mn:.4g}, max={mx:.4g}, finite={frac:.1f}%"
    return f"{name}: no finite values"

def safe_log10(x, floor=1e-30):
    x = np.asarray(x)
    return np.log10(np.maximum(x, floor))

def fig_imshow(arr, title, vmin=None, vmax=None):
    arr = np.asarray(arr)
    fig = plt.figure(figsize=(7.2, 4.6))
    plt.title(title)
    if vmin is None:
        vmin = float(np.nanmin(arr)) if np.isfinite(arr).any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
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

def sanitize_phi_k(phi_raw, k_raw, *, k_is_log10, k_floor, k_cap):
    """
    Convert k to linear if needed, clip to [k_floor, k_cap] on active cells,
    keep inactive as NaN (so mask works), and return cleaned phi/k.
    """
    phi = np.array(phi_raw, dtype=np.float32, copy=False)
    k0  = np.array(k_raw, dtype=np.float32, copy=False)

    # Convert from log10(k) -> k (linear) if requested
    if k_is_log10:
        k = np.power(10.0, k0, dtype=np.float32)
    else:
        k = k0

    # Keep NaNs if present; enforce finite where possible
    phi = np.where(np.isfinite(phi), phi, np.nan).astype(np.float32)
    k   = np.where(np.isfinite(k),   k,   np.nan).astype(np.float32)

    # Active mask definition: phi>0 and finite and k finite
    mask = np.isfinite(phi) & np.isfinite(k) & (phi > 0)

    if mask.sum() == 0:
        raise ValueError("No active cells found (phi<=0 or phi/k invalid).")

    # Enforce strictly positive k on active cells
    k_active = k[mask]
    # If k has non-positive values, clip them up
    k_active = np.clip(k_active, k_floor, k_cap).astype(np.float32)

    k2 = k.copy()
    k2[mask] = k_active
    # Inactive -> NaN (so plots show domain cleanly)
    k2[~mask] = np.nan
    phi2 = phi.copy()
    phi2[~mask] = np.nan

    return phi2, k2, mask

def auto_detect_log10_k(k_raw):
    """
    Heuristic:
    - If k has lots of negatives OR values mostly within [-5, 8], it's likely log10(k).
    - If k max < ~20 and min < 0, very likely log10(k).
    """
    k = np.asarray(k_raw, dtype=np.float32)
    finite = np.isfinite(k)
    if not finite.any():
        return False, "k has no finite values (cannot detect)."

    mn = float(np.nanmin(k))
    mx = float(np.nanmax(k))

    likely = False
    reason = ""

    if (mn < 0) and (mx < 20):
        likely = True
        reason = f"detected mn={mn:.3g}<0 and mx={mx:.3g}<20 (looks like log10(k))."
    elif (mn > -6) and (mx < 8):
        likely = True
        reason = f"detected mn={mn:.3g}, mx={mx:.3g} within typical log10(k) range."
    else:
        reason = f"detected mn={mn:.3g}, mx={mx:.3g} (looks like linear k)."

    return likely, reason

# -------------------------
# Sidebar inputs
# -------------------------
with st.sidebar:
    st.header("1) Upload inputs")
    up_npz = st.file_uploader("phi/k NPZ", type=["npz"])
    up_csv = st.file_uploader("schedule CSV", type=["csv"])

    st.divider()
    st.header("2) Permeability interpretation")
    auto_log10 = st.checkbox("Auto-detect if k is log10(k)", value=True)
    k_is_log10_user = st.checkbox("k in NPZ is log10(k)", value=False)

    st.caption("Safety bounds (apply on active cells only)")
    k_floor = st.number_input("k floor (linear)", value=1e-6, format="%.2e")
    k_cap   = st.number_input("k cap (linear)", value=1e6, format="%.2e")

    st.divider()
    st.header("3) Well location")
    well_mode = st.selectbox("Well placement", ["max_k", "center", "manual"], index=0)
    manual_i = st.number_input("manual i", value=0, step=1)
    manual_j = st.number_input("manual j", value=0, step=1)

    st.divider()
    st.header("4) Schedule scaling")
    q_scale = st.number_input("Schedule scale factor (multiplies q)", value=1.0)
    normalize_q = st.checkbox("Normalize q to max(|q|)=1", value=False)

    st.divider()
    st.header("5) Output / metrics")
    thr_area = st.slider("Area threshold (Sg > thr)", 0.0, 0.5, 0.05, 0.01)

    st.divider()
    st.header("6) Paper parameters")
    use_defaults = st.checkbox("Use paper-calibrated defaults", value=True)

    params = dict(DEFAULT_PARAMS)
    if not use_defaults:
        # Only expose if you really want to tune
        for key in [
            "D0","alpha_p","src_amp","prod_frac",
            "Swr","Sgr_max","C_L","hc","mob_exp","anisD",
            "eps_h","nu","m_spread","ap_diff","qp_amp"
        ]:
            params[key] = st.number_input(key, value=float(params[key]))

    st.divider()
    run_btn = st.button("Run forward prediction", type="primary")

# -------------------------
# Main logic
# -------------------------
if up_npz is None or up_csv is None:
    st.info("Upload a phi/k NPZ and a schedule CSV to begin.")
    st.stop()

try:
    phi_raw, k_raw, meta = load_npz(up_npz)
    t, q = load_schedule_csv(up_csv)

    q = q.astype(np.float32) * np.float32(q_scale)
    if normalize_q:
        m = float(np.max(np.abs(q))) if q.size else 1.0
        if m > 0:
            q = (q / m).astype(np.float32)

except Exception as e:
    st.error(f"Failed to read inputs: {e}")
    st.stop()

# Decide k_is_log10
k_is_log10 = bool(k_is_log10_user)
detect_note = ""
if auto_log10:
    likely, reason = auto_detect_log10_k(k_raw)
    detect_note = reason
    # If auto says likely log10, override unless user explicitly checked/unchecked?
    # Here: auto-detect wins ONLY if user didn't manually enable it.
    if (not k_is_log10_user) and likely:
        k_is_log10 = True

if detect_note:
    st.caption(f"Auto-detect: {detect_note}  → using k_is_log10={k_is_log10}")

# Sanitize fields (this is critical)
try:
    phi, k_lin, mask = sanitize_phi_k(
        phi_raw, k_raw,
        k_is_log10=k_is_log10,
        k_floor=float(k_floor),
        k_cap=float(k_cap),
    )
except Exception as e:
    st.error(f"Input fields invalid: {e}")
    st.stop()

# Diagnostics
with st.expander("Diagnostics (recommended)", expanded=True):
    st.write(stats_line("phi (raw)", phi_raw))
    st.write(stats_line("k (raw)", k_raw))
    st.write(stats_line("phi (clean)", phi))
    st.write(stats_line("k_lin (clean)", k_lin))
    st.write(f"active cells = {int(mask.sum())} / {mask.size} ({100*mask.mean():.1f}%)")
    # show if any non-positive k survived (should not)
    if np.isfinite(k_lin).any():
        kmin = float(np.nanmin(k_lin))
        if kmin <= 0:
            st.warning(f"k_lin has non-positive values (min={kmin}). This will destabilize the solver.")

# Previews
colA, colB = st.columns(2)
with colA:
    st.subheader("Input: porosity (phi) [masked inactive]")
    st.pyplot(fig_imshow(phi, f"phi | shape={phi.shape}"))
with colB:
    st.subheader("Input: permeability (log10(k)) [masked inactive]")
    st.pyplot(fig_imshow(safe_log10(k_lin), f"log10(k_lin) | shape={k_lin.shape}"))

# Compute well coordinate (use ve_core.prepare_phi_k if available; tolerate 2-return older versions)
try:
    out = prepare_phi_k(phi, k_lin)  # expects phi/k; may return (phi_clean,k_norm,mask) or (k_norm,mask)
    if isinstance(out, tuple) and len(out) == 3:
        phi_clean, k_norm, mask2 = out
    elif isinstance(out, tuple) and len(out) == 2:
        k_norm, mask2 = out
        phi_clean = phi
    else:
        raise ValueError("prepare_phi_k returned unexpected output.")
except Exception:
    # fallback: simple normalization for well picking
    k_norm = np.where(mask, k_lin, np.nan).astype(np.float32)
    kpos = k_norm[np.isfinite(k_norm)]
    k_eff = float(np.exp(np.mean(np.log(np.maximum(kpos, 1e-30))))) if kpos.size else 1.0
    k_norm = np.where(mask, k_norm / (k_eff + 1e-12), np.nan).astype(np.float32)
    mask2 = mask

# Select well
if well_mode == "manual":
    well_ij = (int(manual_i), int(manual_j))
else:
    well_ij = None
wi, wj = choose_well_ij(k_norm, mask2, well_mode, ij=well_ij)
st.caption(f"Selected well (i,j)=({wi},{wj}) | active cells={int(mask2.sum())}")

st.subheader("Injection schedule")
st.pyplot(fig_schedule(t, q, tidx=0))

if not run_btn:
    st.stop()

# Run forward model using CLEANED fields (key fix)
with st.spinner("Running VE+Darcy+Land forward model..."):
    try:
        res = run_forward(
            phi=phi,               # masked inactive
            k=k_lin,               # LINEAR, positive k
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
if Nt <= 0:
    st.error("No timesteps returned.")
    st.stop()

tidx = st.slider("Select timestep (tidx)", 0, Nt - 1, 0)

# Plot Sg with safety (mask invalid)
sg = np.array(res.sg_list[tidx], dtype=np.float32)
sg_plot = np.where(np.isfinite(sg), sg, np.nan).astype(np.float32)

left, right = st.columns(2)
with left:
    st.subheader(f"Sg predicted | tidx={tidx}")
    if np.isfinite(sg_plot).any():
        st.pyplot(fig_imshow(sg_plot, f"Sg predicted | tidx={tidx}", vmin=0.0, vmax=1.0))
    else:
        st.warning(f"Sg predicted | tidx={tidx} has no finite values.")
with right:
    st.subheader(f"Pressure surrogate | tidx={tidx}")
    p = res.p_list[tidx] if getattr(res, "p_list", None) is not None else None
    if p is None:
        st.write("Pressure output disabled.")
    else:
        p = np.array(p, dtype=np.float32)
        p_plot = np.where(np.isfinite(p), p, np.nan).astype(np.float32)
        if np.isfinite(p_plot).any():
            st.pyplot(fig_imshow(p_plot, f"p predicted | tidx={tidx}"))
        else:
            st.warning("Pressure has no finite values at this timestep.")

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

# Downloads
st.subheader("Download results")
out_npz = io.BytesIO()
sg_stack = np.stack([np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0) for s in res.sg_list], axis=0).astype(np.float32)
np.savez_compressed(out_npz, sg=sg_stack, t=res.t, q=res.q, area=res.area, r_eq=res.r_eq)
st.download_button("Download predicted Sg (NPZ)", data=out_npz.getvalue(), file_name="sg_predicted.npz")

out_csv = pd.DataFrame({"t": res.t, "q": res.q, "area": res.area, "r_eq": res.r_eq}).to_csv(index=False).encode("utf-8")
st.download_button("Download time series (CSV)", data=out_csv, file_name="plume_timeseries.csv")




