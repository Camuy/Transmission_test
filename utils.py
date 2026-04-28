"""
utils.py — Shared utilities for the Transmission Test Dashboard.
"""

import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
from scipy import signal as sci_signal
from scipy.integrate import trapezoid as trapz

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SPACEMENTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spacements")

COLORS = {
    "displacement": "#2196F3",
    "velocity":     "#4CAF50",
    "acceleration": "#FF5722",
    "force":        "#9C27B0",
    "force_cmd":    "#CE93D8",
    "voltage":      "#FF9800",
    "current":      "#03A9F4",
    "power_in":     "#1565C0",
    "power_out":    "#E91E63",
    "energy_in":    "#42A5F5",
    "energy_out":   "#EC407A",
    "efficiency":   "#8BC34A",
    "eff_cum":      "#33691E",
    "rms":          "#FFC107",
}

TEMPLATE   = "plotly_white"
HOVER_MODE = "x unified"


# ─────────────────────────────────────────────────────────────────────────────
# Case discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_cases(root: str = SPACEMENTS_ROOT) -> dict:
    """Return {case_label: path_to_specimen.dat}, sorted alphabetically."""
    files = sorted(
        glob.glob(os.path.join(root, "**", "specimen.dat"), recursive=True)
    )
    return {os.path.relpath(os.path.dirname(f), root): f for f in files}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Caricamento file .dat…")
def load_dat(filepath: str) -> pd.DataFrame:
    """Parse an MTS specimen.dat → clean numeric DataFrame."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    cols = [c.strip().strip('"') for c in lines[3].strip().split(",")]
    df = pd.read_csv(
        filepath, skiprows=5, header=None,
        on_bad_lines="skip", low_memory=False,
    )
    n = df.shape[1]
    df.columns = (
        cols[:n] if len(cols) >= n
        else cols + [f"Col_{i}" for i in range(len(cols), n)]
    )
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(
        subset=["Time", "Ch 1 Displacement", "Ch 1 Force"]
    ).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Signal processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def butter_lp(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth LP filter.  Returns data unchanged if cutoff ≤ 0."""
    nyq = 0.5 * fs
    if cutoff <= 0 or cutoff >= nyq:
        return data.copy()
    b, a = sci_signal.butter(order, cutoff / nyq, btype="low")
    return sci_signal.filtfilt(b, a, data)


def welch_psd(series: pd.Series, fs: float, nperseg: int = 512):
    """Return (freq_array, psd_array) via Welch method."""
    data = series.dropna().values
    nperseg = min(nperseg, len(data) // 4)
    f, pxx = sci_signal.welch(data, fs=fs, nperseg=nperseg)
    return f, pxx


# ─────────────────────────────────────────────────────────────────────────────
# Physics computation (cached by filepath + cutoff)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Calcolo segnali derivati…")
def compute_derived(filepath: str, cutoff: float) -> tuple:
    """
    Return (df_extended, fs_hz).
    All physical channels are added; filtering applied if cutoff > 0.
    """
    df = load_dat(filepath).copy()
    t  = df["Time"].values
    fs = 1.0 / float(np.diff(t).mean())

    # ── Electrical output ─────────────────────────────────────────────────
    df["Voltage_V"]    = butter_lp(df["Aux Input 1"].values * 20, cutoff, fs)   # V
    df["Current_A"]    = butter_lp(df["Aux Input 2"].values / 10.0 * 1000, cutoff, fs)   # A
    df["Current_mA"]   = df["Current_A"]  * 1e3     # A
    df["Power_out_W"]  = df["Voltage_V"]  * df["Current_A"] 
    df["Power_out_mW"] = df["Power_out_W"] * 1e3

    # ── Mechanical input ──────────────────────────────────────────────────
    disp_m = butter_lp(df["Ch 1 Displacement"].values / 1e3, cutoff, fs)  # m
    df["Disp_m_filt"] = disp_m

    vel = butter_lp(np.gradient(disp_m, t), cutoff, fs)   # m/s
    acc = butter_lp(np.gradient(vel,    t), cutoff, fs)   # m/s²
    df["Velocity_ms"]  = vel
    df["Velocity_mms"] = vel * 1e3    # mm/s
    df["Accel_ms2"]    = acc
    df["Accel_mms2"]   = acc * 1e3   # mm/s²

    force = butter_lp(df["Ch 1 Force"].values * 1e3, cutoff, fs) * 10  # N
    df["Force_N"] = force 

    df["Power_in_W"]  = force * vel
    df["Power_in_mW"] = df["Power_in_W"] * 1e3

    # ── Cumulative energy (trapezoidal) ───────────────────────────────────
    dt_arr = np.diff(t)
    pin  = df["Power_in_W"].values
    pout = df["Power_out_W"].values
    df["Energy_in_J"]   = np.r_[0, np.cumsum(0.5 * (pin[:-1]  + pin[1:])  * dt_arr)]
    df["Energy_out_J"]  = np.r_[0, np.cumsum(0.5 * (pout[:-1] + pout[1:]) * dt_arr)]
    df["Energy_out_mJ"] = df["Energy_out_J"] * 1e3

    # ── Instantaneous efficiency ──────────────────────────────────────────
    THR  = 0.01   # W — noise floor threshold
    valid = df["Power_in_W"].abs() > THR
    df["Efficiency_pct"] = np.nan
    df.loc[valid, "Efficiency_pct"] = (
        df.loc[valid, "Power_out_W"] / df.loc[valid, "Power_in_W"].abs() * 100.0
    ).clip(-200, 200)

    df["Cumul_Eff_pct"] = (
        df["Energy_out_J"].abs() / (df["Energy_in_J"].abs() + 1e-12) * 100.0
    ).clip(0, 200)

    # ── Rolling RMS (1-s window) ──────────────────────────────────────────
    win = max(1, int(fs))
    df["Disp_rms_mm"]  = (df["Ch 1 Displacement"]
                          .pow(2).rolling(win, center=True, min_periods=1).mean().pow(0.5))
    df["Vel_rms_mms"]  = (df["Velocity_mms"]
                          .pow(2).rolling(win, center=True, min_periods=1).mean().pow(0.5))
    df["Force_rms_N"]  = (df["Force_N"]
                          .pow(2).rolling(win, center=True, min_periods=1).mean().pow(0.5))

    return df, fs


# ─────────────────────────────────────────────────────────────────────────────
# Per-cycle statistics (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_cycle_stats(
    filepath: str, cutoff: float,
    t_start: float = None, t_end: float = None,
) -> pd.DataFrame:
    """Return a DataFrame with one row per cycle and key statistics."""
    df, _ = compute_derived(filepath, cutoff)
    if t_start is not None:
        df = df[(df["Time"] >= t_start) & (df["Time"] <= t_end)]
    if "Ch 1 Count" not in df.columns or df.empty:
        return pd.DataFrame()

    cycle_col = df["Ch 1 Count"].ffill().astype(int)
    rows = []
    for cyc_id, grp in df.groupby(cycle_col):
        if len(grp) < 5:
            continue
        t_g   = grp["Time"].values
        e_in  = trapz(grp["Power_in_W"].values.clip(min=0),  t_g)
        e_out = trapz(grp["Power_out_W"].values.clip(min=0), t_g)
        eff   = e_out / e_in * 100.0 if e_in > 1e-9 else np.nan
        rows.append({
            "Ciclo":        int(cyc_id),
            "E_in_J":       e_in,
            "E_out_J":      e_out,
            "Eff_pct":      eff,
            "Work_mec_J":   abs(trapz(grp["Force_N"].values, grp["Disp_m_filt"].values)),
            "Disp_pkpk_mm": float(grp["Ch 1 Displacement"].max() - grp["Ch 1 Displacement"].min()),
            "Force_pk_N":   float(grp["Force_N"].abs().max()),
            "Vel_pk_mms":   float(grp["Velocity_mms"].abs().max()),
            "V_mean_V":     float(grp["Voltage_V"].mean()),
            "I_mean_A":     float(grp["Current_A"].mean()),
            "Pout_pk_W":    float(grp["Power_out_W"].abs().max()),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Shared sidebar (call from every page)
# ─────────────────────────────────────────────────────────────────────────────

def sidebar_controls() -> dict:
    """Render shared sidebar widgets and return a data-context dict."""
    st.sidebar.title("⚙️ Transmission Test")
    st.sidebar.markdown("---")

    cases = discover_cases()
    if not cases:
        st.error(f"Nessun `specimen.dat` trovato in `{SPACEMENTS_ROOT}`.")
        st.stop()

    case = st.sidebar.selectbox(
        "Seleziona caso", list(cases.keys()), key="sel_case"
    )
    fp = cases[case]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtro LP — Butterworth 4°")
    apply  = st.sidebar.checkbox("Abilita filtro", value=True, key="apply_lp")
    fc     = st.sidebar.slider(
        "Frequenza di taglio [Hz]", 0.5, 40.0, 5.0, 0.5,
        disabled=not apply, key="fc_hz",
        help="Filtro zero-phase (filtfilt) applicato a spostamento, forza, velocità e accelerazione",
    )
    cutoff = fc if apply else 0.0

    df_full, fs = compute_derived(fp, cutoff)
    t_min = float(df_full["Time"].min())
    t_max = float(df_full["Time"].max())

    st.sidebar.markdown("---")
    st.sidebar.subheader("Finestra temporale")
    use_tw  = st.sidebar.checkbox("Limita finestra", value=False, key="use_tw")
    t_start, t_end = t_min, t_max
    if use_tw:
        rng     = st.sidebar.slider(
            "Range [s]", t_min, t_max, (t_min, t_max), 0.1, key="t_range"
        )
        t_start, t_end = rng

    df = (
        df_full if not use_tw
        else df_full[
            (df_full["Time"] >= t_start) & (df_full["Time"] <= t_end)
        ].copy()
    )

    t_arr     = df["Time"].values
    total_in  = trapz(np.abs(df["Power_in_W"].values),  t_arr)
    total_out = trapz(np.abs(df["Power_out_W"].values), t_arr)
    geff      = total_out / total_in * 100.0 if total_in > 1e-6 else 0.0
    n_cyc     = (
        int(df["Ch 1 Count"].max() - df["Ch 1 Count"].min())
        if "Ch 1 Count" in df.columns else 0
    )
    test_freq = (
        float(df["Ch 1 Command Frequency"].median())
        if "Ch 1 Command Frequency" in df.columns else 0.0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Campioni:** {len(df):,}")
    st.sidebar.markdown(f"**Fs acq.:** {fs:.1f} Hz")
    st.sidebar.markdown(f"**Durata:** {t_arr[-1] - t_arr[0]:.2f} s")
    st.sidebar.markdown(f"**Cicli:** {n_cyc}")
    st.sidebar.markdown(f"**Freq. test:** {test_freq:.3f} Hz")

    return dict(
        df=df, fs=fs, filepath=fp, case=case, cutoff=cutoff,
        total_in=total_in, total_out=total_out, global_eff=geff,
        n_cycles=n_cyc, test_freq=test_freq,
        t_start=t_start, t_end=t_end,
    )
