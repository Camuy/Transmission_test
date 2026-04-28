"""
pages/1_Spostamenti.py — Analisi cinematica: spostamento, velocità, accelerazione
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from utils import sidebar_controls, COLORS, TEMPLATE, HOVER_MODE, welch_psd, compute_cycle_stats

st.set_page_config(
    page_title="Spostamenti | Transmission Test",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

ctx = sidebar_controls()
df  = ctx["df"]
fs  = ctx["fs"]

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("📐 Analisi Cinematica — Spostamenti, Velocità, Accelerazione")
st.caption(
    f"Caso: **{ctx['case']}**  ·  Fs: **{fs:.1f} Hz**  ·  "
    f"Freq. test: **{ctx['test_freq']:.3f} Hz**  ·  Cicli: **{ctx['n_cycles']}**"
)

# ─────────────────────────────────────────────────────────────────────────────
# KPI row
# ─────────────────────────────────────────────────────────────────────────────
c = st.columns(6)
c[0].metric("Corsa max [mm]",        f"{df['Ch 1 Displacement'].abs().max():.2f}")
c[1].metric("Corsa pk-pk [mm]",      f"{df['Ch 1 Displacement'].max()-df['Ch 1 Displacement'].min():.2f}")
c[2].metric("RMS spostamento [mm]",  f"{float(np.sqrt(np.mean(df['Ch 1 Displacement']**2))):.3f}")
c[3].metric("Vel. picco [mm/s]",     f"{df['Velocity_mms'].abs().max():.2f}")
c[4].metric("Vel. RMS [mm/s]",       f"{float(np.sqrt(np.mean(df['Velocity_mms']**2))):.3f}")
c[5].metric("Acc. picco [m/s²]",     f"{df['Accel_ms2'].abs().max():.3f}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Main linked figure — 3 panels (Disp · Vel · Acc)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Spostamento · Velocità · Accelerazione — asse temporale collegato")

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=[
        "Spostamento [mm]",
        "Velocità [mm/s]",
        "Accelerazione [m/s²]",
    ],
)

_t = df["Time"]

# Displacement (raw + filtered overlay)
fig.add_trace(go.Scatter(
    x=_t, y=df["Ch 1 Displacement"],
    name="Spostamento (grezzo)", mode="lines",
    line=dict(color="rgba(33,150,243,0.35)", width=1),
    hovertemplate="%{y:.3f} mm<extra>Grezzo</extra>",
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=_t, y=df["Disp_m_filt"] * 1e3,
    name="Spostamento (filtrato)", mode="lines",
    line=dict(color=COLORS["displacement"], width=1.5),
    hovertemplate="%{y:.3f} mm<extra>Filtrato</extra>",
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=_t, y=df["Disp_rms_mm"],
    name="RMS (1 s)", mode="lines",
    line=dict(color=COLORS["rms"], width=1.2, dash="dash"),
    hovertemplate="%{y:.3f} mm<extra>RMS</extra>",
), row=1, col=1)

# Velocity
fig.add_trace(go.Scatter(
    x=_t, y=df["Velocity_mms"],
    name="Velocità", mode="lines",
    line=dict(color=COLORS["velocity"], width=1.2),
    fill="tozeroy", fillcolor="rgba(76,175,80,0.07)",
    hovertemplate="%{y:.2f} mm/s<extra>Vel</extra>",
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=_t, y=df["Vel_rms_mms"],
    name="RMS vel", mode="lines",
    line=dict(color=COLORS["rms"], width=1.2, dash="dash"),
    hovertemplate="%{y:.2f} mm/s<extra>RMS vel</extra>",
), row=2, col=1)

# Acceleration
fig.add_trace(go.Scatter(
    x=_t, y=df["Accel_ms2"],
    name="Accelerazione", mode="lines",
    line=dict(color=COLORS["acceleration"], width=1.2),
    fill="tozeroy", fillcolor="rgba(255,87,34,0.07)",
    hovertemplate="%{y:.4f} m/s²<extra>Acc</extra>",
), row=3, col=1)

for r in range(1, 4):
    fig.add_hline(y=0, line=dict(color="gray", width=0.4, dash="dot"), row=r, col=1)

fig.update_xaxes(
    title_text="Tempo [s]", row=3, col=1,
    rangeslider=dict(visible=True, thickness=0.04),
)
fig.update_yaxes(title_text="mm",    row=1, col=1)
fig.update_yaxes(title_text="mm/s",  row=2, col=1)
fig.update_yaxes(title_text="m/s²",  row=3, col=1)
fig.update_layout(
    height=680,
    template=TEMPLATE,
    hovermode=HOVER_MODE,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="right", x=1),
    margin=dict(t=80, b=20, l=70, r=30),
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Phase portrait + PSD
# ─────────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Ritratto di fase — Vel vs Disp")
    st.caption("Colorito per tempo (viola=inizio → giallo=fine)")

    fig_ph = go.Figure()
    fig_ph.add_trace(go.Scatter(
        x=df["Ch 1 Displacement"],
        y=df["Velocity_mms"],
        mode="markers",
        marker=dict(
            color=df["Time"],
            colorscale="Plasma",
            size=2,
            opacity=0.6,
            colorbar=dict(title="Tempo [s]", thickness=12),
        ),
        hovertemplate="Disp: %{x:.2f} mm<br>Vel: %{y:.2f} mm/s<extra></extra>",
        name="",
    ))
    fig_ph.update_layout(
        height=380,
        template=TEMPLATE,
        xaxis_title="Spostamento [mm]",
        yaxis_title="Velocità [mm/s]",
        margin=dict(t=30, b=50, l=70, r=20),
    )
    st.plotly_chart(fig_ph, use_container_width=True)

with col2:
    st.markdown("#### Densità spettrale (Welch) — Spostamento")

    f_disp, p_disp = welch_psd(df["Ch 1 Displacement"], fs)
    f_vel,  p_vel  = welch_psd(df["Velocity_mms"],      fs)

    fig_psd = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=["PSD Spostamento", "PSD Velocità"])
    fig_psd.add_trace(go.Scatter(
        x=f_disp, y=10 * np.log10(p_disp + 1e-20),
        mode="lines", line=dict(color=COLORS["displacement"], width=1.5),
        name="Disp PSD", hovertemplate="%{x:.3f} Hz  %{y:.1f} dB<extra></extra>",
    ), row=1, col=1)
    fig_psd.add_trace(go.Scatter(
        x=f_vel,  y=10 * np.log10(p_vel + 1e-20),
        mode="lines", line=dict(color=COLORS["velocity"], width=1.5),
        name="Vel PSD", hovertemplate="%{x:.3f} Hz  %{y:.1f} dB<extra></extra>",
    ), row=2, col=1)

    # Annotate dominant frequency
    dom_f = f_disp[np.argmax(p_disp)]
    fig_psd.add_vline(x=dom_f, line=dict(color="red", width=1, dash="dash"),
                      annotation_text=f"{dom_f:.3f} Hz", row=1, col=1)

    fig_psd.update_xaxes(title_text="Frequenza [Hz]", row=2, col=1)
    fig_psd.update_yaxes(title_text="dB re mm²/Hz", row=1, col=1)
    fig_psd.update_yaxes(title_text="dB re (mm/s)²/Hz", row=2, col=1)
    fig_psd.update_layout(
        height=380, template=TEMPLATE, hovermode="x unified",
        showlegend=False, margin=dict(t=50, b=50, l=80, r=20),
    )
    st.plotly_chart(fig_psd, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Per-cycle kinematics
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Statistiche per ciclo — cinematica")

cdf = compute_cycle_stats(ctx["filepath"], ctx["cutoff"], ctx["t_start"], ctx["t_end"])
if not cdf.empty:
    col3, col4 = st.columns(2)

    with col3:
        fig_amp = go.Figure()
        fig_amp.add_trace(go.Bar(
            x=cdf["Ciclo"], y=cdf["Disp_pkpk_mm"],
            name="Disp pk-pk [mm]",
            marker_color=COLORS["displacement"],
            hovertemplate="Ciclo %{x}<br>Disp pk-pk: %{y:.3f} mm<extra></extra>",
        ))
        fig_amp.update_layout(
            height=320, template=TEMPLATE,
            xaxis_title="Ciclo #", yaxis_title="Disp pk-pk [mm]",
            title="Ampiezza di spostamento per ciclo",
            margin=dict(t=50, b=50, l=70, r=20),
        )
        st.plotly_chart(fig_amp, use_container_width=True)

    with col4:
        fig_vel = go.Figure()
        fig_vel.add_trace(go.Bar(
            x=cdf["Ciclo"], y=cdf["Vel_pk_mms"],
            name="Vel picco [mm/s]",
            marker_color=COLORS["velocity"],
            hovertemplate="Ciclo %{x}<br>Vel pk: %{y:.2f} mm/s<extra></extra>",
        ))
        fig_vel.update_layout(
            height=320, template=TEMPLATE,
            xaxis_title="Ciclo #", yaxis_title="Velocità picco [mm/s]",
            title="Velocità di picco per ciclo",
            margin=dict(t=50, b=50, l=70, r=20),
        )
        st.plotly_chart(fig_vel, use_container_width=True)

    # Histogram of displacement values
    st.markdown("#### Distribuzione spostamento & velocità")
    col5, col6 = st.columns(2)

    with col5:
        fig_hd = go.Figure()
        fig_hd.add_trace(go.Histogram(
            x=df["Ch 1 Displacement"], nbinsx=80,
            marker_color=COLORS["displacement"], opacity=0.75, name="Disp",
        ))
        fig_hd.update_layout(
            height=280, template=TEMPLATE,
            xaxis_title="Spostamento [mm]", yaxis_title="Conteggio",
            margin=dict(t=20, b=50, l=70, r=20), showlegend=False,
        )
        st.plotly_chart(fig_hd, use_container_width=True)

    with col6:
        fig_hv = go.Figure()
        fig_hv.add_trace(go.Histogram(
            x=df["Velocity_mms"], nbinsx=80,
            marker_color=COLORS["velocity"], opacity=0.75, name="Vel",
        ))
        fig_hv.update_layout(
            height=280, template=TEMPLATE,
            xaxis_title="Velocità [mm/s]", yaxis_title="Conteggio",
            margin=dict(t=20, b=50, l=70, r=20), showlegend=False,
        )
        st.plotly_chart(fig_hv, use_container_width=True)
