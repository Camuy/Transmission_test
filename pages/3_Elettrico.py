"""
pages/3_Elettrico.py — Analisi canali elettrici: tensione, corrente, potenza out
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import sidebar_controls, COLORS, TEMPLATE, HOVER_MODE, welch_psd, compute_cycle_stats

st.set_page_config(
    page_title="Elettrico | Transmission Test",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ctx = sidebar_controls()
df  = ctx["df"]
fs  = ctx["fs"]

# ─────────────────────────────────────────────────────────────────────────────
st.title("⚡ Analisi Elettrica — Tensione · Corrente · Potenza out")
st.caption(
    f"Caso: **{ctx['case']}**  ·  "
    f"Aux1/(1/20) → V  ·  Aux2/10 → A  ·  "
    f"Fs: **{fs:.1f} Hz**  ·  Cicli: **{ctx['n_cycles']}**"
)

# ─────────────────────────────────────────────────────────────────────────────
# KPI
# ─────────────────────────────────────────────────────────────────────────────
c = st.columns(6)
c[0].metric("Tensione max [V]",    f"{df['Voltage_V'].max():.3f}")
c[1].metric("Tensione RMS [V]",    f"{float(np.sqrt(np.mean(df['Voltage_V']**2))):.3f}")
c[2].metric("Corrente max [A]",   f"{df['Current_A'].max():.3f}")
c[3].metric("Corrente RMS [A]",   f"{float(np.sqrt(np.mean(df['Current_A']**2))):.3f}")
c[4].metric("P_out picco [W]",    f"{df['Power_out_W'].abs().max():.3f}")
c[5].metric("P_out media [W]",    f"{df['Power_out_W'].mean():.3f}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Main linked figure — V + I + P_out (3 panels)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Tensione · Corrente · Potenza — asse temporale collegato")

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=[
        "Tensione [V]",
        "Corrente [A]",
        "Potenza out [W]",
    ],
)

_t = df["Time"]

fig.add_trace(go.Scatter(
    x=_t, y=df["Voltage_V"],
    name="Tensione", mode="lines",
    line=dict(color=COLORS["voltage"], width=1.2),
    fill="tozeroy", fillcolor="rgba(255,152,0,0.08)",
    hovertemplate="%{y:.5f} V<extra>Tensione</extra>",
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=_t, y=df["Current_A"],
    name="Corrente", mode="lines",
    line=dict(color=COLORS["current"], width=1.2),
    fill="tozeroy", fillcolor="rgba(3,169,244,0.08)",
    hovertemplate="%{y:.5f} A<extra>Corrente</extra>",
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=_t, y=df["Power_out_W"],
    name="Potenza out", mode="lines",
    line=dict(color=COLORS["power_out"], width=1.2),
    fill="tozeroy", fillcolor="rgba(233,30,99,0.08)",
    hovertemplate="%{y:.6f} W<extra>P_out</extra>",
), row=3, col=1)

for r in range(1, 4):
    fig.add_hline(y=0, line=dict(color="gray", width=0.4, dash="dot"), row=r, col=1)

fig.update_xaxes(
    title_text="Tempo [s]", row=3, col=1,
    rangeslider=dict(visible=True, thickness=0.04),
)
fig.update_yaxes(title_text="V",  row=1, col=1)
fig.update_yaxes(title_text="A", row=2, col=1)
fig.update_yaxes(title_text="W", row=3, col=1)
fig.update_layout(
    height=640,
    template=TEMPLATE,
    hovermode=HOVER_MODE,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="right", x=1),
    margin=dict(t=80, b=20, l=70, r=30),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# I–V characteristic + distributions
# ─────────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Caratteristica I–V (colorato per tempo)")
    fig_iv = go.Figure()
    fig_iv.add_trace(go.Scatter(
        x=df["Voltage_V"],
        y=df["Current_A"],
        mode="markers",
        marker=dict(
            color=df["Time"],
            colorscale="Viridis",
            size=2.5,
            opacity=0.6,
            colorbar=dict(title="Tempo [s]", thickness=12),
        ),
        hovertemplate="V: %{x:.5f} V<br>I: %{y:.5f} A<extra></extra>",
        name="",
    ))
    fig_iv.update_layout(
        height=360, template=TEMPLATE,
        xaxis_title="Tensione [V]",
        yaxis_title="Corrente [A]",
        margin=dict(t=30, b=50, l=70, r=20),
    )
    st.plotly_chart(fig_iv, use_container_width=True)

with col2:
    st.markdown("#### Distribuzione Potenza out")
    
    # 1. Preparazione dati e parametri
    pout_clip = df["Power_out_W"].clip(lower=0).dropna()
    n_samples = len(pout_clip)
    n_bins = 80
    
    fig_hist = go.Figure()

    # Istogramma
    fig_hist.add_trace(go.Histogram(
        x=pout_clip,
        nbinsx=n_bins,
        marker_color=COLORS["power_out"],
        opacity=0.8,
        name="P_out ≥ 0",
    ))

    # 2. Calcolo KDE corretta
    if n_samples > 1 and pout_clip.std() > 1e-12:
        from scipy import stats as scipy_stats
        bin_width = (pout_clip.max() - pout_clip.min()) / n_bins
        kde_x = np.linspace(pout_clip.min(), pout_clip.max(), 300)
        kde_func = scipy_stats.gaussian_kde(pout_clip)
        kde_y_density = kde_func(kde_x)
        kde_y = kde_y_density * n_samples * bin_width

        fig_hist.add_trace(go.Scatter(
            x=kde_x, y=kde_y,
            mode="lines", 
            line=dict(color="#880E4F", width=2.5),
            name="KDE (Trend)",
        ))

    # 3. Layout
    fig_hist.update_layout(
        height=360, 
        template=TEMPLATE,
        xaxis_title="Potenza out [W]",
        yaxis_title="Conteggio",
        margin=dict(t=30, b=50, l=70, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Aux channels overview
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Canali Aux grezzi (tutti e 6)")

aux_cols  = [c for c in df.columns if c.startswith("Aux Input")]
aux_names = {
    "Aux Input 1": f"Aux1 → Tensione  (raw V)",
    "Aux Input 2": f"Aux2 → Corrente (raw V)",
    "Aux Input 3": "Aux3 (raw V)",
    "Aux Input 4": "Aux4 (raw V)",
    "Aux Input 5": "Aux5 (raw V)",
    "Aux Input 6": "Aux6 (raw V)",
}
aux_colors = [COLORS["voltage"], COLORS["current"],
              "#66BB6A", "#AB47BC", "#FFA726", "#26C6DA"]

if aux_cols:
    fig_aux = make_subplots(
        rows=len(aux_cols), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[aux_names.get(c, c) for c in aux_cols],
    )
    for i, col in enumerate(aux_cols):
        fig_aux.add_trace(go.Scatter(
            x=df["Time"], y=df[col],
            name=col, mode="lines",
            line=dict(color=aux_colors[i % len(aux_colors)], width=1),
            hovertemplate=f"%{{y:.5f}} V<extra>{col}</extra>",
        ), row=i + 1, col=1)

    fig_aux.update_xaxes(title_text="Tempo [s]", row=len(aux_cols), col=1)
    for i in range(1, len(aux_cols) + 1):
        fig_aux.update_yaxes(title_text="V", row=i, col=1)

    fig_aux.update_layout(
        height=80 + 130 * len(aux_cols),
        template=TEMPLATE,
        hovermode=HOVER_MODE,
        showlegend=False,
        margin=dict(t=60, b=30, l=70, r=20),
    )
    st.plotly_chart(fig_aux, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Per-cycle electrical stats
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Statistiche elettriche per ciclo")

cdf = compute_cycle_stats(ctx["filepath"], ctx["cutoff"], ctx["t_start"], ctx["t_end"])
if not cdf.empty:
    col3, col4 = st.columns(2)

    with col3:
        fig_v = go.Figure()
        fig_v.add_trace(go.Bar(
            x=cdf["Ciclo"], y=cdf["V_mean_V"],
            marker_color=COLORS["voltage"],
            hovertemplate="Ciclo %{x}<br>V media: %{y:.5f} V<extra></extra>",
        ))
        fig_v.update_layout(
            height=300, template=TEMPLATE,
            xaxis_title="Ciclo #", yaxis_title="Tensione media [V]",
            title="Tensione media per ciclo",
            margin=dict(t=50, b=50, l=70, r=20),
        )
        st.plotly_chart(fig_v, use_container_width=True)

    with col4:
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(
            x=cdf["Ciclo"], y=cdf["Pout_pk_W"],
            marker_color=COLORS["power_out"],
            hovertemplate="Ciclo %{x}<br>P_out pk: %{y:.5f} W<extra></extra>",
        ))
        fig_p.update_layout(
            height=300, template=TEMPLATE,
            xaxis_title="Ciclo #", yaxis_title="P_out picco [W]",
            title="Potenza out di picco per ciclo",
            margin=dict(t=50, b=50, l=70, r=20),
        )
        st.plotly_chart(fig_p, use_container_width=True)
