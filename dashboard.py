"""
dashboard.py — Panoramica generale (main page)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from utils import sidebar_controls, COLORS, TEMPLATE, HOVER_MODE, compute_cycle_stats

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Panoramica | Transmission Test",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ctx = sidebar_controls()
df  = ctx["df"]
fs  = ctx["fs"]

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("⚙️ Panoramica — Transmission Test")
st.caption(
    f"Caso: **{ctx['case']}**  ·  "
    f"Durata: **{df['Time'].max()-df['Time'].min():.2f} s**  ·  "
    f"Cicli: **{ctx['n_cycles']}**  ·  "
    f"Frequenza test: **{ctx['test_freq']:.3f} Hz**  ·  "
    f"Fs acquisizione: **{fs:.1f} Hz**"
)

# ─────────────────────────────────────────────────────────────────────────────
# KPI row
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Metriche chiave")
c = st.columns(4)
c[0].metric("Energia in",   f"{ctx['total_in']:.2f} J")
c[1].metric("P_in picco",   f"{df['Power_in_W'].abs().max():.2f} W")
c[2].metric("Efficienza",   f"{ctx['global_eff']:.2f} %")
c[3].metric("Cicli",        f"{ctx['n_cycles']}")

c = st.columns(4)
c[0].metric("Energia out",  f"{ctx['total_out']:.2f} J")
c[1].metric("P_out picco",  f"{df['Power_out_W'].abs().max():.2f} W")
c[2].metric("Forza picco",  f"{df['Force_N'].abs().max():.2f} N")
c[3].metric("Vel. picco",   f"{df['Velocity_mms'].abs().max():.2f} mm/s")



st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Main overview — 5 panels, fully linked on x-axis
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Tutti i segnali — asse temporale condiviso")
st.caption("Zoom e pan su un grafico si propagano a tutti gli altri. Hover mostra tutti i valori simultaneamente.")

fig = make_subplots(
    rows=5, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.025,
    subplot_titles=[
        "Spostamento [mm]",
        "Forza [N]",
        "Velocità [mm/s]",
        "Potenza meccanica in [W]",
        "Potenza elettrica out [W]",
    ],
    row_heights=[1, 1, 1, 1, 1],
)

_t = df["Time"]

# Panel 1 – Displacement
fig.add_trace(go.Scatter(
    x=_t, y=df["Ch 1 Displacement"],
    name="Spostamento", mode="lines",
    line=dict(color=COLORS["displacement"], width=1),
    hovertemplate="%{y:.3f} mm<extra>Disp</extra>",
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=_t, y=df["Disp_rms_mm"],
    name="RMS disp", mode="lines",
    line=dict(color=COLORS["rms"], width=1, dash="dot"),
    hovertemplate="%{y:.3f} mm<extra>RMS</extra>",
), row=1, col=1)

# Panel 2 – Force
fig.add_trace(go.Scatter(
    x=_t, y=df["Force_N"],
    name="Forza", mode="lines",
    line=dict(color=COLORS["force"], width=1),
    hovertemplate="%{y:.2f} N<extra>Forza</extra>",
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=_t, y=df["Force_rms_N"],
    name="RMS forza", mode="lines",
    line=dict(color=COLORS["rms"], width=1, dash="dot"),
    hovertemplate="%{y:.2f} N<extra>RMS</extra>",
), row=2, col=1)

# Panel 3 – Velocity
fig.add_trace(go.Scatter(
    x=_t, y=df["Velocity_mms"],
    name="Velocità", mode="lines",
    line=dict(color=COLORS["velocity"], width=1),
    hovertemplate="%{y:.2f} mm/s<extra>Vel</extra>",
), row=3, col=1)

# Panel 4 – Power in
fig.add_trace(go.Scatter(
    x=_t, y=df["Power_in_W"],
    name="Potenza in", mode="lines",
    line=dict(color=COLORS["power_in"], width=1),
    fill="tozeroy", fillcolor="rgba(21,101,192,0.08)",
    hovertemplate="%{y:.3f} W<extra>P_in</extra>",
), row=4, col=1)

# Panel 5 – Power out
fig.add_trace(go.Scatter(
    x=_t, y=df["Power_out_W"],
    name="Potenza out", mode="lines",
    line=dict(color=COLORS["power_out"], width=1),
    fill="tozeroy", fillcolor="rgba(233,30,99,0.08)",
    hovertemplate="%{y:.3f} W<extra>P_out</extra>",
), row=5, col=1)

# Zero reference lines
for r in range(1, 6):
    fig.add_hline(y=0, line=dict(color="gray", width=0.4, dash="dot"), row=r, col=1)

# Range slider on bottom axis only
fig.update_xaxes(
    title_text="Tempo [s]", row=5, col=1,
    rangeslider=dict(visible=True, thickness=0.04),
)
fig.update_yaxes(title_text="mm",   row=1, col=1)
fig.update_yaxes(title_text="N",    row=2, col=1)
fig.update_yaxes(title_text="mm/s", row=3, col=1)
fig.update_yaxes(title_text="W",    row=4, col=1)
fig.update_yaxes(title_text="W",    row=5, col=1)

fig.update_layout(
    height=870,
    template=TEMPLATE,
    hovermode=HOVER_MODE,
    showlegend=True,
    legend=dict(
        orientation="h", yanchor="bottom", y=1.01,
        xanchor="right", x=1, font=dict(size=11),
    ),
    margin=dict(t=80, b=20, l=70, r=30),
)

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Summary row
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### Bilancio energetico globale")

    fig_e = go.Figure()
    fig_e.add_trace(go.Bar(
        x=["Energia in [J]"],
        y=[ctx["total_in"]],
        name="Energia in",
        marker_color=COLORS["energy_in"],
        text=[f"{ctx['total_in']:.3f} J"],
        textposition="auto",
    ))
    fig_e.add_trace(go.Bar(
        x=["Energia out [J]"],
        y=[ctx["total_out"]],
        name="Energia out",
        marker_color=COLORS["energy_out"],
        text=[f"{ctx['total_out']:.3f} J"],
        textposition="auto",
    ))
    fig_e.update_layout(
        barmode="group",
        height=300,
        template=TEMPLATE,
        yaxis=dict(title="Energia [J]", type="linear"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=50, l=70, r=20),
        annotations=[dict(
            text=f"Efficienza: {ctx['global_eff']:.4f} %",
            xref="paper", yref="paper", x=0.5, y=-0.18,
            showarrow=False, font=dict(size=13, color="#333"),
        )],
    )
    st.plotly_chart(fig_e, use_container_width=True)

with col_b:
    st.markdown("#### Statistiche descrittive")
    sdict = {
        "Spostamento [mm]": df["Ch 1 Displacement"],
        "Forza [N]":        df["Force_N"],
        "Velocità [mm/s]":  df["Velocity_mms"],
        "Tensione [V]":     df["Voltage_V"],
        "Corrente [A]":    df["Current_A"],
        "P_in [W]":         df["Power_in_W"],
        "P_out [W]":       df["Power_out_W"],
    }
    stats = pd.DataFrame({
        k: {
            "Min":    v.min(),
            "Media":  v.mean(),
            "Std":    v.std(),
            "Max":    v.max(),
            "RMS":    float(np.sqrt(np.mean(v**2))),
        }
        for k, v in sdict.items()
    }).T.round(4)
    st.dataframe(stats, use_container_width=True)

# Per-cycle quick overview
st.markdown("---")
st.markdown("#### Riepilogo per ciclo")
cdf = compute_cycle_stats(ctx["filepath"], ctx["cutoff"], ctx["t_start"], ctx["t_end"])
if not cdf.empty:
    cols_show = ["Ciclo", "Disp_pkpk_mm", "Force_pk_N", "Vel_pk_mms",
                 "E_in_J", "E_out_mJ", "Eff_pct"]
    cols_show = [c for c in cols_show if c in cdf.columns]
    rename = {
        "Disp_pkpk_mm": "Disp pk-pk [mm]",
        "Force_pk_N":   "Forza pk [N]",
        "Vel_pk_mms":   "Vel pk [mm/s]",
        "E_in_J":       "E in [J]",
        "E_out_mJ":     "E out [mJ]",
        "Eff_pct":      "Eff. [%]",
    }
    st.dataframe(
        cdf[cols_show].rename(columns=rename).set_index("Ciclo").round(4),
        use_container_width=True,
    )
