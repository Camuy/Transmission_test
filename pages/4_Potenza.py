"""
pages/4_Potenza.py — Analisi potenza e bilancio energetico
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import sidebar_controls, COLORS, TEMPLATE, HOVER_MODE, compute_cycle_stats

st.set_page_config(
    page_title="Potenza | Transmission Test",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ctx = sidebar_controls()
df  = ctx["df"]
fs  = ctx["fs"]

# ─────────────────────────────────────────────────────────────────────────────
st.title("⚙️ Analisi Potenza — Bilancio Energetico")
st.caption(
    f"Caso: **{ctx['case']}**  ·  "
    f"P_in = F × v  ·  P_out = V × I  ·  "
    f"Durata: **{df['Time'].max()-df['Time'].min():.2f} s**"
)

# ─────────────────────────────────────────────────────────────────────────────
# KPI
# ─────────────────────────────────────────────────────────────────────────────
c = st.columns(6)
c[0].metric("Energia in [J]",    f"{ctx['total_in']:.3f}")
c[1].metric("Energia out [J]",  f"{ctx['total_out']:.3f}")
c[2].metric("P_in picco [W]",    f"{df['Power_in_W'].abs().max():.3f}")
c[3].metric("P_in RMS [W]",      f"{float(np.sqrt(np.mean(df['Power_in_W']**2))):.3f}")
c[4].metric("P_out picco [W]",  f"{df['Power_out_W'].abs().max():.4f}")
c[5].metric("Efficienza glob.",  f"{ctx['global_eff']:.5f} %")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Main linked figure — P_in + P_out + E_in + E_out  (4 panels)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Potenza & Energia — asse temporale collegato")

fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    subplot_titles=[
        "Potenza meccanica in [W]",
        "Potenza elettrica out [W]",
        "Energia cumulativa in [J]",
        "Energia cumulativa out [J]",
    ],
    row_heights=[1.2, 1.2, 1, 1],
)

_t = df["Time"]

# P_in
fig.add_trace(go.Scatter(
    x=_t, y=df["Power_in_W"],
    name="P_in", mode="lines",
    line=dict(color=COLORS["power_in"], width=1.2),
    fill="tozeroy", fillcolor="rgba(21,101,192,0.10)",
    hovertemplate="%{y:.4f} W<extra>P_in</extra>",
), row=1, col=1)

# P_out
fig.add_trace(go.Scatter(
    x=_t, y=df["Power_out_W"],
    name="P_out", mode="lines",
    line=dict(color=COLORS["power_out"], width=1.2),
    fill="tozeroy", fillcolor="rgba(233,30,99,0.10)",
    hovertemplate="%{y:.6f} W<extra>P_out</extra>",
), row=2, col=1)

# E_in cumulative
fig.add_trace(go.Scatter(
    x=_t, y=df["Energy_in_J"],
    name="E_in", mode="lines",
    line=dict(color=COLORS["energy_in"], width=1.5),
    hovertemplate="%{y:.4f} J<extra>E_in</extra>",
), row=3, col=1)

# E_out cumulative
fig.add_trace(go.Scatter(
    x=_t, y=df["Energy_out_J"],
    name="E_out", mode="lines",
    line=dict(color=COLORS["energy_out"], width=1.5),
    hovertemplate="%{y:.6f} J<extra>E_out</extra>",
), row=4, col=1)

for r in range(1, 5):
    fig.add_hline(y=0, line=dict(color="gray", width=0.4, dash="dot"), row=r, col=1)

fig.update_xaxes(
    title_text="Tempo [s]", row=4, col=1,
    rangeslider=dict(visible=True, thickness=0.04),
)
fig.update_yaxes(title_text="W",  row=1, col=1)
fig.update_yaxes(title_text="W", row=2, col=1)
fig.update_yaxes(title_text="J",  row=3, col=1)
fig.update_yaxes(title_text="J", row=4, col=1)

fig.update_layout(
    height=800,
    template=TEMPLATE,
    hovermode=HOVER_MODE,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="right", x=1),
    margin=dict(t=80, b=20, l=70, r=30),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Power distributions comparison
# ─────────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Distribuzione P_in")
    fig_pin = go.Figure()
    fig_pin.add_trace(go.Histogram(
        x=df["Power_in_W"], nbinsx=80,
        marker_color=COLORS["power_in"], opacity=0.8, name="P_in",
    ))
    # Positive vs negative contribution annotation
    pin_pos = df["Power_in_W"][df["Power_in_W"] > 0]
    pin_neg = df["Power_in_W"][df["Power_in_W"] < 0]
    fig_pin.add_vline(x=0, line=dict(color="black", width=1.2, dash="dash"))
    fig_pin.add_annotation(
        x=0.25, y=1, xref="paper", yref="paper",
        text=f"▶ Positiva: {len(pin_pos)/(len(df))*100:.1f}%",
        showarrow=False, font=dict(size=11, color=COLORS["power_in"]),
    )
    fig_pin.add_annotation(
        x=0.0, y=0.92, xref="paper", yref="paper",
        text=f"◀ Negativa: {len(pin_neg)/(len(df))*100:.1f}%",
        showarrow=False, font=dict(size=11, color="#E57373"),
    )
    fig_pin.update_layout(
        height=320, template=TEMPLATE,
        xaxis_title="Potenza in [W]", yaxis_title="Conteggio",
        margin=dict(t=30, b=50, l=70, r=20), showlegend=False,
    )
    st.plotly_chart(fig_pin, use_container_width=True)

with col2:
    st.markdown("#### Distribuzione P_out (positiva)")
    pout_pos = df["Power_out_W"].clip(lower=0)
    fig_pout = go.Figure()
    fig_pout.add_trace(go.Histogram(
        x=pout_pos, nbinsx=80,
        marker_color=COLORS["power_out"], opacity=0.8, name="P_out",
    ))
    fig_pout.update_layout(
        height=320, template=TEMPLATE,
        xaxis_title="Potenza out [W]", yaxis_title="Conteggio",
        margin=dict(t=30, b=50, l=70, r=20), showlegend=False,
    )
    st.plotly_chart(fig_pout, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Per-cycle energy balance
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Bilancio energetico per ciclo")

cdf = compute_cycle_stats(ctx["filepath"], ctx["cutoff"], ctx["t_start"], ctx["t_end"])
if not cdf.empty:
    col3, col4 = st.columns(2)

    with col3:
        fig_ein = go.Figure()
        fig_ein.add_trace(go.Bar(
            x=cdf["Ciclo"], y=cdf["E_in_J"],
            name="E_in",
            marker_color=COLORS["energy_in"],
            hovertemplate="Ciclo %{x}<br>E in: %{y:.4f} J<extra></extra>",
        ))
        fig_ein.update_layout(
            height=300, template=TEMPLATE,
            xaxis_title="Ciclo #", yaxis_title="Energia in [J]",
            title="Energia meccanica in ingresso per ciclo",
            margin=dict(t=50, b=50, l=70, r=20),
        )
        st.plotly_chart(fig_ein, use_container_width=True)

    with col4:
        fig_eout = go.Figure()
        fig_eout.add_trace(go.Bar(
            x=cdf["Ciclo"], y=cdf["E_out_J"],
            name="E_out",
            marker_color=COLORS["energy_out"],
            hovertemplate="Ciclo %{x}<br>E out: %{y:.4f} J<extra></extra>",
        ))
        fig_eout.update_layout(
            height=300, template=TEMPLATE,
            xaxis_title="Ciclo #", yaxis_title="Energia elettrica out [J]",
            title="Energia elettrica estratta per ciclo",
            margin=dict(t=50, b=50, l=70, r=20),
        )
        st.plotly_chart(fig_eout, use_container_width=True)

    # ── Power-velocity scatter: P_in vs vel (mechanical operating point) ──
    st.markdown("#### Punto di lavoro meccanico — P_in vs velocità")
    fig_pv = go.Figure()
    fig_pv.add_trace(go.Scatter(
        x=df["Velocity_mms"],
        y=df["Power_in_W"],
        mode="markers",
        marker=dict(
            color=df["Force_N"],
            colorscale="RdBu",
            size=2,
            opacity=0.5,
            colorbar=dict(title="Forza [N]", thickness=12),
        ),
        hovertemplate="Vel: %{x:.2f} mm/s<br>P_in: %{y:.3f} W<extra></extra>",
        name="",
    ))
    fig_pv.add_vline(x=0, line=dict(color="gray", width=0.5, dash="dot"))
    fig_pv.add_hline(y=0, line=dict(color="gray", width=0.5, dash="dot"))
    fig_pv.update_layout(
        height=350, template=TEMPLATE,
        xaxis_title="Velocità [mm/s]",
        yaxis_title="Potenza in [W]",
        margin=dict(t=20, b=50, l=70, r=20),
    )
    st.plotly_chart(fig_pv, use_container_width=True)
