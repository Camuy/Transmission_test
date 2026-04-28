"""
pages/5_Efficienza.py — Analisi dell'efficienza di trasmissione
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

from utils import sidebar_controls, COLORS, TEMPLATE, HOVER_MODE, compute_cycle_stats

st.set_page_config(
    page_title="Efficienza | Transmission Test",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ctx = sidebar_controls()
df  = ctx["df"]
fs  = ctx["fs"]

# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 Efficienza di Trasmissione")
st.caption(
    f"Caso: **{ctx['case']}**  ·  "
    f"η = E_out / E_in × 100  ·  "
    f"Soglia P_in: 0.01 W  ·  Durata: **{df['Time'].max()-df['Time'].min():.2f} s**"
)

# ─────────────────────────────────────────────────────────────────────────────
# KPI
# ─────────────────────────────────────────────────────────────────────────────
eff_valid = df["Efficiency_pct"].dropna()
c = st.columns(6)
c[0].metric("Efficienza globale [%]",  f"{ctx['global_eff']:.3f}")
c[1].metric("η istant. media [%]",     f"{eff_valid.mean():.3f}" if not eff_valid.empty else "—")
c[2].metric("η istant. max [%]",       f"{eff_valid.max():.3f}"  if not eff_valid.empty else "—")
c[3].metric("Energia in [J]",          f"{ctx['total_in']:.3f}")
c[4].metric("Energia out [J]",        f"{ctx['total_out']:.3f}")
c[5].metric("Cicli analizzati",        f"{ctx['n_cycles']}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Main linked figure — Eff inst + Eff cumul + P_in (reference)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Efficienza nel tempo — asse temporale collegato")
st.caption("Il pannello P_in serve come riferimento per correlare l'efficienza all'andamento della potenza.")

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=[
        "Efficienza istantanea [%]",
        "Efficienza cumulativa [%]",
        "Potenza in [W] — riferimento",
    ],
    row_heights=[1.2, 1.2, 0.8],
)

_t = df["Time"]

# Efficiency instantaneous
fig.add_trace(go.Scatter(
    x=_t[eff_valid.index], y=eff_valid,
    name="η istant.", mode="lines",
    line=dict(color=COLORS["efficiency"], width=0.8),
    opacity=0.7,
    hovertemplate="%{y:.4f} %<extra>η istant.</extra>",
), row=1, col=1)
# Zero reference
fig.add_hline(y=ctx["global_eff"], row=1, col=1,
              line=dict(color="orange", width=1.5, dash="dash"),
              annotation_text=f"η glob = {ctx['global_eff']:.4f}%",
              annotation_position="top right")
#fig.update_yaxes(range=[-2, min(50, eff_valid.quantile(0.99) * 1.5 if not eff_valid.empty else 50)],
#                 row=1, col=1)

# Cumulative efficiency
fig.add_trace(go.Scatter(
    x=_t, y=df["Cumul_Eff_pct"],
    name="η cum.", mode="lines",
    line=dict(color=COLORS["eff_cum"], width=1.5),
    hovertemplate="%{y:.5f} %<extra>η cum.</extra>",
), row=2, col=1)
fig.add_hline(y=ctx["global_eff"], row=2, col=1,
              line=dict(color="orange", width=1.2, dash="dash"))

# P_in reference
fig.add_trace(go.Scatter(
    x=_t, y=df["Power_in_W"],
    name="P_in ref", mode="lines",
    line=dict(color=COLORS["power_in"], width=1),
    fill="tozeroy", fillcolor="rgba(21,101,192,0.08)",
    hovertemplate="%{y:.3f} W<extra>P_in</extra>",
), row=3, col=1)
fig.add_hline(y=0, row=3, col=1, line=dict(color="gray", width=0.4, dash="dot"))

fig.update_xaxes(
    title_text="Tempo [s]", row=3, col=1,
    rangeslider=dict(visible=True, thickness=0.04),
)
fig.update_yaxes(title_text="%", row=1, col=1)
fig.update_yaxes(title_text="%", row=2, col=1)
fig.update_yaxes(title_text="W", row=3, col=1)

fig.update_layout(
    height=720,
    template=TEMPLATE,
    hovermode=HOVER_MODE,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="right", x=1),
    margin=dict(t=80, b=20, l=70, r=30),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Distribution + per-cycle efficiency
# ─────────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Distribuzione efficienza istantanea")
    eff_clip = eff_valid.clip(0, 100)
    fig_dist = go.Figure()

    fig_dist.add_trace(go.Histogram(
        x=eff_clip, nbinsx=70,
        marker_color=COLORS["efficiency"], opacity=0.75,
        name="η [%]",
    ))
    # KDE
    if eff_clip.std() > 1e-10:
        kde_x = np.linspace(eff_clip.min(), eff_clip.max(), 400)
        kde_y = scipy_stats.gaussian_kde(eff_clip)(kde_x)
        norm  = len(eff_clip) * (eff_clip.max() - eff_clip.min()) / 70
        fig_dist.add_trace(go.Scatter(
            x=kde_x, y=kde_y * norm,
            mode="lines", line=dict(color="#1B5E20", width=2),
            name="KDE",
        ))

    fig_dist.add_vline(
        x=ctx["global_eff"],
        line=dict(color="orange", width=1.5, dash="dash"),
        annotation_text=f"η glob: {ctx['global_eff']:.4f}%",
        annotation_position="top right",
    )
    fig_dist.update_layout(
        height=360, template=TEMPLATE,
        xaxis_title="Efficienza [%]",
        yaxis_title="Conteggio campioni",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, b=50, l=70, r=20),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    st.markdown("#### Efficienza per ciclo")
    cdf = compute_cycle_stats(ctx["filepath"], ctx["cutoff"], ctx["t_start"], ctx["t_end"])

    if not cdf.empty and "Eff_pct" in cdf.columns:
        valid_eff = cdf.dropna(subset=["Eff_pct"])
        fig_cyc = go.Figure()
        fig_cyc.add_trace(go.Bar(
            x=valid_eff["Ciclo"],
            y=valid_eff["Eff_pct"],
            marker_color=COLORS["efficiency"],
            hovertemplate="Ciclo %{x}<br>η: %{y:.4f} %<extra></extra>",
            name="η per ciclo",
        ))
        mean_eff = valid_eff["Eff_pct"].mean()
        fig_cyc.add_hline(
            y=mean_eff,
            line=dict(color="orange", width=1.5, dash="dash"),
            annotation_text=f"Media: {mean_eff:.4f}%",
            annotation_position="top right",
        )
        fig_cyc.update_layout(
            height=360, template=TEMPLATE,
            xaxis_title="Ciclo #",
            yaxis_title="Efficienza [%]",
            margin=dict(t=30, b=50, l=70, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig_cyc, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Efficiency vs displacement amplitude + vs cycle energy in
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Correlazioni con l'efficienza per ciclo")

cdf = compute_cycle_stats(ctx["filepath"], ctx["cutoff"], ctx["t_start"], ctx["t_end"])
if not cdf.empty and "Eff_pct" in cdf.columns:
    valid_c = cdf.dropna(subset=["Eff_pct"])

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### η vs ampiezza spostamento")
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=valid_c["Disp_pkpk_mm"],
            y=valid_c["Eff_pct"],
            mode="markers+text",
            marker=dict(
                color=valid_c["Ciclo"],
                colorscale="Viridis",
                size=9,
                opacity=0.8,
                colorbar=dict(title="Ciclo", thickness=12),
            ),
            text=valid_c["Ciclo"].astype(str),
            textposition="top center",
            textfont=dict(size=8),
            hovertemplate="Ciclo %{text}<br>Disp pk-pk: %{x:.2f} mm<br>η: %{y:.4f}%<extra></extra>",
        ))
        fig_s.update_layout(
            height=340, template=TEMPLATE,
            xaxis_title="Spostamento pk-pk [mm]",
            yaxis_title="Efficienza [%]",
            margin=dict(t=20, b=50, l=70, r=20),
        )
        st.plotly_chart(fig_s, use_container_width=True)

    with col4:
        st.markdown("#### η vs Energia in per ciclo")
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            x=valid_c["E_in_J"],
            y=valid_c["Eff_pct"],
            mode="markers+text",
            marker=dict(
                color=valid_c["Ciclo"],
                colorscale="Viridis",
                size=9,
                opacity=0.8,
                showscale=False,
            ),
            text=valid_c["Ciclo"].astype(str),
            textposition="top center",
            textfont=dict(size=8),
            hovertemplate="Ciclo %{text}<br>E_in: %{x:.4f} J<br>η: %{y:.4f}%<extra></extra>",
        ))
        fig_e.update_layout(
            height=340, template=TEMPLATE,
            xaxis_title="Energia in [J]",
            yaxis_title="Efficienza [%]",
            margin=dict(t=20, b=50, l=70, r=20),
        )
        st.plotly_chart(fig_e, use_container_width=True)

    # Full table
    st.markdown("#### Tabella per ciclo — efficienza completa")
    rename = {
        "Ciclo": "Ciclo",
        "Disp_pkpk_mm": "Disp pk-pk [mm]",
        "Force_pk_N":   "Forza pk [N]",
        "Vel_pk_mms":   "Vel pk [mm/s]",
        "E_in_J":       "E_in [J]",
        "E_out_mJ":     "E_out [mJ]",
        "Eff_pct":      "η [%]",
        "Work_mec_J":   "Lavoro mec. [J]",
        "Pout_pk_mW":   "P_out pk [mW]",
    }
    show_cols = [k for k in rename if k in cdf.columns]
    st.dataframe(
        cdf[show_cols].rename(columns=rename).set_index("Ciclo").round(5),
        use_container_width=True,
    )
