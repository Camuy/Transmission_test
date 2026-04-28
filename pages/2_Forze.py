"""
pages/2_Forze.py — Analisi delle forze e loop d'isteresi
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc

from utils import sidebar_controls, COLORS, TEMPLATE, HOVER_MODE, welch_psd, compute_cycle_stats

st.set_page_config(
    page_title="Forze | Transmission Test",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

ctx = sidebar_controls()
df  = ctx["df"]
fs  = ctx["fs"]

# ─────────────────────────────────────────────────────────────────────────────
st.title("🔩 Analisi Forze — Loop d'Isteresi e Spettro")
st.caption(
    f"Caso: **{ctx['case']}**  ·  Fs: **{fs:.1f} Hz**  ·  "
    f"Freq. test: **{ctx['test_freq']:.3f} Hz**  ·  Cicli: **{ctx['n_cycles']}**"
)

# ─────────────────────────────────────────────────────────────────────────────
# KPI
# ─────────────────────────────────────────────────────────────────────────────
c = st.columns(5)
c[0].metric("Forza picco [N]",    f"{df['Force_N'].abs().max():.2f}")
c[1].metric("Forza picco+ [N]",   f"{df['Force_N'].max():.2f}")
c[2].metric("Forza picco− [N]",   f"{df['Force_N'].min():.2f}")
c[3].metric("Forza RMS [N]",      f"{float(np.sqrt(np.mean(df['Force_N']**2))):.3f}")
c[4].metric("Forza media [N]",    f"{df['Force_N'].mean():.3f}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Main linked figure — Force + Force Abs Error
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Forza nel tempo — asse temporale collegato")

n_rows = 3 if "Ch 1 Force Error" in df.columns else 2
sub_titles = ["Forza applicata [N]", "Errore di forza [N]", "Errore assoluto [N]"][:n_rows]

fig = make_subplots(
    rows=n_rows, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=sub_titles,
)

fig.add_trace(go.Scatter(
    x=df["Time"], y=df["Force_N"],
    name="Forza", mode="lines",
    line=dict(color=COLORS["force"], width=1.2),
    fill="tozeroy", fillcolor="rgba(156,39,176,0.07)",
    hovertemplate="%{y:.3f} N<extra>Forza</extra>",
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=df["Time"], y=df["Force_rms_N"],
    name="RMS", mode="lines",
    line=dict(color=COLORS["rms"], width=1.2, dash="dash"),
    hovertemplate="%{y:.3f} N<extra>RMS</extra>",
), row=1, col=1)

if "Ch 1 Force Error" in df.columns:
    fig.add_trace(go.Scatter(
        x=df["Time"], y=df["Ch 1 Force Error"] * 1e3,
        name="Errore forza", mode="lines",
        line=dict(color="#E57373", width=1),
        hovertemplate="%{y:.4f} N<extra>Errore</extra>",
    ), row=2, col=1)

if "Ch 1 Force Abs. Error" in df.columns:
    fig.add_trace(go.Scatter(
        x=df["Time"], y=df["Ch 1 Force Abs. Error"] * 1e3,
        name="|Errore|", mode="lines",
        line=dict(color="#FF8A65", width=1),
        fill="tozeroy", fillcolor="rgba(255,138,101,0.1)",
        hovertemplate="%{y:.4f} N<extra>|Errore|</extra>",
    ), row=n_rows, col=1)

for r in range(1, n_rows + 1):
    fig.add_hline(y=0, line=dict(color="gray", width=0.4, dash="dot"), row=r, col=1)

fig.update_xaxes(
    title_text="Tempo [s]", row=n_rows, col=1,
    rangeslider=dict(visible=True, thickness=0.04),
)
fig.update_yaxes(title_text="N", row=1, col=1)
fig.update_layout(
    height=580,
    template=TEMPLATE,
    hovermode=HOVER_MODE,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="right", x=1),
    margin=dict(t=80, b=20, l=70, r=30),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Hysteresis loop + PSD
# ─────────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Loop d'isteresi F–d (colorato per ciclo)")

    # Color each cycle differently
    cdf_raw = compute_cycle_stats(ctx["filepath"], ctx["cutoff"], ctx["t_start"], ctx["t_end"])
    n_c = max(1, len(cdf_raw))
    palette = pc.sample_colorscale("Turbo", [i / max(n_c - 1, 1) for i in range(n_c)])

    fig_hys = go.Figure()

    if "Ch 1 Count" in df.columns and not cdf_raw.empty:
        cycle_col = df["Ch 1 Count"].ffill().astype(int)
        for idx, (cyc_id, grp) in enumerate(df.groupby(cycle_col)):
            if len(grp) < 5:
                continue
            clr = palette[min(idx, len(palette) - 1)]
            fig_hys.add_trace(go.Scatter(
                x=grp["Ch 1 Displacement"],
                y=grp["Force_N"],
                mode="lines",
                line=dict(color=clr, width=1),
                name=f"C{cyc_id}",
                showlegend=False,
                hovertemplate=f"Ciclo {cyc_id}<br>Disp: %{{x:.2f}} mm<br>F: %{{y:.2f}} N<extra></extra>",
            ))
    else:
        # fallback: color by time
        fig_hys.add_trace(go.Scatter(
            x=df["Ch 1 Displacement"],
            y=df["Force_N"],
            mode="markers",
            marker=dict(
                color=df["Time"], colorscale="Turbo",
                size=2, opacity=0.7,
                colorbar=dict(title="Tempo [s]", thickness=12),
            ),
            hovertemplate="Disp: %{x:.2f} mm<br>F: %{y:.2f} N<extra></extra>",
        ))

    # Colorscale legend bar (dummy scatter for scale reference)
    fig_hys.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(
            colorscale="Turbo",
            color=[0, n_c],
            colorbar=dict(title="Ciclo #", thickness=12),
            showscale=True,
        ),
        showlegend=False,
    ))

    fig_hys.add_vline(x=0, line=dict(color="gray", width=0.5, dash="dot"))
    fig_hys.add_hline(y=0, line=dict(color="gray", width=0.5, dash="dot"))
    fig_hys.update_layout(
        height=400, template=TEMPLATE,
        xaxis_title="Spostamento [mm]",
        yaxis_title="Forza [N]",
        margin=dict(t=30, b=50, l=70, r=20),
    )
    st.plotly_chart(fig_hys, use_container_width=True)

with col2:
    st.markdown("#### Densità spettrale di potenza — Forza")

    f_f, p_f = welch_psd(df["Force_N"], fs)
    dom_f = f_f[np.argmax(p_f)]

    fig_psd = go.Figure()
    fig_psd.add_trace(go.Scatter(
        x=f_f, y=10 * np.log10(p_f + 1e-20),
        mode="lines",
        line=dict(color=COLORS["force"], width=1.5),
        fill="tozeroy", fillcolor="rgba(156,39,176,0.08)",
        hovertemplate="f: %{x:.3f} Hz<br>PSD: %{y:.1f} dB<extra></extra>",
    ))
    fig_psd.add_vline(
        x=dom_f, line=dict(color="red", width=1.2, dash="dash"),
        annotation_text=f"Freq. dom: {dom_f:.3f} Hz",
        annotation_position="top right",
    )
    fig_psd.update_layout(
        height=400, template=TEMPLATE,
        xaxis_title="Frequenza [Hz]",
        yaxis_title="PSD [dB re N²/Hz]",
        margin=dict(t=30, b=50, l=70, r=20),
        showlegend=False,
    )
    st.plotly_chart(fig_psd, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Per-cycle force stats
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Statistiche forza per ciclo")

cdf = compute_cycle_stats(ctx["filepath"], ctx["cutoff"], ctx["t_start"], ctx["t_end"])
if not cdf.empty:
    col3, col4 = st.columns(2)

    with col3:
        fig_fpk = go.Figure()
        fig_fpk.add_trace(go.Bar(
            x=cdf["Ciclo"], y=cdf["Force_pk_N"],
            marker_color=COLORS["force"],
            hovertemplate="Ciclo %{x}<br>Forza pk: %{y:.2f} N<extra></extra>",
        ))
        fig_fpk.update_layout(
            height=300, template=TEMPLATE,
            xaxis_title="Ciclo #", yaxis_title="Forza picco [N]",
            title="Forza di picco per ciclo",
            margin=dict(t=50, b=50, l=70, r=20),
        )
        st.plotly_chart(fig_fpk, use_container_width=True)

    with col4:
        fig_wk = go.Figure()
        fig_wk.add_trace(go.Bar(
            x=cdf["Ciclo"], y=cdf["Work_mec_J"] * 1e3,
            marker_color="#7B1FA2",
            hovertemplate="Ciclo %{x}<br>Lavoro: %{y:.3f} mJ<extra></extra>",
        ))
        fig_wk.update_layout(
            height=300, template=TEMPLATE,
            xaxis_title="Ciclo #", yaxis_title="Lavoro meccanico [mJ]",
            title="Lavoro meccanico (area loop F–d) per ciclo",
            margin=dict(t=50, b=50, l=70, r=20),
        )
        st.plotly_chart(fig_wk, use_container_width=True)

    # Distribution
    st.markdown("#### Distribuzione forza")
    fig_fhist = go.Figure()
    fig_fhist.add_trace(go.Histogram(
        x=df["Force_N"], nbinsx=80,
        marker_color=COLORS["force"], opacity=0.75,
    ))
    fig_fhist.update_layout(
        height=260, template=TEMPLATE,
        xaxis_title="Forza [N]", yaxis_title="Conteggio",
        margin=dict(t=20, b=50, l=70, r=20), showlegend=False,
    )
    st.plotly_chart(fig_fhist, use_container_width=True)
