"""
charts.py
=========
Reusable Plotly chart factory for the Food Delivery Analytics dashboard.

Each function accepts a DataFrame (or grouped DataFrame) and returns a
plotly.graph_objects.Figure ready for st.plotly_chart().

Design principles:
  - Consistent colour palette across all charts
  - All chart titles and axis labels are descriptive
  - Scatter plots on large data use a random 5,000-row sample for performance
  - No hardcoded data — all values come from the DataFrame
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Consistent colour palette ────────────────────────────────────────────────
PRIMARY_COLOR = "#3b82d4"
ACCENT_COLORS = px.colors.qualitative.Plotly
SEQUENTIAL_COLORSCALE = "Blues"
LAYOUT_FONT = dict(family="Segoe UI, Arial, sans-serif", size=13, color="#1f2328")
PLOT_BG = "#ffffff"
PAPER_BG = "#f7f8fa"
SCATTER_SAMPLE_SIZE = 5_000   # max rows for scatter plots


def _base_layout(title: str, height: int = 420) -> dict:
    """Return a consistent Plotly layout dict."""
    return dict(
        title=dict(text=title, font=dict(size=15, color="#1f2328")),
        font=LAYOUT_FONT,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        margin=dict(l=50, r=30, t=50, b=50),
        height=height,
    )


# ── Line Chart ────────────────────────────────────────────────────────────────

def line_chart(df: pd.DataFrame, x: str, y: str, title: str,
               x_label: str = None, y_label: str = None, height: int = 380) -> go.Figure:
    """Simple line chart for time-series or trend data."""
    fig = px.line(df, x=x, y=y, markers=True, color_discrete_sequence=[PRIMARY_COLOR])
    fig.update_layout(
        **_base_layout(title, height),
        xaxis_title=x_label or x,
        yaxis_title=y_label or y,
    )
    fig.update_traces(line=dict(width=2))
    return fig


# ── Bar Chart (vertical) ──────────────────────────────────────────────────────

def bar_chart(df: pd.DataFrame, x: str, y: str, title: str,
              x_label: str = None, y_label: str = None,
              color: str = PRIMARY_COLOR, height: int = 380) -> go.Figure:
    """Vertical bar chart."""
    fig = px.bar(df, x=x, y=y, color_discrete_sequence=[color])
    fig.update_layout(
        **_base_layout(title, height),
        xaxis_title=x_label or x,
        yaxis_title=y_label or y,
        showlegend=False,
    )
    return fig


# ── Horizontal Bar Chart ──────────────────────────────────────────────────────

def hbar_chart(df: pd.DataFrame, x: str, y: str, title: str,
               x_label: str = None, y_label: str = None,
               color: str = PRIMARY_COLOR, height: int = 380) -> go.Figure:
    """Horizontal bar chart — good for zone/category comparisons."""
    fig = px.bar(df, x=x, y=y, orientation="h", color_discrete_sequence=[color])
    fig.update_layout(
        **_base_layout(title, height),
        xaxis_title=x_label or x,
        yaxis_title=y_label or y,
        showlegend=False,
        yaxis=dict(categoryorder="total ascending"),
    )
    return fig


# ── Box Plot ──────────────────────────────────────────────────────────────────

def box_plot(df: pd.DataFrame, x: str, y: str, title: str,
             x_label: str = None, y_label: str = None,
             height: int = 420, category_order: list = None) -> go.Figure:
    """Box plot showing distribution of y grouped by x."""
    kwargs = {}
    if category_order:
        kwargs["category_orders"] = {x: category_order}
    fig = px.box(df, x=x, y=y, color=x,
                 color_discrete_sequence=ACCENT_COLORS, **kwargs)
    fig.update_layout(
        **_base_layout(title, height),
        xaxis_title=x_label or x,
        yaxis_title=y_label or y,
        showlegend=False,
    )
    return fig


# ── Histogram ─────────────────────────────────────────────────────────────────

def histogram(df: pd.DataFrame, x: str, title: str,
              x_label: str = None, nbins: int = 40, height: int = 380) -> go.Figure:
    """Distribution histogram."""
    fig = px.histogram(df, x=x, nbins=nbins, color_discrete_sequence=[PRIMARY_COLOR])
    fig.update_layout(
        **_base_layout(title, height),
        xaxis_title=x_label or x,
        yaxis_title="Count",
        bargap=0.05,
        showlegend=False,
    )
    return fig


# ── Scatter Plot ──────────────────────────────────────────────────────────────

def scatter_plot(df: pd.DataFrame, x: str, y: str, title: str,
                 x_label: str = None, y_label: str = None,
                 color_col: str = None, add_trendline: bool = False,
                 height: int = 420) -> go.Figure:
    """
    Scatter plot.  Automatically samples up to SCATTER_SAMPLE_SIZE rows
    for large datasets to maintain dashboard performance.
    """
    if len(df) > SCATTER_SAMPLE_SIZE:
        plot_df = df.sample(SCATTER_SAMPLE_SIZE, random_state=42)
        note = f"(showing {SCATTER_SAMPLE_SIZE:,} random sample of {len(df):,} records)"
    else:
        plot_df = df
        note = ""

    trend = "ols" if add_trendline else None
    kwargs = {}
    if color_col and color_col in plot_df.columns:
        kwargs["color"] = color_col
        kwargs["color_discrete_sequence"] = ACCENT_COLORS
    else:
        kwargs["color_discrete_sequence"] = [PRIMARY_COLOR]

    fig = px.scatter(plot_df, x=x, y=y, trendline=trend, opacity=0.45, **kwargs)
    full_title = f"{title} {note}".strip()
    fig.update_layout(
        **_base_layout(full_title, height),
        xaxis_title=x_label or x,
        yaxis_title=y_label or y,
    )
    return fig


# ── Donut / Pie Chart ─────────────────────────────────────────────────────────

def donut_chart(df: pd.DataFrame, names: str, values: str, title: str,
                height: int = 380) -> go.Figure:
    """Donut chart for proportional data."""
    fig = px.pie(df, names=names, values=values, hole=0.4,
                 color_discrete_sequence=ACCENT_COLORS)
    fig.update_layout(
        **_base_layout(title, height),
        showlegend=True,
    )
    fig.update_traces(textinfo="percent+label")
    return fig


# ── Grouped Bar Chart ─────────────────────────────────────────────────────────

def grouped_bar_chart(df: pd.DataFrame, x: str, y: str, color: str,
                      title: str, x_label: str = None, y_label: str = None,
                      height: int = 400) -> go.Figure:
    """Grouped bar chart (e.g. weekend vs weekday across metrics)."""
    fig = px.bar(df, x=x, y=y, color=color, barmode="group",
                 color_discrete_sequence=ACCENT_COLORS)
    fig.update_layout(
        **_base_layout(title, height),
        xaxis_title=x_label or x,
        yaxis_title=y_label or y,
    )
    return fig


# ── Heatmap ───────────────────────────────────────────────────────────────────

def heatmap(df_pivot: pd.DataFrame, title: str,
            x_label: str = "", y_label: str = "",
            height: int = 420) -> go.Figure:
    """
    Heatmap from a pre-pivoted DataFrame.
    df_pivot should have rows as y-axis and columns as x-axis.
    """
    fig = go.Figure(data=go.Heatmap(
        z=df_pivot.values,
        x=df_pivot.columns.tolist(),
        y=df_pivot.index.tolist(),
        colorscale=SEQUENTIAL_COLORSCALE,
        colorbar=dict(title="Value"),
    ))
    fig.update_layout(
        **_base_layout(title, height),
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    return fig


# ── Metric Comparison Bar ────────────────────────────────────────────────────

def metric_comparison_bar(categories: list, values: list, title: str,
                           overall_avg: float = None, height: int = 380) -> go.Figure:
    """
    Bar chart comparing metric values across categories,
    with an optional dashed 'Overall Average' reference line.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=[PRIMARY_COLOR] * len(categories),
        name="Avg Delivery Time",
    ))
    if overall_avg is not None:
        fig.add_shape(
            type="line",
            x0=-0.5, x1=len(categories) - 0.5,
            y0=overall_avg, y1=overall_avg,
            line=dict(color="#e05c32", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=len(categories) - 1,
            y=overall_avg,
            text=f"Overall avg: {overall_avg:.1f} min",
            showarrow=False,
            font=dict(color="#e05c32", size=11),
            yshift=10,
        )
    fig.update_layout(
        **_base_layout(title, height),
        xaxis_title="Category",
        yaxis_title="Avg Delivery Time (min)",
        showlegend=False,
    )
    return fig
