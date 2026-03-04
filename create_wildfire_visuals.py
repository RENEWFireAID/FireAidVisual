#!/usr/bin/env python3
"""Create exploratory visuals for Alaska wildfire location points."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter


DATA_PATH = Path("AK_fire_location_points_NAD83.txt")
METADATA_PATH = Path("AlaskaFireHistory_LocationPoints_FGDG_metadata.xml")
OUT_DIR = Path("visuals")


def classify_cause(df: pd.DataFrame) -> pd.Series:
    firecause = df.get("FIRECAUSE", pd.Series(index=df.index, dtype="object"))
    general = df.get("GENERALCAUSE", pd.Series(index=df.index, dtype="object"))

    firecause = firecause.fillna("").astype(str).str.strip()
    general = general.fillna("").astype(str).str.strip()

    unknown_tokens = {"", "?", "rx?", "na", "none", "null"}
    base = firecause.where(~firecause.str.lower().isin(unknown_tokens), general)
    base = base.fillna("").str.strip()
    low = base.str.lower()

    out = pd.Series("Human", index=df.index, dtype="object")

    out[low.eq("") | low.eq("undetermined")] = "Undetermined"
    out[low.str.contains("false alarm", na=False)] = "False Alarm"
    out[(low.str.contains("prescribed", na=False)) | (df.get("PRESCRIBEDFIRE", "N") == "Y")] = "Prescribed"
    out[(low.str.contains("lightning", na=False)) | low.eq("natural")] = "Natural"
    out[low.eq("human")] = "Human"

    return out


def format_big_number(value: float, _position: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def ensure_style() -> None:
    plt.style.use("tableau-colorblind10")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)

    df["FIRESEASON"] = pd.to_numeric(df["FIRESEASON"], errors="coerce")
    df["DISCOVERY_DT"] = pd.to_datetime(df["DISCOVERYDATETIME"], errors="coerce")
    df["DISCOVERY_MONTH"] = df["DISCOVERY_DT"].dt.month

    for col in ["LATITUDE", "LONGITUDE", "ESTIMATEDTOTALACRES", "ACTUALTOTALACRES"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Use estimated total acres for consistency because it has near-complete coverage.
    df["ACRES_FOR_ANALYSIS"] = df["ESTIMATEDTOTALACRES"]
    df["ACRES_FOR_ANALYSIS"] = df["ACRES_FOR_ANALYSIS"].fillna(df["ACTUALTOTALACRES"])
    df["ACRES_FOR_ANALYSIS"] = df["ACRES_FOR_ANALYSIS"].fillna(0).clip(lower=0)

    df["CAUSE_GROUP"] = classify_cause(df)
    df["DECADE"] = (df["FIRESEASON"] // 10) * 10

    return df


def plot_fire_count_by_year(df: pd.DataFrame) -> None:
    yearly = df.dropna(subset=["FIRESEASON"]).groupby("FIRESEASON").size()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(yearly.index, yearly.values, linewidth=1.8, color="#1f77b4", label="Annual fire count")
    ax.plot(yearly.index, yearly.rolling(5, min_periods=1).mean(), linewidth=2.2, color="#d62728", label="5-year moving average")

    ax.set_title("Alaska Fire Incidents by Year (1939-2024)")
    ax.set_xlabel("Fire Season")
    ax.set_ylabel("Number of incidents")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(FuncFormatter(format_big_number))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_fire_count_by_year.png")
    plt.close(fig)


def plot_acres_by_year(df: pd.DataFrame) -> None:
    yearly_acres = (
        df.dropna(subset=["FIRESEASON"])
        .groupby("FIRESEASON")["ACRES_FOR_ANALYSIS"]
        .sum(min_count=1)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(yearly_acres.index, yearly_acres.values, width=0.8, color="#ff7f0e", alpha=0.75)
    ax.plot(
        yearly_acres.index,
        yearly_acres.rolling(5, min_periods=1).mean(),
        color="#2ca02c",
        linewidth=2.0,
        label="5-year moving average",
    )

    ax.set_title("Estimated Acres Burned by Year")
    ax.set_xlabel("Fire Season")
    ax.set_ylabel("Acres")
    ax.yaxis.set_major_formatter(FuncFormatter(format_big_number))
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_acres_burned_by_year.png")
    plt.close(fig)


def plot_monthly_pattern(df: pd.DataFrame) -> None:
    month_order = np.arange(1, 13)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    monthly = (
        df.dropna(subset=["DISCOVERY_MONTH"])
        .groupby(["DISCOVERY_MONTH", "CAUSE_GROUP"])  # type: ignore[arg-type]
        .size()
        .unstack(fill_value=0)
    )
    monthly = monthly.reindex(month_order, fill_value=0)

    preferred_cols = ["Natural", "Human", "False Alarm", "Prescribed", "Undetermined"]
    cols = [c for c in preferred_cols if c in monthly.columns] + [c for c in monthly.columns if c not in preferred_cols]
    monthly = monthly[cols]

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(monthly), dtype=float)

    for col in monthly.columns:
        values = monthly[col].to_numpy()
        ax.bar(month_order, values, bottom=bottom, label=col, width=0.85)
        bottom += values

    ax.set_title("Discovery Month Pattern by Cause Group")
    ax.set_xlabel("Discovery Month")
    ax.set_ylabel("Number of incidents")
    ax.set_xticks(month_order)
    ax.set_xticklabels(month_labels)
    ax.legend(loc="upper right", ncols=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_discovery_month_by_cause.png")
    plt.close(fig)


def plot_cause_share_by_decade(df: pd.DataFrame) -> None:
    cause_decade = (
        df.dropna(subset=["DECADE"])
        .groupby(["DECADE", "CAUSE_GROUP"])  # type: ignore[arg-type]
        .size()
        .unstack(fill_value=0)
    )

    cause_decade = cause_decade.sort_index()
    share = cause_decade.div(cause_decade.sum(axis=1), axis=0) * 100

    preferred_cols = ["Human", "Natural", "False Alarm", "Prescribed", "Undetermined"]
    cols = [c for c in preferred_cols if c in share.columns] + [c for c in share.columns if c not in preferred_cols]
    share = share[cols]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(share.index.astype(int), [share[c].to_numpy() for c in share.columns], labels=share.columns, alpha=0.9)

    ax.set_title("Cause Composition by Decade")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Share of incidents (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", ncols=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_cause_share_by_decade.png")
    plt.close(fig)


def plot_spatial_density(df: pd.DataFrame) -> None:
    spatial = df.dropna(subset=["LONGITUDE", "LATITUDE"]).copy()
    spatial = spatial[(spatial["LONGITUDE"].between(-178, -129)) & (spatial["LATITUDE"].between(51, 71))]

    fig, ax = plt.subplots(figsize=(11, 7))
    hb = ax.hexbin(
        spatial["LONGITUDE"],
        spatial["LATITUDE"],
        gridsize=70,
        bins="log",
        mincnt=1,
        cmap="YlOrRd",
    )

    ax.set_title("Spatial Density of Fire Location Points (log-scaled counts)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("log10(count)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_spatial_density_points.png")
    plt.close(fig)


def plot_spatial_acres(df: pd.DataFrame) -> None:
    spatial = df.dropna(subset=["LONGITUDE", "LATITUDE", "ACRES_FOR_ANALYSIS"]).copy()
    spatial = spatial[(spatial["LONGITUDE"].between(-178, -129)) & (spatial["LATITUDE"].between(51, 71))]

    weights = spatial["ACRES_FOR_ANALYSIS"].to_numpy()
    vmax = float(np.nanmax(weights)) if np.isfinite(np.nanmax(weights)) else 1.0
    vmax = max(vmax, 1.0)

    fig, ax = plt.subplots(figsize=(11, 7))
    hb = ax.hexbin(
        spatial["LONGITUDE"],
        spatial["LATITUDE"],
        C=weights,
        reduce_C_function=np.sum,
        gridsize=70,
        mincnt=1,
        cmap="inferno",
        norm=LogNorm(vmin=1, vmax=vmax),
    )

    ax.set_title("Spatial Concentration of Estimated Burned Acres")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Estimated acres per hex cell (log scale)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_spatial_density_acres.png")
    plt.close(fig)


def write_summary(df: pd.DataFrame) -> None:
    year_min = int(df["FIRESEASON"].min())
    year_max = int(df["FIRESEASON"].max())
    rows = len(df)

    total_acres = float(df["ACRES_FOR_ANALYSIS"].sum())
    median_acres = float(df["ACRES_FOR_ANALYSIS"].median())
    p95_acres = float(df["ACRES_FOR_ANALYSIS"].quantile(0.95))

    top_years_count = (
        df.groupby("FIRESEASON")
        .size()
        .sort_values(ascending=False)
        .head(5)
    )

    top_years_acres = (
        df.groupby("FIRESEASON")["ACRES_FOR_ANALYSIS"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    cause_counts = df["CAUSE_GROUP"].value_counts().head(8)

    lines = [
        "Alaska Fire History Quick Summary",
        "=",
        f"Records: {rows:,}",
        f"Fire seasons covered: {year_min} to {year_max}",
        f"Total estimated acres (analysis field): {total_acres:,.0f}",
        f"Median acres per incident: {median_acres:.2f}",
        f"95th percentile incident size: {p95_acres:,.2f} acres",
        "",
        "Top 5 years by incident count:",
    ]

    for year, count in top_years_count.items():
        lines.append(f"- {int(year)}: {int(count):,} incidents")

    lines.append("")
    lines.append("Top 5 years by estimated acres:")
    for year, acres in top_years_acres.items():
        lines.append(f"- {int(year)}: {acres:,.0f} acres")

    lines.append("")
    lines.append("Cause group counts:")
    for cause, count in cause_counts.items():
        lines.append(f"- {cause}: {int(count):,}")

    lines.append("")
    if METADATA_PATH.exists():
        lines.append(f"Metadata source: {METADATA_PATH.name}")

    (OUT_DIR / "00_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    ensure_style()

    df = load_data()

    plot_fire_count_by_year(df)
    plot_acres_by_year(df)
    plot_monthly_pattern(df)
    plot_cause_share_by_decade(df)
    plot_spatial_density(df)
    plot_spatial_acres(df)
    write_summary(df)

    print("Created visuals in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
