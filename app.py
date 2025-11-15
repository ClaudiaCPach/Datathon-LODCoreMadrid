import os
import random
import math
from typing import Dict, List, Optional, Literal, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------
# Page configuration & CSS
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Living on the Edge",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --color-slate: #3D517B;
    --color-primary: #568EE2;
    --color-navy: #051C33;
    --color-teal: #6FB5BA;
    --color-sand: #DFD1B6;
    --color-lilac: #A59FD0;
    --color-error: #C33241;
    --color-success: #377F86;
    --color-warning: #C35309;
}

/* App background + default text colour (light background, dark text) */
.stApp {
    background: linear-gradient(180deg, #ffffff 0%, var(--color-sand) 100%);
    color: var(--color-navy);
}

/* Main header */
.main-header {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--color-primary), var(--color-slate));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1rem;
}

/* Tagline under header */
.tagline {
    color: var(--color-navy);
    font-style: italic;
    text-align: center;
    margin-top: 0;
    margin-bottom: 1.5rem;
}

/* Sidebar */
.sidebar .sidebar-content {
    background: linear-gradient(180deg, var(--color-navy) 0%, #111827 100%);
}
.sidebar .sidebar-content,
.sidebar .sidebar-content * {
    color: #F9FAFB !important;
}

/* Municipality cards */
.municipality-card {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(5, 28, 51, 0.12);
    margin: 1rem 0;
    border-left: 5px solid var(--color-primary);
    transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
}
.municipality-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(5, 28, 51, 0.2);
}

/* Score badge */
.score-badge {
    background: linear-gradient(90deg, var(--color-primary), var(--color-teal));
    color: #ffffff;
    padding: 0.5rem 1rem;
    border-radius: 999px;
    font-weight: 600;
    display: inline-block;
    margin: 0.3rem 0;
    box-shadow: 0 4px 10px rgba(5, 28, 51, 0.25);
    font-size: 0.9rem;
}

/* Municipality title */
.municipality-name {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--color-slate);
    margin-bottom: 0.8rem;
}

/* Detail panel */
.detail-panel {
    background: #ffffff;
    border: 1px solid rgba(61, 81, 123, 0.15);
    border-radius: 15px;
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: 0 10px 25px rgba(5, 28, 51, 0.15);
}

/* Icons in detail list */
.concept-icon {
    font-size: 1.3rem;
    margin-right: 0.5rem;
}

/* Sidebar section background cards */
.metric-container {
    background: linear-gradient(145deg, #111827, #1f2937);
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    border-left: 4px solid var(--color-success);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, var(--color-primary), var(--color-teal));
    color: #ffffff;
    border: none;
    border-radius: 999px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    transition: all 0.2s ease;
    box-shadow: 0 4px 10px rgba(5, 28, 51, 0.35);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(5, 28, 51, 0.45);
}

/* Comparison header */
.comparison-header {
    background: var(--color-slate);
    color: #ffffff;
    padding: 0.8rem;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: 600;
}

/* Plot container */
.plot-container {
    border-radius: 16px;
    overflow: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Criteria and column mappings
# ---------------------------------------------------------------------
CRITERIA: List[str] = [
    "AccessibilityHoursMonthly",  # cost (lower is better)
    "EducationQuality",          # benefit
    "AirQuality",                # benefit
    "BuildingQuality",           # benefit
    "TransportInfraQuality",     # benefit
    "EconomicDynamism",          # benefit
    "HousePriceSqm",             # cost (lower is better)
]

CRITERIA_LABELS: Dict[str, str] = {
    "AccessibilityHoursMonthly": "Ahorro de tiempo en desplazamientos",
    "EducationQuality": "Calidad de la educación",
    "AirQuality": "Calidad del aire y del entorno",
    "BuildingQuality": "Atractividad de las viviendas",
    "TransportInfraQuality": "Calidad de las infraestructuras de transporte",
    "EconomicDynamism": "Dinamismo económico",
    "HousePriceSqm": "Precio de la vivienda (€/m²)",
}

CRITERIA_ICONS: Dict[str, str] = {
    "AccessibilityHoursMonthly": "⌛",
    "EducationQuality": "🎓",
    "AirQuality": "🌬️",
    "BuildingQuality": "🏠",
    "TransportInfraQuality": "🚆",
    "EconomicDynamism": "💼",
    "HousePriceSqm": "💰",
}

BENEFIT_COLUMNS: Dict[str, str] = {
    "EducationQuality": "ATR_ServiciosDeEducacion_ClusterEstadistica",
    "AirQuality": "ATR_CalidadDelAire_ClusterEstadistica",
    "BuildingQuality": "ATR_AtractividadDeLosInmuebles_ClusterEstadistica",
    "TransportInfraQuality": "ATR_AtractividadDeLasInfraestructurasDeTransporte_ClusterEstadistica",
    "EconomicDynamism": "ATR_DinamismosEconomico_ClusterEstadistica",
}

COST_COLUMNS: Dict[str, str] = {
    "HousePriceSqm": "IDE_PrecioPorMetroCuadrado",
}

ACC_COLUMNS: Dict[str, Dict[str, str]] = {
    "sport": {
        "coche": "ACC_deporte_tiempo_coche",
        "TransportePublico": "ACC_deporte_tiempo_TransportePublico",
    },
    "gp": {
        "coche": "ACC_sanidad_tiempo_coche_OfertaAsistencial_MedicinaGeneralDeFamilia",
        "TransportePublico": "ACC_sanidad_tiempo_TransportePublico_OfertaAsistencial_MedicinaGeneralDeFamilia",
    },
    "pharmacy": {
        "coche": "ACC_farmacias_tiempo_coche",
        "TransportePublico": "ACC_farmacias_tiempo_TransportePublico",
    },
    "gas": {
        "coche": "ACC_gasolineras_tiempo_coche",
        "TransportePublico": "ACC_gasolineras_tiempo_coche",
    },
    "supermarket": {
        "coche": "OSM_supermercados_tiempo_coche",
        "TransportePublico": "OSM_supermercados_tiempo_TransportePublico",
    },
    # Education – public vs public+private
    "edu_preinf_public": {
        "coche": "ACC_educacion_tiempo_coche_Preinfantil_Publicos",
        "TransportePublico": "ACC_educacion_tiempo_TransportePublico_Preinfantil_Publicos",
    },
    "edu_preinf_pubpriv": {
        "coche": "ACC_educacion_tiempo_coche_Preinfantil_Publicos",
        "TransportePublico": "ACC_educacion_tiempo_TransportePublico_Preinfantil_PublicosPrivados",
    },
    "edu_inf_public": {
        "coche": "ACC_educacion_tiempo_coche_Infantil_Publicos",
        "TransportePublico": "ACC_educacion_tiempo_TransportePublico_Infantil_Publicos",
    },
    "edu_inf_pubpriv": {
        "coche": "ACC_educacion_tiempo_coche_Infantil_PublicosPrivados",
        "TransportePublico": "ACC_educacion_tiempo_TransportePublico_Infantil_PublicosPrivados",
    },
    "edu_prim_public": {
        "coche": "ACC_educacion_tiempo_coche_Primaria_Publicos",
        "TransportePublico": "ACC_educacion_tiempo_TransportePublico_Primaria_Publicos",
    },
    "edu_prim_pubpriv": {
        "coche": "ACC_educacion_tiempo_coche_Primaria_PublicosPrivados",
        "TransportePublico": "ACC_educacion_tiempo_TransportePublico_Primaria_PublicosPrivados",
    },
    "edu_sec_public": {
        "coche": "ACC_educacion_tiempo_coche_Secundaria_Publicos",
        "TransportePublico": "ACC_educacion_tiempo_TransportePublico_Secundaria_Publicos",
    },
    "edu_sec_pubpriv": {
        "coche": "ACC_educacion_tiempo_coche_Secundaria_PublicosPrivados",
        "TransportePublico": "ACC_educacion_tiempo_TransportePublico_Secundaria_PublicosPrivados",
    },
}

# ---------------------------------------------------------------------
# AHP helper functions
# ---------------------------------------------------------------------
RI_TABLE: Dict[int, float] = {
    1: 0.00,
    2: 0.00,
    3: 0.52,
    4: 0.89,
    5: 1.11,
    6: 1.25,
    7: 1.35,
    8: 1.40,
    9: 1.45,
    10: 1.49,
    11: 1.52,
    12: 1.54,
    13: 1.56,
}


def preferences_to_matrix(answers, mode: str) -> np.ndarray:
    mode = str(mode).lower()

    if mode == "comparison":
        k = len(answers)
        n = int((1 + np.sqrt(1 + 8 * k)) / 2)
        matrix = np.ones((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                val = float(answers[idx])
                matrix[i, j] = val
                matrix[j, i] = 1.0 / val
                idx += 1
        return matrix

    if mode == "ranking":
        ranking = np.asarray(answers, dtype=float)
        n = len(ranking)
        matrix = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if ranking[i] == ranking[j]:
                    val = 1.0
                else:
                    d = int(max(ranking[i], ranking[j]) / min(ranking[i], ranking[j]))
                    d = min(d, 9)
                    val = d if ranking[i] < ranking[j] else 1.0 / d
                matrix[i, j] = val
                matrix[j, i] = 1.0 / val
        return matrix

    raise ValueError("mode must be either 'comparison' or 'ranking'")


def compute_cr(A: np.ndarray) -> float:
    vals, _ = np.linalg.eig(A)
    lam_max = max(vals.real)
    n = A.shape[0]
    CI = float((lam_max - n) / (n - 1)) if n > 1 else 0.0
    RI = RI_TABLE.get(n, 1.35)
    CR = float(CI / RI) if RI else 0.0
    return CR


def project_to_consistent(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    M = np.log(A)
    ones = np.ones((n, n))
    M1 = M @ ones
    P = (1.0 / n) * (M1 - M1.T)
    B = np.exp(P)
    B = (B + 1.0 / B.T) / 2.0
    np.fill_diagonal(B, 1.0)
    return B


def compute_weights(A: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(A)
    w = np.abs(vecs[:, np.argmax(vals.real)])
    return w / w.sum()


def preferences_to_weights(answers: np.ndarray, mode: str) -> np.ndarray:
    A = preferences_to_matrix(answers, mode)
    A = A if compute_cr(A) < 0.1 else project_to_consistent(A)
    return compute_weights(A)

# ---------------------------------------------------------------------
# Questionnaire mappings
# ---------------------------------------------------------------------
CAR_FREQ_LABELS = [
    "Almost never (0–1 days/week)",
    "Occasionally (2–3 days/week)",
    "Frequently (4–5 days/week)",
    "Almost always (6–7 days/week)",
]
CAR_FREQ_TO_WCAR: Dict[str, float] = {
    CAR_FREQ_LABELS[0]: 0.5 / 7.0,
    CAR_FREQ_LABELS[1]: 2.5 / 7.0,
    CAR_FREQ_LABELS[2]: 4.5 / 7.0,
    CAR_FREQ_LABELS[3]: 6.5 / 7.0,
}

SPORT_FREQ_LABELS = CAR_FREQ_LABELS
SPORT_FREQ_TO_W: Dict[str, float] = {
    SPORT_FREQ_LABELS[0]: 0.5 / 7.0,
    SPORT_FREQ_LABELS[1]: 2.5 / 7.0,
    SPORT_FREQ_LABELS[2]: 4.5 / 7.0,
    SPORT_FREQ_LABELS[3]: 6.5 / 7.0,
}

HOSPITAL_USE_LABELS = [
    "Only for emergencies",
    "Regular check-ups",
    "Accompanying people at risk",
    "Recurrent disease",
]
HOSPITAL_USE_TO_W: Dict[str, float] = {
    "Only for emergencies": 0.2,
    "Regular check-ups": 0.6,
    "Accompanying people at risk": 0.8,
    "Recurrent disease": 1.0,
}

EDU_LEVEL_OPTIONS = ["Preinfantil", "Infantil", "Primaria", "Secundaria"]


def edu_level_to_key(level: str, variant: Literal["public", "pubpriv"]) -> str:
    lv = level.lower()
    if lv == "preinfantil":
        return "edu_preinf_public" if variant == "public" else "edu_preinf_pubpriv"
    if lv == "infantil":
        return "edu_inf_public" if variant == "public" else "edu_inf_pubpriv"
    if lv == "primaria":
        return "edu_prim_public" if variant == "public" else "edu_prim_pubpriv"
    if lv == "secundaria":
        return "edu_sec_public" if variant == "public" else "edu_sec_pubpriv"
    raise ValueError(f"Unknown education level: {level}")

# ---------------------------------------------------------------------
# Data loading & images
# ---------------------------------------------------------------------
@st.cache_data
def load_data() -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "merged_dataset.csv")
    if not os.path.exists(csv_path):
        st.error(f"No se encuentra merged_dataset.csv en {csv_path}")
        st.stop()

    df = pd.read_csv(csv_path)

    shp_path = os.path.join(
        script_dir, "boundaries", "recintos_municipales_inspire_peninbal_etrs89.shp"
    )
    if not os.path.exists(shp_path):
        st.error(f"No se encuentra el archivo SHP en {shp_path}")
        st.stop()

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf = gpd.read_file(shp_path)

    madrid_gdf = gdf[gdf["CODNUT2"] == "ES30"].copy()
    if len(madrid_gdf) == 0:
        st.error("No se encontraron municipios de Madrid en los datos geográficos.")
        st.stop()

    madrid_gdf["NAMEUNIT"] = madrid_gdf["NAMEUNIT"].astype(str)
    df["Nombre"] = df["Nombre"].astype(str)

    merged_gdf = madrid_gdf.merge(df, left_on="NAMEUNIT", right_on="Nombre", how="inner")
    if len(merged_gdf) == 0:
        st.error("No se pudieron combinar los datos geográficos con los datos de merged_dataset.")
        st.write("Ejemplos en CSV:", df["Nombre"].head(10).tolist())
        st.write("Ejemplos en SHP:", madrid_gdf["NAMEUNIT"].head(10).tolist())
        st.stop()

    return df, merged_gdf


@st.cache_data
def load_placeholder_images() -> Dict[str, Optional[Image.Image]]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images: Dict[str, Optional[Image.Image]] = {}
    for i in range(1, 7):
        try:
            img_path = os.path.join(script_dir, "photos", f"placeholder{i}.jpeg")
            img = Image.open(img_path)
            images[f"placeholder{i}"] = img
        except Exception:
            images[f"placeholder{i}"] = None
    return images

# ---------------------------------------------------------------------
# Accessibility aggregation and ranking
# ---------------------------------------------------------------------
@st.cache_data
def compute_accessibility_hours(
    df: pd.DataFrame,
    w_car: float,
    w_sport: float,
    w_hospital: float,
    edu_has_kids: bool,
    edu_variant: Optional[Literal["public", "pubpriv"]],
    edu_levels: List[str],
    edu_acc_weight: float,
) -> pd.DataFrame:
    out = df[["codigo", "Nombre"]].copy()
    total = np.zeros(len(df), dtype=float)

    def blend_minutes(col_car: str, col_pt: str) -> np.ndarray:
        mc = df[col_car].astype(float)
        mp = df[col_pt].astype(float) if col_pt in df.columns else mc
        return w_car * mc + (1.0 - w_car) * mp

    def add_hours(key: str, minutes_one_way: np.ndarray, visits_per_month: float, weight: float = 1.0) -> None:
        nonlocal total
        hours = weight * visits_per_month * (2.0 * minutes_one_way) / 60.0
        out[f"hrs_{key}"] = hours
        total += hours

    # Supermarkets
    mins_super = blend_minutes(
        ACC_COLUMNS["supermarket"]["coche"],
        ACC_COLUMNS["supermarket"]["TransportePublico"],
    )
    add_hours("supermarket", mins_super, visits_per_month=8.0, weight=1.0)

    # Gas stations
    mins_gas = df[ACC_COLUMNS["gas"]["coche"]].astype(float)
    add_hours("gas", mins_gas, visits_per_month=2.0, weight=max(0.0, min(1.0, w_car)))

    # Sport
    mins_sport = blend_minutes(
        ACC_COLUMNS["sport"]["coche"],
        ACC_COLUMNS["sport"]["TransportePublico"],
    )
    add_hours("sport", mins_sport, visits_per_month=4.0, weight=max(0.0, min(1.0, w_sport)))

    # Health – GP and pharmacies
    mins_gp = blend_minutes(
        ACC_COLUMNS["gp"]["coche"],
        ACC_COLUMNS["gp"]["TransportePublico"],
    )
    add_hours("gp", mins_gp, visits_per_month=0.25, weight=max(0.0, min(1.0, w_hospital)))

    mins_pharm = blend_minutes(
        ACC_COLUMNS["pharmacy"]["coche"],
        ACC_COLUMNS["pharmacy"]["TransportePublico"],
    )
    add_hours("pharmacy", mins_pharm, visits_per_month=1.0, weight=max(0.0, min(1.0, w_hospital)))

    # Education
    if edu_has_kids and edu_variant in ("public", "pubpriv") and edu_levels and edu_acc_weight > 0.0:
        per_level = 1.0 / len(edu_levels)
        for level in edu_levels:
            svc_key = edu_level_to_key(level, edu_variant)
            cols = ACC_COLUMNS[svc_key]
            mins = blend_minutes(cols["coche"], cols["TransportePublico"])
            add_hours(
                f"edu_{level.lower()}",
                mins,
                visits_per_month=2.0,
                weight=per_level * edu_acc_weight,
            )

    out["AccessibilityHoursMonthly"] = total
    return out


@st.cache_data
def normalize_criteria(
    df: pd.DataFrame,
    benefit_cols: Dict[str, str],
    cost_cols: Dict[str, str],
    accessibility_col: str = "AccessibilityHoursMonthly",
) -> pd.DataFrame:
    out = df.copy()

    for crit, col in benefit_cols.items():
        x = out[col].astype(float)
        rng = x.max() - x.min()
        out[f"NORM_{crit}"] = (x - x.min()) / (rng if rng != 0 else 1.0)

    xp = out[cost_cols["HousePriceSqm"]].astype(float)
    prng = xp.max() - xp.min()
    out["NORM_HousePriceSqm"] = 1.0 - (xp - xp.min()) / (prng if prng != 0 else 1.0)

    xa = out[accessibility_col].astype(float)
    arng = xa.max() - xa.min()
    out["NORM_AccessibilityHoursMonthly"] = 1.0 - (xa - xa.min()) / (arng if arng != 0 else 1.0)

    return out


def compute_scores(df_norm: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    out = df_norm.copy()
    score = np.zeros(len(out), dtype=float)

    for crit in CRITERIA:
        w = float(weights.get(crit, 0.0))
        col = f"NORM_{crit}"
        contrib = w * out[col].astype(float)
        out[f"CONTRIB_{crit}"] = contrib
        score += contrib.values

    out["Score"] = score
    max_score = out["Score"].max()
    out["weighted_score"] = (out["Score"] / max_score * 100.0) if max_score > 0 else 0.0
    return out.sort_values("Score", ascending=False).reset_index(drop=True)


def equal_weights() -> Dict[str, float]:
    w = 1.0 / len(CRITERIA)
    return {c: w for c in CRITERIA}

# ---------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------
def create_heatmap(gdf: gpd.GeoDataFrame):
    gdf_plot = gdf.to_crs(epsg=4326)

    fig = px.choropleth_mapbox(
        gdf_plot,
        geojson=gdf_plot.geometry.__geo_interface__,
        locations=gdf_plot.index,
        color="weighted_score",
        color_continuous_scale=[
            "#DFD1B6",  # Sand - low
            "#6FB5BA",  # Teal - medium
            "#568EE2",  # Primary - high
            "#3D517B",  # Slate - very high
        ],
        range_color=[gdf_plot["weighted_score"].min(), gdf_plot["weighted_score"].max()],
        mapbox_style="open-street-map",
        zoom=8,
        center={"lat": 40.4168, "lon": -3.7038},
        opacity=0.7,
        title="Mapa de municipios según tu perfil",
        custom_data=[gdf_plot["Nombre"]],
        labels={"weighted_score": "Puntuación (más alto = mejor)"},
    )

    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>Puntuación: %{z:.1f}<extra></extra>"
    )
    fig.update_layout(
        height=600,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        clickmode="event+select",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ---------------------------------------------------------------------
# Municipality details + cards
# ---------------------------------------------------------------------
def show_single_municipality_details(
    muni: pd.Series,
    images: Dict[str, Optional[Image.Image]],
    title: Optional[str] = None,
) -> None:
    if title:
        st.markdown(f"**{title}**")

    col1, col2 = st.columns([1, 2])

    with col1:
        random.seed(hash(muni["Nombre"]))
        img_key = f"placeholder{random.randint(1, 6)}"
        if images.get(img_key) is not None:
            st.image(images[img_key], caption=muni["Nombre"], width=150)

    with col2:
        st.markdown(f"**{muni['Nombre']}**")
        st.markdown(
            f'<div class="score-badge">Puntuación: {muni["weighted_score"]:.1f}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"👥 **Población:** {int(muni['IDE_PoblacionTotal']):,}")
        st.markdown(
            f"💰 **Precio vivienda:** {muni['IDE_PrecioPorMetroCuadrado']:.0f} €/m²"
        )
        st.markdown(
            f"⌛ **Horas al mes en transporte:** {muni['AccessibilityHoursMonthly']:.1f}"
        )

    st.markdown("**Desglose por criterio**")

    for crit in CRITERIA:
        norm_col = f"NORM_{crit}"
        contrib_col = f"CONTRIB_{crit}"
        if norm_col not in muni or contrib_col not in muni:
            continue

        icon = CRITERIA_ICONS[crit]
        label = CRITERIA_LABELS[crit]
        value = float(muni[norm_col])
        contrib = float(muni[contrib_col])

        col_icon, col_label, col_bar = st.columns([0.3, 1.8, 2.4])
        with col_icon:
            st.markdown(f'<span class="concept-icon">{icon}</span>', unsafe_allow_html=True)
        with col_label:
            st.markdown(f"**{label}**")
        with col_bar:
            pct = int(value * 100)
            if pct >= 70:
                color = "#377F86"  # Success
            elif pct >= 40:
                color = "#C35309"  # Warning
            else:
                color = "#C33241"  # Error

            progress_html = f"""
            <div style="background-color: #e9ecef; border-radius: 10px; height: 20px; width: 100%;">
                <div style="background-color: {color}; height: 20px; width: {pct}%; border-radius: 10px;
                           display: flex; align-items: center; justify-content: center; color: white; font-size: 12px;">
                    {pct}%
                </div>
            </div>
            <div style="font-size: 11px; color: #555;">Contribución ponderada: {contrib:.3f}</div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)


def render_municipality_card(muni: pd.Series, images: Dict[str, Optional[Image.Image]]) -> None:
    st.markdown('<div class="municipality-card">', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        random.seed(hash(muni["Nombre"]))
        img_key = f"placeholder{random.randint(1, 6)}"
        if images.get(img_key) is not None:
            st.image(images[img_key], caption=muni["Nombre"], use_container_width=True)

    with col2:
        st.markdown(
            f"<div class='municipality-name'>{muni['Nombre']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="score-badge">Puntuación: {muni["weighted_score"]:.1f}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            (
                f"👥 **Población:** {int(muni['IDE_PoblacionTotal']):,}<br>"
                f"💰 **Precio vivienda:** {muni['IDE_PrecioPorMetroCuadrado']:.0f} €/m²<br>"
                f"⌛ **Horas al mes en transporte:** {muni['AccessibilityHoursMonthly']:.1f}"
            ),
            unsafe_allow_html=True,
        )

        if st.button("Ver detalles", key=f"details_btn_{muni['codigo']}"):
            st.session_state["selected_municipality"] = muni
            st.session_state["details_origin"] = "list"                    # <- came from list
            st.session_state["suppress_map_selection"] = True              # ignore old map click
            st.session_state["switch_view_to"] = "📋 Lista de municipios"
            st.rerun()


    st.markdown("</div>", unsafe_allow_html=True)


def show_municipality_details(
    municipality: pd.Series,
    images: Dict[str, Optional[Image.Image]],
    all_scores: pd.DataFrame,
) -> None:
    with st.container():
        header_col1, header_col2 = st.columns([4, 1])
        with header_col1:
            st.markdown("## 📍 Detalles del municipio")
            st.markdown(f"**{municipality['Nombre']}**")
        with header_col2:
            close_key = f"close_details_{municipality['codigo']}"
        if st.button("❌ Cerrar", key=close_key):
            origin = st.session_state.get("details_origin")

            # Request a view switch; will be applied at top of main()
            if origin == "map":
                st.session_state["switch_view_to"] = "🗺️ Mapa de municipios"
            elif origin == "list":
                st.session_state["switch_view_to"] = "📋 Lista de municipios"

            st.session_state["suppress_map_selection"] = True

            for key in [
                "selected_municipality",
                "comparison_municipality",
                "comparison_selector",
                "comparison_selector_in_panel",
                "details_origin",
            ]:
                st.session_state.pop(key, None)

            st.rerun()


            # Clear selection + comparison info
            for key in [
                "selected_municipality",
                "comparison_municipality",
                "comparison_selector",
                "comparison_selector_in_panel",
                "details_origin",
            ]:
                st.session_state.pop(key, None)

            st.rerun()


        comparison_mode = "comparison_municipality" in st.session_state

        if comparison_mode:
            comparison_muni = st.session_state.comparison_municipality

            col1, col2, col3 = st.columns([5, 1, 5])
            with col1:
                show_single_municipality_details(
                    municipality, images, "🏘️ Municipio principal"
                )
            with col2:
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                st.markdown("**VS**", unsafe_allow_html=True)
            with col3:
                comp_header_col1, comp_header_col2 = st.columns([3, 1])
                with comp_header_col1:
                    st.markdown("**🔍 Comparación**")
                with comp_header_col2:
                    end_comparison_key = f"end_comparison_{municipality['codigo']}"
                    if st.button("🔄", key=end_comparison_key, help="Terminar comparación"):
                        st.session_state["clear_comparison_only"] = True
                        st.rerun()

                show_single_municipality_details(comparison_muni, images, None)

                st.markdown("---")
                st.markdown("**Cambiar municipio:**")

                options = [
                    f"{row['Nombre']} (Puntuación: {row['weighted_score']:.1f})"
                    for _, row in all_scores.iterrows()
                    if row["codigo"] != municipality["codigo"]
                ]

                if options:
                    current_selection = (
                        f"{comparison_muni['Nombre']} "
                        f"(Puntuación: {comparison_muni['weighted_score']:.1f})"
                    )
                    try:
                        current_index = options.index(current_selection) + 1
                    except ValueError:
                        current_index = 0

                    selected = st.selectbox(
                        "Selecciona otro municipio:",
                        ["Selecciona un municipio..."] + options,
                        index=current_index,
                        key="comparison_selector_in_panel",
                    )

                    if selected != "Selecciona un municipio..." and selected != current_selection:
                        comp_name = selected.split(" (Puntuación:")[0]
                        comp_row = all_scores[all_scores["Nombre"] == comp_name].iloc[0]
                        st.session_state["comparison_municipality"] = comp_row
                        st.rerun()
        else:
            show_single_municipality_details(municipality, images)

            st.markdown("---")
            st.subheader("🔍 Comparar con otro municipio")
            options = [
                f"{row['Nombre']} (Puntuación: {row['weighted_score']:.1f})"
                for _, row in all_scores.iterrows()
                if row["codigo"] != municipality["codigo"]
            ]
            if options:
                selected = st.selectbox(
                    "Selecciona municipio para comparar:",
                    ["Selecciona un municipio..."] + options,
                    key="comparison_selector",
                )
                if selected != "Selecciona un municipio...":
                    comp_name = selected.split(" (Puntuación:")[0]
                    comp_row = all_scores[all_scores["Nombre"] == comp_name].iloc[0]
                    st.session_state["comparison_municipality"] = comp_row
                    st.rerun()

# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------
def main() -> None:
    # --- Handle programmatic view switching BEFORE any widgets are created ---
    if "view_selector" not in st.session_state:
        st.session_state["view_selector"] = "🗺️ Mapa de municipios"

    if (
        "switch_view_to" in st.session_state
        and st.session_state["switch_view_to"] is not None
    ):
        # apply requested switch, then clear the flag
        st.session_state["view_selector"] = st.session_state["switch_view_to"]
        st.session_state["switch_view_to"] = None    
    if st.session_state.get("clear_comparison_only", False):
        for key in [
            "comparison_municipality",
            "comparison_selector_in_panel",
            "clear_comparison_only",
        ]:
            st.session_state.pop(key, None)

    # Header
    st.markdown(
        '<h1 class="main-header">🏘️ Living on the Edge</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="tagline">Encuentra tu municipio ideal en la Comunidad de Madrid según tu estilo de vida, tu familia y tus prioridades.</p>',
        unsafe_allow_html=True,
    )

    # Load data
    df_raw, gdf_raw = load_data()
    images = load_placeholder_images()

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("Tu perfil y prioridades")

        # Movilidad
        st.subheader("Movilidad – coche")
        car_use = st.selectbox(
            "¿Con qué frecuencia usarías el coche?",
            options=CAR_FREQ_LABELS,
            index=2,
        )
        w_car = CAR_FREQ_TO_WCAR[car_use]

        # Familia – educación
        st.subheader("Familia – educación")
        has_kids_ans = st.radio(
            "¿Tienes hij@s pequeñ@s?",
            options=["No", "Sí"],
            horizontal=True,
        )
        edu_has_kids = has_kids_ans == "Sí"

        edu_variant: Optional[Literal["public", "pubpriv"]] = None
        edu_levels: List[str] = []
        edu_trade: float = 0.0

        if edu_has_kids:
            school_type = st.radio(
                "¿Van a colegio público o privado?",
                options=["Solo público", "Privado o mixto"],
                horizontal=True,
            )
            edu_variant = "public" if school_type == "Solo público" else "pubpriv"

            edu_levels = st.multiselect(
                "¿En qué etapas tienes hij@s?",
                options=EDU_LEVEL_OPTIONS,
                help="Si eliges varias, las ponderamos igual.",
            )

            edu_trade = st.slider(
                "¿Qué te importa más en educación?",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="0 = solo calidad educativa (ATR), 1 = solo accesibilidad (ACC).",
            )

        # Estilo de vida – deporte
        st.subheader("Estilo de vida – deporte")
        sport_use = st.selectbox(
            "¿Con qué frecuencia harías deporte?",
            options=SPORT_FREQ_LABELS,
            index=1,
        )
        w_sport = SPORT_FREQ_TO_W[sport_use]

        # Salud
        st.subheader("Salud – hospitales y farmacias")
        hosp_use = st.selectbox(
            "¿Qué uso haces de hospitales/centros de salud?",
            options=HOSPITAL_USE_LABELS,
            index=1,
        )
        w_hospital = HOSPITAL_USE_TO_W[hosp_use]

        # Población
        st.subheader("Tamaño del municipio")
        if "IDE_PoblacionTotal" in df_raw.columns:
            min_pop_data = int(df_raw["IDE_PoblacionTotal"].min())
            max_pop_data = int(df_raw["IDE_PoblacionTotal"].max())
        else:
            min_pop_data, max_pop_data = 0, 50000

        pop_min, pop_max = st.slider(
            "Rango de población del municipio",
            min_value=min_pop_data,
            max_value=max_pop_data,
            value=(min_pop_data, min(50000, max_pop_data)),
            step=1000,
        )

        # Ranking de características
        st.subheader("Prioriza estas características (1 = más importante)")
        ranks: List[float] = []
        for crit in CRITERIA:
            label = f"{CRITERIA_ICONS[crit]} {CRITERIA_LABELS[crit]}"
            rank = st.number_input(
                label,
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                key=f"rank_{crit}",
            )
            ranks.append(float(rank))

        st.button("Aplicar preferencias")  # purely visual

    # ---------------- Back-end logic ----------------
    df = df_raw.copy()
    if "IDE_PoblacionTotal" in df.columns:
        df = df[
            (df["IDE_PoblacionTotal"] >= pop_min)
            & (df["IDE_PoblacionTotal"] <= pop_max)
        ].copy()

    acc_df = compute_accessibility_hours(
        df=df,
        w_car=float(w_car),
        w_sport=float(w_sport),
        w_hospital=float(w_hospital),
        edu_has_kids=edu_has_kids,
        edu_variant=edu_variant,
        edu_levels=edu_levels,
        edu_acc_weight=float(edu_trade),
    )

    df_scored = df.merge(
        acc_df[["codigo", "AccessibilityHoursMonthly"]],
        on="codigo",
        how="left",
    )

    norm_df = normalize_criteria(
        df_scored,
        benefit_cols=BENEFIT_COLUMNS,
        cost_cols=COST_COLUMNS,
    )

    try:
        w_vec = preferences_to_weights(np.array(ranks, dtype=float), mode="ranking")
        weights: Dict[str, float] = {
            CRITERIA[i]: float(w_vec[i]) for i in range(len(CRITERIA))
        }
    except Exception as e:
        st.sidebar.warning(f"No se pudieron calcular los pesos AHP ({e}). Usamos pesos iguales.")
        weights = equal_weights()

    scores_df = compute_scores(norm_df, weights)

    gdf = gdf_raw.merge(
        scores_df[
            [
                "codigo",
                "Nombre",
                "Score",
                "weighted_score",
                "AccessibilityHoursMonthly",
                "IDE_PoblacionTotal",
                "IDE_PrecioPorMetroCuadrado",
            ]
            + [c for c in scores_df.columns if c.startswith("NORM_") or c.startswith("CONTRIB_")]
        ],
        on=["Nombre"],
        how="inner",
    )

    # ---------------- Main view selector ----------------
    view_option = st.radio(
        "Selecciona vista:",
        ["🗺️ Mapa de municipios", "📋 Lista de municipios"],
        horizontal=True,
        key="view_selector",
    )


    if view_option == "🗺️ Mapa de municipios":
        if len(gdf) > 0:
            st.markdown(
                "**Consejo:** haz clic en un municipio del mapa para ver más detalles abajo 👇"
            )
            suppress = st.session_state.pop("suppress_map_selection", False)

            fig = create_heatmap(gdf)
            event = st.plotly_chart(
                fig,
                key="heatmap",
                width="stretch",
                on_select="rerun",
                selection_mode="points",
            )

            if (
                not suppress
                and event
                and event.selection
                and event.selection["point_indices"]
            ):
                idx = event.selection["point_indices"][0]
                clicked_name = gdf.iloc[idx]["Nombre"]
                selected_row = scores_df[scores_df["Nombre"] == clicked_name].iloc[0]

                st.session_state["selected_municipality"] = selected_row
                st.session_state["details_origin"] = "map"
                st.session_state["switch_view_to"] = "🗺️ Mapa de municipios"
  # <- stay in map
        else:
            st.warning("No hay municipios disponibles para mostrar.")
            

    elif view_option == "📋 Lista de municipios":
        if len(scores_df) == 0:
            st.info("No hay municipios disponibles para mostrar.")
        else:
            st.markdown("### 📋 Municipios ordenados por puntuación")

            page_size = 10
            total = len(scores_df)
            num_pages = max(1, math.ceil(total / page_size))

            page = st.number_input(
                "Página",
                min_value=1,
                max_value=num_pages,
                value=st.session_state.get("list_page", 1),
                step=1,
                key="list_page",
            )

            start = (page - 1) * page_size
            end = start + page_size
            page_df = scores_df.iloc[start:end]

            for _, row in page_df.iterrows():
                render_municipality_card(row, images)

    # ---------------- Municipality details (for both views) ----------------
    if "selected_municipality" in st.session_state:
        st.markdown("---")
        show_municipality_details(
            st.session_state["selected_municipality"], images, scores_df
        )


if __name__ == "__main__":
    main()
