# app.py
# LodCORE — Madrid (<50k) municipalities ranking by Accessibility & QoL
# Implements questionnaire, accessibility aggregation, AHP weights using the provided "task 2" functions,
# and ranking with sensitivity analysis.
#
# Python 3.10+, Streamlit 1.x

from __future__ import annotations
from typing import Dict, List, Tuple, Literal, Optional

import json
import math
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="LodCORE — Accessibility & QoL", page_icon=":house:", layout="wide")
DATA_DEFAULT_PATHS = ["data/merged_dataset.csv", "merged_dataset.csv"]

# ---------------------------------------------------------------------
# Dataset columns & criteria
# ---------------------------------------------------------------------
CRITERIA: List[str] = [
    "AccessibilityHoursMonthly",          # cost (lower better)
    "EducationQuality",                   # benefit
    "AirQuality",                         # benefit
    "BuildingQuality",                    # benefit
    "TransportInfraQuality",              # benefit
    "EconomicDynamism",                   # benefit
    "HousePriceSqm",                      # cost (lower better)
]

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

# Accessibility columns — one-way minutes. We will blend coche/PT per answers.
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
        "TransportePublico": "ACC_gasolineras_tiempo_coche",  # fallback
    },
    "supermarket": {
        "coche": "OSM_supermercados_tiempo_coche",
        "TransportePublico": "OSM_supermercados_tiempo_TransportePublico",
    },
    # Education — public vs public+private
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
# Provided "Task 2" AHP helper functions (use as-is)
# ---------------------------------------------------------------------
import numpy as np  # re-import to match provided snippet's namespace

def preferences_to_matrix(answers, mode):
    """
    Convert a list of user preferences into a full reciprocal Saaty matrix.

    Args:
        answers: list or array of floats
            - If mode == 'comparison': ordered rowwise upper-triangular (excluding diag).
            - If mode == 'ranking': ranking indices or scores.
        mode: str ('comparison' or 'ranking')

    Returns:
        np.ndarray: complete reciprocal matrix (n x n)
    """
    mode = str(mode).lower()

    if mode == "comparison":
        # Determine matrix size from number of off-diagonal upper entries
        # k = n*(n-1)/2 -> n = (1 + sqrt(1 + 8k))/2
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

    elif mode == "ranking":
        ranking = np.asarray(answers, dtype=float)
        n = len(ranking)
        matrix = np.ones((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                if ranking[i] == ranking[j]:
                    val = 1.0
                else:
                    # ratio between rank values, capped at 9
                    d = int(max(ranking[i], ranking[j]) / min(ranking[i], ranking[j]))
                    d = min(d, 9)
                    # smaller rank = higher importance
                    val = d if ranking[i] < ranking[j] else 1.0 / d
                matrix[i, j] = val
                matrix[j, i] = 1.0 / val
        return matrix

    else:
        raise ValueError("mode must be either 'comparison' or 'ranking'")

RI_TABLE = {1:0.00, 2:0.00, 3:0.52, 4:0.89, 5:1.11, 6:1.25, 7:1.35, 8:1.40, 9:1.45, 10:1.49, 11:1.52, 12:1.54, 13:1.56}

def compute_cr(A: np.ndarray):
    """
    Computes the Consistency Ratio of a matrix (should be a reciprocal matrix).
    """
    vals, _ = np.linalg.eig(A)
    lam_max = max(vals.real)
    n = A.shape[0]
    CI = float((lam_max - n) / (n - 1)) if n > 1 else 0.0
    RI = RI_TABLE.get(n, 1.35)
    CR = float(CI / RI) if RI else 0.0
    return CR

def project_to_consistent(A: np.ndarray) -> np.ndarray:
    """
    Projection procedure from Gaceta R. Soc. Mat. Esp.:
        M = log(A)
        P = (1/n) * (M·1ₙ - (M·1ₙ)ᵀ)
        B = exp(P)
    Then enforce exact reciprocity and 1s on the diagonal.
    """
    n = A.shape[0]
    M = np.log(A)
    ones = np.ones((n, n))
    M1 = M @ ones
    P = (1.0 / n) * (M1 - M1.T)
    B = np.exp(P)
    # numerical stabilization
    B = (B + 1.0 / B.T) / 2.0
    np.fill_diagonal(B, 1.0)
    return B

def compute_weights(A: np.ndarray) -> np.ndarray:
    """Principal eigenvector normalized to sum=1."""
    vals, vecs = np.linalg.eig(A)
    w = np.abs(vecs[:, np.argmax(vals.real)])
    return w / w.sum()

def preferences_to_weights(answers: np.ndarray, mode: str) -> np.ndarray:
    """Performs the whole AHP procedure. Receives the preferences of the user and returns the weights vector."""
    A = preferences_to_matrix(answers, mode)
    A = A if compute_cr(A) < 0.1 else project_to_consistent(A)
    return compute_weights(A)

# ---------------------------------------------------------------------
# Utilities (schema, loading, etc.)
# ---------------------------------------------------------------------
@st.cache_data
def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load merged_dataset.csv.

    Args:
        path: Optional explicit path or None to try defaults.

    Returns:
        DataFrame with dataset.
    """
    candidates = [path] if path else DATA_DEFAULT_PATHS
    last_err = None
    for p in candidates:
        try:
            return pd.read_csv(p)
        except Exception as e:
            last_err = e
    raise FileNotFoundError(f"Could not load dataset. Tried {candidates}. Last error: {last_err}")

def compact_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact schema summary for display."""
    rows: List[Dict[str, object]] = []
    for col in df.columns:
        s = df[col]
        info = {"column": col, "dtype": str(s.dtype), "non_null": int(s.notna().sum()), "nulls": int(s.isna().sum())}
        if pd.api.types.is_numeric_dtype(s):
            info["min"] = float(np.nanmin(s.values))
            info["max"] = float(np.nanmax(s.values))
            info["mean"] = float(np.nanmean(s.values))
        else:
            info["examples"] = ", ".join(map(str, s.dropna().astype(str).head(3).tolist()))
        rows.append(info)
    out = pd.DataFrame(rows)
    for k in ["min","max","mean","examples"]:
        if k not in out.columns:
            out[k] = np.nan
    return out[["column","dtype","non_null","nulls","min","max","mean","examples"]]

def save_schema_json() -> None:
    """Persist schema orientation and mappings to schema.json for traceability."""
    schema = {
        "keys": {"codigo": "int", "Nombre": "str"},
        "benefit_columns": BENEFIT_COLUMNS,
        "cost_columns": COST_COLUMNS,
        "accessibility_minutes_one_way": ACC_COLUMNS,
        "assumptions": [
            "*ClusterEstadistica are ordinal benefit scores; higher is better.",
            "*ClusterPoblacion ignored for scoring.",
            "Times are minutes (one-way); we compute round-trip hours.",
            "Population filter <50k is always applied.",
        ],
    }
    with open("schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------
# Questionnaire mappings
# ---------------------------------------------------------------------
CAR_FREQ_LABELS = [
    "Almost never (0–1 days/week)",
    "Occasionally (2–3 days/week)",
    "Frequently (4–5 days/week)",
    "Almost always (6–7 days/week)",
]
CAR_FREQ_TO_WCAR = {
    CAR_FREQ_LABELS[0]: 0.5/7.0,
    CAR_FREQ_LABELS[1]: 2.5/7.0,
    CAR_FREQ_LABELS[2]: 4.5/7.0,
    CAR_FREQ_LABELS[3]: 6.5/7.0,
}

SPORT_FREQ_LABELS = CAR_FREQ_LABELS
SPORT_FREQ_TO_WEIGHT = {
    SPORT_FREQ_LABELS[0]: 0.5/7.0,
    SPORT_FREQ_LABELS[1]: 2.5/7.0,
    SPORT_FREQ_LABELS[2]: 4.5/7.0,
    SPORT_FREQ_LABELS[3]: 6.5/7.0,
}

HOSPITAL_USE_LABELS = [
    "Only for emergencies",
    "Regular check-ups",
    "Accompanying people at risk",
    "Recurrent disease",
]
HOSPITAL_USE_TO_WEIGHT = {
    "Only for emergencies": 0.2,
    "Regular check-ups": 0.6,
    "Accompanying people at risk": 0.8,
    "Recurrent disease": 1.0,
}

EDU_LEVEL_OPTIONS = ["Preinfantil", "Infantil", "Primaria", "Secundaria"]

def edu_level_to_key(level: str, variant: Literal["public","pubpriv"]) -> str:
    """Map education level + variant to ACC service key."""
    lv = level.lower()
    if lv == "preinfantil":
        return "edu_preinf_public" if variant == "public" else "edu_preinf_pubpriv"
    if lv == "infantil":
        return "edu_inf_public" if variant == "public" else "edu_inf_pubpriv"
    if lv == "primaria":
        return "edu_prim_public" if variant == "public" else "edu_prim_pubpriv"
    if lv == "secundaria":
        return "edu_sec_public" if variant == "public" else "edu_sec_pubpriv"
    raise ValueError(f"Unknown level: {level}")

# ---------------------------------------------------------------------
# Accessibility aggregation (pure → cached)
# ---------------------------------------------------------------------
@st.cache_data
def compute_accessibility_hours(
    df: pd.DataFrame,
    w_car: float,
    w_sport: float,
    w_hospital: float,
    edu_has_kids: bool,
    edu_variant: Optional[Literal["public","pubpriv"]],
    edu_levels: List[str],
    edu_acc_weight: float,
) -> pd.DataFrame:
    """Compute monthly round-trip hours per municipality from questionnaire answers.

    Args:
        df: Input dataset (already filtered <50k).
        w_car: Car-usage weight in [0,1] used to blend coche/PT minutes.
        w_sport: Sports frequency weight in [0,1].
        w_hospital: Hospital/pharmacy usage weight in [0,1].
        edu_has_kids: Whether to include education travel.
        edu_variant: 'public' or 'pubpriv' for education columns.
        edu_levels: Education stages to include equally if any.
        edu_acc_weight: Trade-off (0→ignore ACC for education; 1→full weight).

    Returns:
        DataFrame with per-service hours and AccessibilityHoursMonthly.
    """
    out = df[["codigo","Nombre"]].copy()
    total = np.zeros(len(df), dtype=float)

    def blend_minutes(col_coche: str, col_pt: str) -> np.ndarray:
        mc = df[col_coche].astype(float) if col_coche in df.columns else np.nan
        mp = df[col_pt].astype(float) if col_pt in df.columns else mc
        return w_car * mc + (1.0 - w_car) * mp

    def add_hours(key: str, minutes_one_way: np.ndarray, visits_per_month: float, weight: float = 1.0) -> None:
        hrs = weight * visits_per_month * (2.0 * minutes_one_way) / 60.0
        out[f"hrs_{key}"] = hrs
        nonlocal total
        total += hrs

    # Supermarket — essential (8/month)
    mins_super = blend_minutes(ACC_COLUMNS["supermarket"]["coche"], ACC_COLUMNS["supermarket"]["TransportePublico"])
    add_hours("supermarket", mins_super, 8.0, 1.0)

    # Gas — tied to car usage (2/month)
    mins_gas = df[ACC_COLUMNS["gas"]["coche"]].astype(float)
    add_hours("gas", mins_gas, 2.0, max(0.0, min(1.0, w_car)))

    # Sport — scaled by sports frequency (4/month)
    mins_sport = blend_minutes(ACC_COLUMNS["sport"]["coche"], ACC_COLUMNS["sport"]["TransportePublico"])
    add_hours("sport", mins_sport, 4.0, max(0.0, min(1.0, w_sport)))

    # Health — GP (0.25/month) + Pharmacy (1/month), scaled by hospital use
    mins_gp = blend_minutes(ACC_COLUMNS["gp"]["coche"], ACC_COLUMNS["gp"]["TransportePublico"])
    add_hours("gp", mins_gp, 0.25, max(0.0, min(1.0, w_hospital)))

    mins_pharm = blend_minutes(ACC_COLUMNS["pharmacy"]["coche"], ACC_COLUMNS["pharmacy"]["TransportePublico"])
    add_hours("pharmacy", mins_pharm, 1.0, max(0.0, min(1.0, w_hospital)))

    # Education — include selected levels equally; scaled by edu_acc_weight
    if edu_has_kids and edu_variant in ("public", "pubpriv") and len(edu_levels) > 0 and edu_acc_weight > 0.0:
        per_level = 1.0 / len(edu_levels)
        for level in edu_levels:
            svc_key = edu_level_to_key(level, "public" if edu_variant == "public" else "pubpriv")
            mins = blend_minutes(ACC_COLUMNS[svc_key]["coche"], ACC_COLUMNS[svc_key]["TransportePublico"])
            # assume 2 visits/month (meetings, events)
            add_hours(f"edu_{level.lower()}", mins, 2.0, per_level * edu_acc_weight)

    out["AccessibilityHoursMonthly"] = total
    return out

# ---------------------------------------------------------------------
# Normalization & ranking (pure → cached)
# ---------------------------------------------------------------------
@st.cache_data
def normalize_criteria(
    df: pd.DataFrame,
    benefit_cols: Dict[str, str],
    cost_cols: Dict[str, str],
    accessibility_hours_col: str = "AccessibilityHoursMonthly",
) -> pd.DataFrame:
    """Normalize all criteria to [0,1] with higher=better; invert costs."""
    out = df.copy()

    # benefits
    for crit, col in benefit_cols.items():
        x = out[col].astype(float)
        rng = x.max() - x.min()
        out[f"NORM_{crit}"] = (x - x.min()) / (rng if rng != 0 else 1.0)

    # costs
    price = out[cost_cols["HousePriceSqm"]].astype(float)
    prng = price.max() - price.min()
    out["NORM_HousePriceSqm"] = 1.0 - (price - price.min()) / (prng if prng != 0 else 1.0)

    acc = out[accessibility_hours_col].astype(float)
    arng = acc.max() - acc.min()
    out["NORM_AccessibilityHoursMonthly"] = 1.0 - (acc - acc.min()) / (arng if arng != 0 else 1.0)
    return out

@st.cache_data
def rank_municipalities(df_norm: pd.DataFrame, weights: Dict[str, float], top_n: int = 15) -> pd.DataFrame:
    """Weighted-sum score; return Top-N with contributions."""
    out = df_norm.copy()
    score = np.zeros(len(out), dtype=float)
    for crit in CRITERIA:
        w = float(weights.get(crit, 0.0))
        col = f"NORM_{crit}"
        contrib = w * out[col].astype(float)
        out[f"CONTRIB_{crit}"] = contrib
        score += contrib.values
    out["Score"] = score
    out = out.sort_values("Score", ascending=False).reset_index(drop=True)
    keep = ["codigo","Nombre","Score"] + [c for c in out.columns if c.startswith("CONTRIB_")]
    return out[keep].head(top_n)

def equal_weights() -> Dict[str, float]:
    """Equal weights over CRITERIA."""
    w = 1.0 / len(CRITERIA)
    return {c: w for c in CRITERIA}

def perturb_weights(weights: Dict[str, float], targets: List[str], delta: float) -> Dict[str, float]:
    """Apply ±delta relative change to selected criteria, renormalize to sum=1."""
    w = weights.copy()
    for t in targets:
        if t in w:
            w[t] = max(0.0, w[t] * (1.0 + delta))
    s = sum(w.values()) or 1.0
    return {k: v/s for k, v in w.items()}

# ---------------------------------------------------------------------
# Load data and apply mandatory population filter (<50k)
# ---------------------------------------------------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload merged_dataset.csv (optional)", type=["csv"])
df = load_data(uploaded if uploaded is not None else None)

if "IDE_PoblacionTotal" in df.columns:
    df = df[df["IDE_PoblacionTotal"] < 50000].copy()

save_schema_json()

with st.expander("Compact schema summary", expanded=False):
    st.dataframe(compact_schema(df), use_container_width=True)

# ---------------------------------------------------------------------
# TASK 1 — Questionnaire (as specified)
# ---------------------------------------------------------------------
st.header("User questionnaire")

with st.form("user_form"):
    st.subheader("Mobility — Car usage")
    car_use = st.selectbox("How often do you intend to use a car?", options=CAR_FREQ_LABELS, index=2)
    w_car = float(CAR_FREQ_TO_WCAR[car_use])
    st.caption("Gas stations weight is tied to car usage.")

    st.subheader("Family — Education needs")
    has_kids = st.radio("Do you have young kids?", options=["No", "Yes"], horizontal=True, index=0)
    edu_variant: Optional[Literal["public","pubpriv"]] = None
    edu_levels: List[str] = []
    edu_trade = 0.0
    if has_kids == "Yes":
        sch = st.radio("Do your kids go to public or private schools?", ["Public", "Private or Public+Private"], horizontal=True)
        edu_variant = "public" if sch == "Public" else "pubpriv"
        edu_levels = st.multiselect("Do you have kids in these education levels? (equal weight if multiple)", EDU_LEVEL_OPTIONS, default=[])
        edu_trade = st.slider(
            "How much do you value education quality over school accessibility?",
            0.0, 1.0, 0.5, 0.05,
            help="0 → 100% QUALITY (ATR) vs 1 → 100% ACCESSIBILITY (ACC) for education."
        )

    st.subheader("Lifestyle — Sports")
    sport_use = st.selectbox("How often do you intend to do sports?", options=SPORT_FREQ_LABELS, index=1)
    w_sport = float(SPORT_FREQ_TO_WEIGHT[sport_use])

    st.subheader("Health — Hospital usage")
    hosp_use = st.selectbox("What use do you give to hospitals?", options=HOSPITAL_USE_LABELS, index=1)
    w_hospital = float(HOSPITAL_USE_TO_WEIGHT[hosp_use])

    submitted = st.form_submit_button("Apply preferences")

# ---------------------------------------------------------------------
# TASK 2 — Accessibility aggregation
# ---------------------------------------------------------------------
st.header("Accessibility aggregation")
acc_df = compute_accessibility_hours(
    df=df,
    w_car=w_car,
    w_sport=w_sport,
    w_hospital=w_hospital,
    edu_has_kids=(has_kids == "Yes"),
    edu_variant=edu_variant,
    edu_levels=edu_levels,
    edu_acc_weight=float(edu_trade),
)
st.dataframe(acc_df.head(20), use_container_width=True)

work = df.merge(acc_df[["codigo","Nombre","AccessibilityHoursMonthly"]], on=["codigo","Nombre"], how="left")

# ---------------------------------------------------------------------
# TASK 3 — Weights via PROVIDED FUNCTIONS (comparison or ranking)
# ---------------------------------------------------------------------
st.header("Criteria weighting (AHP via provided functions)")

st.markdown("Choose **Comparison** to input pairwise upper-triangular values (Saaty 1–9), "
            "or **Ranking** to provide an ordered importance vector (1=most important).")

mode = st.radio("Weighting input mode", options=["comparison", "ranking"], horizontal=True, index=0)

answers: List[float] = []
if mode == "comparison":
    n = len(CRITERIA)
    st.caption("Enter upper-triangular (row-wise) comparisons only. Diagonal=1 and reciprocals are auto.")
    with st.form("pairwise_upper"):
        vals: List[float] = []
        for i in range(n):
            cols = st.columns(n)
            for j in range(i + 1, n):
                v = cols[j].number_input(f"{CRITERIA[i]} vs {CRITERIA[j]} (1–9)", min_value=1.0, max_value=9.0, value=1.0, step=1.0, key=f"a_{i}_{j}")
                vals.append(float(v))
        ok = st.form_submit_button("Compute weights")
        answers = vals if ok else vals
else:
    with st.form("ranking_vector"):
        st.caption("Provide ranks or scores (1=most important). Ties allowed.")
        ranks: List[float] = []
        cols = st.columns(3)
        for idx, crit in enumerate(CRITERIA):
            ranks.append(cols[idx % 3].number_input(f"Rank for {crit}", min_value=1, max_value=10, value=idx+1, step=1, key=f"r_{idx}"))
        ok = st.form_submit_button("Compute weights")
        answers = ranks if ok else ranks

try:
    w_vec = preferences_to_weights(np.array(answers, dtype=float), mode)
    weights = {CRITERIA[i]: float(w_vec[i]) for i in range(len(CRITERIA))}
except Exception as e:
    st.warning(f"Weight computation failed ({e}); using equal weights.")
    weights = equal_weights()

with st.expander("Weights in use"):
    st.json(weights)

# ---------------------------------------------------------------------
# TASK 4 — Normalization, ranking, sensitivity
# ---------------------------------------------------------------------
norm_df = normalize_criteria(work, benefit_cols=BENEFIT_COLUMNS, cost_cols=COST_COLUMNS)

st.header("Ranking")
top_n = st.slider("Top N", 5, 30, 15, 1)
rank_df = rank_municipalities(norm_df, weights, top_n=top_n)
st.dataframe(rank_df, use_container_width=True)

st.subheader("Sensitivity (±10% on top-3 criteria)")
top3 = [k for k, _ in sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:3]]
w_plus = perturb_weights(weights, top3, +0.10)
w_minus = perturb_weights(weights, top3, -0.10)
rank_plus = rank_municipalities(norm_df, w_plus, top_n=top_n)[["Nombre","Score"]].assign(case="+10%")
rank_minus = rank_municipalities(norm_df, w_minus, top_n=top_n)[["Nombre","Score"]].assign(case="-10%")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Weights +10% (top-3)**")
    st.dataframe(rank_plus, use_container_width=True)
with c2:
    st.markdown("**Weights -10% (top-3)**")
    st.dataframe(rank_minus, use_container_width=True)

def _topk(df_: pd.DataFrame, k: int = 5) -> set[str]:
    return set(df_.head(k)["Nombre"].tolist())
base5, plus5, minus5 = _topk(rank_df), _topk(rank_plus), _topk(rank_minus)
st.info(f"Top-5 stability — base∩(+10%): {len(base5 & plus5)}/5; base∩(-10%): {len(base5 & minus5)}/5")

# ---------------------------------------------------------------------
# Data / Download
# ---------------------------------------------------------------------
st.header("Data")
data_view = df.merge(
    norm_df[[c for c in norm_df.columns if c.startswith("NORM_") or c in ["codigo","Nombre","AccessibilityHoursMonthly"]]],
    on=["codigo","Nombre"], how="left",
)
st.dataframe(data_view.head(50), use_container_width=True)
st.download_button("Download normalized data (CSV)", data=data_view.to_csv(index=False).encode("utf-8"),
                   file_name="lodcore_normalized.csv", mime="text/csv")

st.caption(
    "Assumptions: *ClusterEstadistica are benefit scores (higher=better); "
    "*ClusterPoblacion ignored; accessibility times are minutes (one-way); "
    "round-trip factor=2; population <50k filter always on."
)
