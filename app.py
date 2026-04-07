"""
═══════════════════════════════════════════════════════════════════════
  DATA MINING STUDIO  ·  Professional Analytics Platform
  Ragsdale Ch.10 · Scikit-learn · Streamlit
═══════════════════════════════════════════════════════════════════════
"""
import warnings; warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, re

# ───────────────────────────────────────────────────────────────────
# PAGE SETUP
# ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Mining Studio",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* deliberately override to Space Grotesk — modern, distinct from generic AI */
html, body, [class*="css"], .stMarkdown { font-family: 'Space Grotesk', sans-serif !important; }
code, pre, .mono { font-family: 'JetBrains Mono', monospace !important; }

/* ── Tokens ─────────────────────────────────────────────────── */
:root {
  --c-bg:      #0d1117;
  --c-surface: #161b22;
  --c-border:  #30363d;
  --c-text:    #f0f6fc;
  --c-muted:   #b1bac4;
  --c-accent:  #58a6ff;
  --c-green:   #3fb950;
  --c-yellow:  #d29922;
  --c-red:     #f85149;
  --c-purple:  #bc8cff;
}

/* ── Global ─────────────────────────────────────────────────── */
.stApp { background: #0d1117; color: #f0f6fc; }
.stApp p, .stApp li, .stApp span, .stApp div { font-size: .95rem !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px; }

/* ── Header ─────────────────────────────────────────────────── */
.studio-header {
  display: flex; align-items: center; gap: 1rem;
  padding: 1.4rem 2rem; border-radius: 12px;
  background: linear-gradient(135deg, #161b22 0%, #1c2230 50%, #161b22 100%);
  border: 1px solid #30363d; margin-bottom: 1.2rem;
}
.studio-header .logo { font-size: 2rem; }
.studio-header h1 { font-size: 1.6rem; font-weight: 700; margin: 0; color: #e6edf3; letter-spacing: -.5px; }
.studio-header p  { font-size: .8rem; color: #8b949e; margin: .2rem 0 0; }

/* ── Cards ───────────────────────────────────────────────────── */
.card {
  background: #161b22; border: 1px solid #30363d;
  border-radius: 10px; padding: 1.1rem 1.3rem; margin: .5rem 0;
}
.card-accent { border-left: 3px solid #58a6ff; }
.card-green  { border-left: 3px solid #3fb950; }
.card-yellow { border-left: 3px solid #d29922; }
.card-red    { border-left: 3px solid #f85149; }
.card-purple { border-left: 3px solid #bc8cff; }

/* ── Metric tiles ────────────────────────────────────────────── */
.metric-tile {
  background: #161b22; border: 1px solid #30363d;
  border-radius: 10px; padding: 1rem 1.2rem;
  transition: border-color .2s;
}
.metric-tile:hover { border-color: #58a6ff; }
.metric-tile .m-label { font-size: .78rem; color: #b1bac4; font-weight: 600;
  text-transform: uppercase; letter-spacing: .7px; }
.metric-tile .m-value { font-size: 1.8rem; font-weight: 700; color: #f0f6fc; line-height: 1.1; }
.metric-tile .m-sub   { font-size: .82rem; color: #8b949e; margin-top: .2rem; }
.metric-tile .m-badge {
  display: inline-block; padding: .15rem .55rem; border-radius: 20px;
  font-size: .72rem; font-weight: 600; margin-top: .3rem;
}
.badge-green  { background: #1a3a1f; color: #3fb950; }
.badge-yellow { background: #3a2f0a; color: #d29922; }
.badge-red    { background: #3a0f0f; color: #f85149; }
.badge-blue   { background: #0d2044; color: #58a6ff; }

/* ── Steps ───────────────────────────────────────────────────── */
.step-header {
  display: flex; align-items: center; gap: .6rem;
  font-size: 1rem; font-weight: 700; color: #e6edf3;
  padding: .6rem 0; border-bottom: 1px solid #30363d; margin: 1.2rem 0 .8rem;
}
.step-num {
  background: #58a6ff; color: #0d1117; font-size: .75rem;
  font-weight: 700; width: 22px; height: 22px;
  border-radius: 50%; display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}

/* ── Tag chips ───────────────────────────────────────────────── */
.tag {
  display: inline-block; padding: .15rem .55rem; border-radius: 20px;
  font-size: .73rem; font-weight: 500;
  background: #161b22; border: 1px solid #30363d; color: #8b949e;
  margin: .15rem .1rem;
}
.tag-blue   { background: #0d2044; border-color: #1f4080; color: #58a6ff; }
.tag-green  { background: #1a3a1f; border-color: #1e5028; color: #3fb950; }
.tag-yellow { background: #3a2f0a; border-color: #5a4412; color: #d29922; }
.tag-purple { background: #2d1f4a; border-color: #4a2e8a; color: #bc8cff; }

/* ── Chat bubbles ────────────────────────────────────────────── */
.bubble-user {
  background: #1c2230; border: 1px solid #1f4080;
  border-radius: 10px; padding: .9rem 1.1rem; margin: .5rem 0;
  font-size: .9rem; line-height: 1.6;
}
.bubble-sys {
  background: #161b22; border: 1px solid #30363d;
  border-radius: 10px; padding: .9rem 1.1rem; margin: .5rem 0;
  font-size: .93rem; color: #b1bac4; line-height: 1.6;
}
.bubble-sys b { color: #58a6ff; }
.bubble-sys code { background: #0d1117; padding: .1rem .3rem; border-radius: 4px;
  font-size: .82rem; color: #3fb950; }

/* ── Section divider ─────────────────────────────────────────── */
.divider {
  height: 1px; background: linear-gradient(90deg, transparent, #30363d, transparent);
  margin: 1.2rem 0;
}

/* ── Streamlit overrides ─────────────────────────────────────── */
.stSelectbox>div>div, .stMultiSelect>div>div,
.stTextInput>div>div, .stTextArea>div>textarea {
  background: #161b22 !important; border-color: #30363d !important; color: #e6edf3 !important;
}
.stSlider .stSlider { color: #58a6ff !important; }
.stButton>button {
  background: #21262d; border: 1px solid #30363d; color: #e6edf3;
  border-radius: 8px; font-weight: 600; transition: all .2s;
}
.stButton>button:hover { background: #30363d; border-color: #58a6ff; color: #58a6ff; }
.stButton>button[kind="primary"] {
  background: #1f6feb !important; border-color: #1f6feb !important; color: white !important;
}
.stButton>button[kind="primary"]:hover { background: #388bfd !important; }
.stTabs [data-baseweb="tab-list"] { background: #161b22; border-bottom: 1px solid #30363d; gap: 0; }
.stTabs [data-baseweb="tab"] {
  background: transparent; color: #b1bac4; border-radius: 0;
  padding: .6rem 1.1rem; font-weight: 500; font-size: .92rem;
}
.stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }
.stDataFrame { background: #161b22 !important; }
[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #30363d; }
.stProgress > div > div { background: #1f6feb !important; }
hr { border-color: #30363d !important; }
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────
# SESSION STATE
# ───────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "stage": "upload",               # upload → profile → confirm → run → results
    "sheets": {},                    # {key: df}
    "profile": {},                   # data profile per sheet
    "context": {},                   # user answers
    "plan": {},                      # resolved analysis plan
    "cfg": {},                       # run configuration
    "results": {},                   # analysis outputs
    "pending_rerun": False,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v
S = st.session_state


# ───────────────────────────────────────────────────────────────────
# HEADER
# ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="studio-header">
  <div class="logo">◈</div>
  <div>
    <h1>Data Mining Studio</h1>
  </div>
</div>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────
# SIDEBAR
# ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ◈ Navigation")
    stages = ["upload","profile","confirm","run","results"]
    stage_labels = {"upload":"① Upload","profile":"② Understand","confirm":"③ Configure","run":"⏳ Running","results":"④ Results"}
    current_idx = stages.index(S["stage"]) if S["stage"] in stages else 0
    for i, (sg, lb) in enumerate(stage_labels.items()):
        active = sg == S["stage"]
        done   = i < current_idx
        color  = "#58a6ff" if active else ("#3fb950" if done else "#30363d")
        st.markdown(f"""<div style="padding:.35rem .5rem;border-left:2px solid {color};
        margin:.15rem 0;font-size:.88rem;color:{'#f0f6fc' if active else '#b1bac4'}">{lb}</div>""",
        unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("📖 Methodology", expanded=False):
        st.markdown("""
**Classification** — predict a discrete label  
LDA · Logistic Regression · KNN · Decision Tree  
Naïve Bayes · Neural Network (MLP) · SVM

**Regression** — predict a continuous value  
Linear · Ridge · Lasso · Tree · MLP

**Clustering** — find natural groups  
K-Means · Hierarchical (Ward)

**Association Rules** — discover co-occurrence  
Apriori: Support · Confidence · Lift

**Key metrics**  
`Sensitivity = TP/(TP+FN)` — recall for positive class  
`Specificity = TN/(TN+FP)` — recall for negative class  
`Cutoff` — probability threshold; lower → higher sensitivity  
`Lift > 1` — rule is better than random chance  
`R²` — variance explained (regression)

**Data integrity**  
Target encoding is fit on training rows only.  
Decision Tree default depth = 6 (prevents memorisation).  
If val accuracy is ≥99%, read the ⚠️ warning in results.
        """)
    with st.expander("🔤 Encoding guide", expanded=False):
        st.markdown("""
**Label Encoding** — ordinal categories (Low<Med<High)  
**One-Hot Encoding** — nominal categories (no order)  
**Target Encoding** — mean of target per category  
**Binary Encoding** — many categories, compact  
**Ordinal Encoding** — custom rank order  

App detects encoding needs automatically and lets you override per column.
        """)

    if S["stage"] not in ("upload","profile"):
        st.markdown("---")
        if st.button("↺ Start over", use_container_width=True):
            for k in list(S.keys()): del S[k]
            for k,v in _DEFAULTS.items(): S[k] = v
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PREPROCESSING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def profile_dataframe(df: pd.DataFrame) -> dict:
    """Deep profile of a dataframe — types, missings, cardinality, skew."""
    p = {}
    for col in df.columns:
        s = df[col]
        n_miss  = int(s.isna().sum())
        n_uniq  = int(s.nunique(dropna=True))
        dtype   = str(s.dtype)
        miss_pct = n_miss / len(s) if len(s) > 0 else 0

        info = {
            "dtype": dtype,
            "n_miss": n_miss,
            "miss_pct": round(miss_pct * 100, 1),
            "n_uniq": n_uniq,
            "cardinality": round(n_uniq / max(len(s.dropna()), 1), 3),
        }

        if pd.api.types.is_numeric_dtype(s):
            info["kind"] = "numeric"
            info["min"]  = float(s.min()) if n_uniq > 0 else None
            info["max"]  = float(s.max()) if n_uniq > 0 else None
            info["mean"] = float(s.mean()) if n_uniq > 0 else None
            info["std"]  = float(s.std())  if n_uniq > 1 else 0.0
            info["skew"] = float(s.skew()) if n_uniq > 2 else 0.0
            info["binary"] = n_uniq == 2
        elif pd.api.types.is_bool_dtype(s):
            info["kind"]   = "boolean"
            info["binary"] = True
        else:
            info["kind"] = "categorical"
            info["binary"] = n_uniq == 2
            info["top_values"] = s.value_counts().head(5).to_dict()

            # Detect datetime-like strings
            try:
                parsed = pd.to_datetime(s.dropna().head(30), errors="coerce")
                if parsed.notna().mean() > 0.8:
                    info["kind"] = "datetime_str"
            except Exception:
                pass

        # Encoding suggestion
        if info["kind"] == "categorical":
            if n_uniq == 2:
                info["enc_suggest"] = "label"
            elif n_uniq <= 8:
                info["enc_suggest"] = "onehot"
            elif n_uniq <= 30:
                info["enc_suggest"] = "target"
            else:
                info["enc_suggest"] = "drop"   # too high cardinality (IDs)
        elif info["kind"] == "datetime_str":
            info["enc_suggest"] = "drop"
        else:
            info["enc_suggest"] = "passthrough"

        p[col] = info
    return p


def suggest_targets(df: pd.DataFrame, profile: dict) -> list:
    """Return ranked list of candidate target columns."""
    candidates = []
    for col, info in profile.items():
        score = 0
        reason = []
        if info["kind"] == "numeric" and info.get("binary"):
            score += 3; reason.append("binary numeric (0/1)")
        elif info["kind"] in ("boolean",) or info.get("binary"):
            score += 3; reason.append("binary column")
        elif info["kind"] == "categorical" and 2 <= info["n_uniq"] <= 15:
            score += 2; reason.append(f"categorical ({info['n_uniq']} classes)")
        elif info["kind"] == "numeric" and info["n_uniq"] > 15:
            score += 1; reason.append("continuous numeric → regression")

        # Boost for common target names
        col_l = col.lower()
        for kw in ["fraud","label","target","class","churn","default","status",
                   "flag","result","outcome","y","diagnosis","risk","type"]:
            if kw in col_l:
                score += 2; reason.append(f"name contains '{kw}'"); break

        # Penalise IDs / timestamps
        for kw in ["id","key","num","audit","date","time","index","seq"]:
            if col_l == kw or col_l.endswith(kw) or col_l.startswith(kw):
                score -= 3; break

        if info["miss_pct"] > 30:
            score -= 1

        if score > 0:
            candidates.append({"col": col, "score": score,
                                "reason": ", ".join(reason), "info": info})
    candidates.sort(key=lambda x: -x["score"])
    return candidates


def smart_preprocess(df: pd.DataFrame, profile: dict,
                     target_col: str,
                     enc_overrides: dict,     # {col: enc_method}
                     impute_strategy: str,
                     drop_cols: list,
                     task: str,
                     train_idx=None) -> tuple:
    """
    Full preprocessing pipeline.
    train_idx: optional array of training row indices — used to fit target encoding
               on training rows only, preventing data leakage.
    Returns (X_df, y_series, feature_names, enc_log, label_maps)
    """
    enc_log    = []
    label_maps = {}     # col → {orig → encoded}

    df2 = df.copy()

    # 1. Drop specified columns
    drop_actual = [c for c in drop_cols if c in df2.columns and c != target_col]
    if drop_actual:
        df2.drop(columns=drop_actual, inplace=True)
        enc_log.append(f"🗑 Dropped {len(drop_actual)} column(s): {', '.join(f'`{c}`' for c in drop_actual)}")

    # 2. Separate target
    y_raw = df2.pop(target_col)
    feature_cols = list(df2.columns)

    # 3. Impute missing values
    for col in feature_cols:
        n_miss = df2[col].isna().sum()
        if n_miss == 0:
            continue
        info = profile.get(col, {})
        if info.get("kind") == "numeric":
            if impute_strategy == "median":
                val = df2[col].median()
            elif impute_strategy == "mean":
                val = df2[col].mean()
            else:   # mode
                val = df2[col].mode().iloc[0] if not df2[col].mode().empty else 0
            df2[col].fillna(val, inplace=True)
            enc_log.append(f"🔧 `{col}`: filled {n_miss} missing values with {impute_strategy} ({val:.3g})")
        else:
            mode_val = df2[col].mode().iloc[0] if not df2[col].mode().empty else "Unknown"
            df2[col].fillna(mode_val, inplace=True)
            enc_log.append(f"🔧 `{col}`: filled {n_miss} missing values with mode ('{mode_val}')")

    # 4. Encode categorical / boolean columns
    new_cols = []
    to_drop  = []

    for col in list(df2.columns):
        info  = profile.get(col, {})
        enc   = enc_overrides.get(col, info.get("enc_suggest", "passthrough"))
        kind  = info.get("kind", "numeric")

        if kind == "numeric" or kind == "datetime_str":
            if kind == "datetime_str":
                df2.drop(columns=[col], inplace=True, errors="ignore")
                enc_log.append(f"🗑 `{col}`: dropped (datetime-like)")
            continue

        if kind in ("categorical", "boolean"):
            if enc == "drop":
                to_drop.append(col)
                enc_log.append(f"🗑 `{col}`: dropped (high-cardinality / ID-like, {info['n_uniq']} unique)")

            elif enc == "label":
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df2[col] = le.fit_transform(df2[col].astype(str))
                label_maps[col] = dict(zip(le.classes_, le.transform(le.classes_)))
                enc_log.append(f"🔢 `{col}` → Label Encoding: {dict(list(label_maps[col].items())[:4])}")

            elif enc == "onehot":
                dummies = pd.get_dummies(df2[col].astype(str), prefix=col, drop_first=True, dtype=int)
                df2 = pd.concat([df2, dummies], axis=1)
                new_cols.extend(dummies.columns.tolist())
                to_drop.append(col)
                enc_log.append(f"🔠 `{col}` → One-Hot ({dummies.shape[1]} new cols): {list(dummies.columns[:3])}…")

            elif enc == "target" and target_col:
                # Target encoding — fit only on training rows to avoid leakage
                if train_idx is not None:
                    enc_map = y_raw.iloc[train_idx].groupby(df2[col].iloc[train_idx]).mean().to_dict()
                else:
                    enc_map = y_raw.groupby(df2[col]).mean().to_dict()
                df2[col + "_tenc"] = df2[col].map(enc_map).fillna(y_raw.mean())
                to_drop.append(col)
                label_maps[col] = enc_map
                enc_log.append(f"🎯 `{col}` → Target Encoding (fit on train only — leakage-free)")

            elif enc == "ordinal":
                from sklearn.preprocessing import OrdinalEncoder
                oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                df2[col] = oe.fit_transform(df2[[col]]).astype(int)
                enc_log.append(f"📊 `{col}` → Ordinal Encoding")

            else:  # passthrough: try numeric coerce
                coerced = pd.to_numeric(df2[col], errors="coerce")
                if coerced.notna().mean() > 0.8:
                    df2[col] = coerced.fillna(coerced.median())
                    enc_log.append(f"🔢 `{col}` → coerced to numeric")
                else:
                    to_drop.append(col)
                    enc_log.append(f"🗑 `{col}` → dropped (cannot encode automatically)")

    if to_drop:
        df2.drop(columns=[c for c in to_drop if c in df2.columns], inplace=True, errors="ignore")

    # 5. Encode target
    y_enc_log = ""
    if y_raw.dtype == object or y_raw.dtype.name == "bool" or y_raw.nunique() <= 20:
        from sklearn.preprocessing import LabelEncoder
        le_y = LabelEncoder()
        y_out = pd.Series(le_y.fit_transform(y_raw.astype(str).fillna("missing")),
                          name=target_col, index=y_raw.index)
        mapping = dict(zip(le_y.classes_, le_y.transform(le_y.classes_)))
        label_maps[f"__target__{target_col}"] = mapping
        y_enc_log = f"🎯 Target `{target_col}` encoded: {mapping}"
    else:
        y_out = y_raw.fillna(y_raw.median()).astype(float)
        y_out.name = target_col

    if y_enc_log:
        enc_log.append(y_enc_log)

    # 6. Ensure all numeric
    for col in list(df2.columns):
        if not pd.api.types.is_numeric_dtype(df2[col]):
            df2.drop(columns=[col], inplace=True, errors="ignore")

    df2 = df2.fillna(0)
    feature_names = list(df2.columns)

    return df2, y_out, feature_names, enc_log, label_maps


# ═══════════════════════════════════════════════════════════════════════════
# OVERSAMPLING / BALANCING
# ═══════════════════════════════════════════════════════════════════════════

def balance_classes(X_arr, y_arr, method: str, seed: int):
    """balance_method ∈ {none, oversample_random, smote, undersample}"""
    from collections import Counter
    counts = Counter(y_arr)
    log = [f"Class distribution before: {dict(counts)}"]

    if method == "none" or len(counts) < 2:
        return X_arr, y_arr, log

    if method == "oversample_random":
        from sklearn.utils import resample
        classes = list(counts.keys())
        max_n   = max(counts.values())
        X_parts, y_parts = [], []
        for cls in classes:
            idx = np.where(y_arr == cls)[0]
            if len(idx) < max_n:
                idx_up = resample(idx, replace=True, n_samples=max_n, random_state=seed)
            else:
                idx_up = idx
            X_parts.append(X_arr[idx_up])
            y_parts.append(y_arr[idx_up])
        X_b = np.vstack(X_parts)
        y_b = np.concatenate(y_parts)
        perm = np.random.RandomState(seed).permutation(len(X_b))
        log.append(f"✅ Random oversampling → {len(X_b)} samples (balanced {max_n} per class)")
        return X_b[perm], y_b[perm], log

    if method == "smote":
        try:
            from imblearn.over_sampling import SMOTE
            min_cls = min(counts.values())
            k = max(1, min(5, min_cls - 1))
            sm = SMOTE(k_neighbors=k, random_state=seed)
            X_b, y_b = sm.fit_resample(X_arr, y_arr)
            log.append(f"✅ SMOTE → {len(X_b)} samples")
            return X_b, y_b, log
        except Exception as e:
            log.append(f"⚠️ SMOTE failed ({e}), using random oversample")
            return balance_classes(X_arr, y_arr, "oversample_random", seed)

    if method == "undersample":
        min_n = min(counts.values())
        X_parts, y_parts = [], []
        for cls in counts:
            idx = np.where(y_arr == cls)[0]
            idx_d = np.random.RandomState(seed).choice(idx, min_n, replace=False)
            X_parts.append(X_arr[idx_d]); y_parts.append(y_arr[idx_d])
        X_b = np.vstack(X_parts)
        y_b = np.concatenate(y_parts)
        perm = np.random.RandomState(seed).permutation(len(X_b))
        log.append(f"✅ Random undersample → {len(X_b)} samples ({min_n} per class)")
        return X_b[perm], y_b[perm], log

    return X_arr, y_arr, log


# ═══════════════════════════════════════════════════════════════════════════
# MODELS REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

def build_clf(name: str, params: dict, seed: int):
    """Build classifier by name with custom params."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    registry = {
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=params.get("max_iter", 1000),
            C=params.get("C", 1.0),
            solver="saga", random_state=seed),
        "LDA": lambda: LinearDiscriminantAnalysis(
            solver=params.get("solver", "svd")),
        "KNN": lambda: KNeighborsClassifier(
            n_neighbors=params.get("k", 5),
            weights=params.get("weights", "uniform")),
        "Decision Tree": lambda: DecisionTreeClassifier(
            max_depth=params.get("max_depth", 6),
            min_samples_leaf=params.get("min_samples_leaf", 4),
            criterion=params.get("criterion", "gini"),
            random_state=seed),
        "Naïve Bayes": lambda: GaussianNB(
            var_smoothing=params.get("var_smoothing", 1e-9)),
        "Neural Network": lambda: MLPClassifier(
            hidden_layer_sizes=tuple(params.get("hidden_layers", [64, 32])),
            activation=params.get("activation", "relu"),
            alpha=params.get("alpha", 1e-4),
            max_iter=params.get("max_iter_nn", 500),
            random_state=seed),
        "SVM": lambda: SVC(
            C=params.get("svm_C", 1.0),
            kernel=params.get("kernel", "rbf"),
            probability=True, random_state=seed),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("rf_max_depth", None),
            min_samples_leaf=params.get("rf_min_leaf", 3),
            random_state=seed),
    }
    return registry[name]()


def build_reg(name: str, params: dict, seed: int):
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor

    registry = {
        "Linear Regression": lambda: LinearRegression(),
        "Ridge": lambda: Ridge(alpha=params.get("ridge_alpha", 1.0)),
        "Lasso": lambda: Lasso(alpha=params.get("lasso_alpha", 0.1), max_iter=2000),
        "Decision Tree": lambda: DecisionTreeRegressor(
            max_depth=params.get("max_depth", 6),
            min_samples_leaf=params.get("min_samples_leaf", 4),
            random_state=seed),
        "Neural Network": lambda: MLPRegressor(
            hidden_layer_sizes=tuple(params.get("hidden_layers", [64, 32])),
            activation=params.get("activation", "relu"),
            alpha=params.get("alpha", 1e-4),
            max_iter=params.get("max_iter_nn", 500),
            random_state=seed),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("rf_max_depth", None),
            random_state=seed),
    }
    return registry[name]()


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def train_classify(clf, Xtr, ytr, Xval, yval, cutoff: float = 0.5):
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                  f1_score, precision_score, recall_score)
    clf.fit(Xtr, ytr)

    # Probability-based prediction with custom cutoff (binary only)
    if hasattr(clf, "predict_proba") and len(np.unique(yval)) == 2:
        prob_val = clf.predict_proba(Xval)
        # find index for positive class (class=1 if exists, else last)
        classes = list(clf.classes_)
        pos_idx = classes.index(1) if 1 in classes else -1
        prob_pos = prob_val[:, pos_idx]
        yp_val = (prob_pos >= cutoff).astype(int)
        prob_tr = clf.predict_proba(Xtr)[:, pos_idx]
    else:
        yp_val  = clf.predict(Xval)
        prob_pos = None
        prob_tr  = None

    yp_tr = clf.predict(Xtr)
    cm    = confusion_matrix(yval, yp_val)

    avg = "binary" if len(np.unique(yval)) == 2 else "weighted"
    result = {
        "clf": clf,
        "train_acc": float(accuracy_score(ytr,  yp_tr)),
        "val_acc":   float(accuracy_score(yval, yp_val)),
        "f1":        float(f1_score(yval, yp_val, average=avg, zero_division=0)),
        "precision": float(precision_score(yval, yp_val, average=avg, zero_division=0)),
        "recall":    float(recall_score(yval, yp_val, average=avg, zero_division=0)),
        "cm": cm,
        "yp_val": yp_val,
        "prob_pos": prob_pos,
        "prob_tr":  prob_tr,
    }

    if cm.size == 4:  # binary
        tn, fp, fn, tp = cm.ravel()
        result["sensitivity"] = float(tp/(tp+fn)) if (tp+fn) > 0 else 0.0
        result["specificity"] = float(tn/(tn+fp)) if (tn+fp) > 0 else 0.0
    else:
        result["sensitivity"] = result["recall"]
        result["specificity"] = 0.0

    return result


def train_regress(reg, Xtr, ytr, Xval, yval):
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    reg.fit(Xtr, ytr)
    yp_val = reg.predict(Xval)
    yp_tr  = reg.predict(Xtr)
    return {
        "reg": reg,
        "train_r2": float(r2_score(ytr, yp_tr)),
        "val_r2":   float(r2_score(yval, yp_val)),
        "mae":  float(mean_absolute_error(yval, yp_val)),
        "rmse": float(np.sqrt(mean_squared_error(yval, yp_val))),
        "yp_val": yp_val,
    }


def cutoff_sweep(prob_pos, y_true, n_points=80):
    """Sweep cutoff 0.05 → 0.95, return acc/sens/spec arrays."""
    cuts = np.linspace(0.05, 0.95, n_points)
    accs, senss, specs = [], [], []
    from sklearn.metrics import accuracy_score, confusion_matrix
    for c in cuts:
        yp = (prob_pos >= c).astype(int)
        accs.append(accuracy_score(y_true, yp))
        cm = confusion_matrix(y_true, yp)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            senss.append(tp/(tp+fn) if (tp+fn) > 0 else 0)
            specs.append(tn/(tn+fp) if (tn+fp) > 0 else 0)
        else:
            senss.append(0); specs.append(0)
    return cuts, accs, senss, specs


def feature_importances(result: dict, feat_names: list):
    clf = result.get("clf") or result.get("reg")
    if clf is None: return None
    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=feat_names)
    if hasattr(clf, "coef_"):
        c = clf.coef_
        if c.ndim > 1: c = np.abs(c).mean(0)
        return pd.Series(np.abs(c), index=feat_names)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# PLOTLY CHART HELPERS (dark theme)
# ═══════════════════════════════════════════════════════════════════════════

DARK = dict(
    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
    font=dict(family="Space Grotesk", color="#f0f6fc", size=12),
    title_font=dict(size=14, color="#f0f6fc"),
    legend=dict(bgcolor="#0d1117", bordercolor="#30363d", borderwidth=1,
                font=dict(color="#f0f6fc")),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", color="#b1bac4",
               tickfont=dict(color="#b1bac4", size=11)),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", color="#b1bac4",
               tickfont=dict(color="#b1bac4", size=11)),
)


def dark_fig(fig, height=380):
    fig.update_layout(**DARK, height=height, margin=dict(l=50,r=30,t=45,b=40))
    return fig


def cm_plotly(cm, title="Confusion Matrix", class_labels=None):
    if class_labels is None:
        class_labels = [str(c) for c in range(cm.shape[0])]
    # Normalise rows for annotation
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
    annotations = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations.append(dict(
                x=j, y=i,
                text=f"<b>{cm[i,j]}</b><br><span style='font-size:10px'>{cm_norm[i,j]:.0%}</span>",
                showarrow=False,
                font=dict(color="white" if cm_norm[i,j] > 0.5 else "#e6edf3")
            ))
    fig = go.Figure(go.Heatmap(
        z=cm_norm, x=[f"Pred: {l}" for l in class_labels],
        y=[f"True: {l}" for l in class_labels],
        colorscale="Blues", showscale=False, zmin=0, zmax=1,
    ))
    fig.update_layout(annotations=annotations, title=title)
    return dark_fig(fig, height=320)


def corr_heatmap(df, title="Correlation Matrix"):
    corr = df.corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto", title=title)
    fig.update_coloraxes(colorbar_tickfont_color="#8b949e",
                          colorbar_title_font_color="#8b949e")
    return dark_fig(fig, height=max(350, 50*len(corr.columns)))


def compare_bar(df_metrics: pd.DataFrame, metric: str, title: str, color_col=None):
    fig = px.bar(df_metrics.sort_values(metric, ascending=True),
                 x=metric, y="Model", orientation="h",
                 title=title, text=metric,
                 color=color_col or metric,
                 color_continuous_scale="Teal")
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_coloraxes(showscale=False)
    return dark_fig(fig, height=max(280, 45*len(df_metrics)))


def cutoff_chart(cuts, accs, senss, specs, opt_cut, title="Cutoff Analysis"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cuts, y=accs, name="Accuracy", line=dict(color="#58a6ff", width=2)))
    fig.add_trace(go.Scatter(x=cuts, y=senss, name="Sensitivity", line=dict(color="#f85149", width=2)))
    fig.add_trace(go.Scatter(x=cuts, y=specs, name="Specificity", line=dict(color="#3fb950", width=2)))
    fig.add_vline(x=opt_cut, line_dash="dash", line_color="#d29922",
                  annotation_text=f"cutoff={opt_cut:.2f}",
                  annotation_font_color="#d29922")
    fig.update_layout(title=title, xaxis_title="Cutoff threshold", yaxis_title="Rate",
                       yaxis_range=[0, 1.05])
    return dark_fig(fig, height=380)


def scatter_avp(y_true, y_pred, title="Actual vs Predicted"):
    fig = px.scatter(x=y_true, y=y_pred, title=title,
                     labels={"x": "Actual", "y": "Predicted"},
                     opacity=0.55, color_discrete_sequence=["#58a6ff"])
    lo = min(float(min(y_true)), float(min(y_pred)))
    hi = max(float(max(y_true)), float(max(y_pred)))
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                  line=dict(color="#30363d", dash="dash"))
    return dark_fig(fig, height=380)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 0 — UPLOAD
# ═══════════════════════════════════════════════════════════════════════════
if S["stage"] == "upload":
    st.markdown("""
    <div class="step-header">
      <div class="step-num">1</div> Upload your data
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        uploaded = st.file_uploader(
            "Drop CSV or Excel files here (multiple sheets supported)",
            type=["csv","xlsx","xls"], accept_multiple_files=True,
            label_visibility="collapsed"
        )

    with col_b:
        st.markdown("""
        <div class="card card-accent">
        <div class="tag tag-blue">CSV</div>
        <div class="tag tag-blue">XLSX</div>
        <div class="tag tag-green">Multi-sheet</div>
        <div class="tag tag-green">Mixed types</div>
        <div class="tag tag-purple">Text + Numeric</div>
        <br><br>
        <span style="color:#8b949e;font-size:.83rem">
        All column types handled automatically:<br>
        numeric · categorical · binary · text-encoded · datetime
        </span>
        </div>
        """, unsafe_allow_html=True)

    if uploaded:
        sheets = {}
        with st.spinner("Reading files…"):
            for f in uploaded:
                if f.size > 80_000_000:
                    st.error(f"⛔ `{f.name}` exceeds 80 MB limit"); continue
                try:
                    if f.name.lower().endswith(".csv"):
                        df = pd.read_csv(f, low_memory=False)
                        sheets[f.name] = df
                    else:
                        xl = pd.read_excel(f, sheet_name=None)
                        for sname, df in xl.items():
                            sheets[f"{f.name} › {sname}"] = df
                except Exception as e:
                    st.error(f"Error reading `{f.name}`: {e}")

        if sheets:
            S["sheets"] = sheets
            # Auto-profile
            S["profile"] = {k: profile_dataframe(df) for k, df in sheets.items()}
            S["stage"]   = "profile"
            st.rerun()
    else:
        st.markdown("""
        <div class="card" style="margin-top:1rem;">
          <span style="color:#8b949e;font-size:.85rem">
          No file yet. Once uploaded, the platform will profile every column
          and ask you what you want to achieve before running any model.
          </span>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — PROFILE + UNDERSTAND
# ═══════════════════════════════════════════════════════════════════════════
if S["stage"] == "profile":
    st.markdown("""
    <div class="step-header">
      <div class="step-num">2</div> Understand your data & goals
    </div>
    """, unsafe_allow_html=True)

    # ── Dataset overview ──────────────────────────────────────────
    tabs_data = st.tabs([f"  {k[:28]}  " for k in S["sheets"].keys()])

    for tab, (sname, df) in zip(tabs_data, S["sheets"].items()):
        with tab:
            prof = S["profile"][sname]
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(f"""<div class="metric-tile"><div class="m-label">Rows</div><div class="m-value">{df.shape[0]:,}</div></div>""", unsafe_allow_html=True)
            with c2: st.markdown(f"""<div class="metric-tile"><div class="m-label">Columns</div><div class="m-value">{df.shape[1]}</div></div>""", unsafe_allow_html=True)
            with c3:
                miss_pct = round(df.isna().sum().sum() / max(df.size, 1) * 100, 1)
                badge_cls = "badge-green" if miss_pct < 5 else ("badge-yellow" if miss_pct < 20 else "badge-red")
                st.markdown(f"""<div class="metric-tile"><div class="m-label">Missing</div><div class="m-value">{miss_pct}%</div><div class="m-badge {badge_cls}">{badge_cls.split('-')[1].upper()}</div></div>""", unsafe_allow_html=True)
            with c4:
                n_cat = sum(1 for v in prof.values() if v.get("kind") == "categorical")
                st.markdown(f"""<div class="metric-tile"><div class="m-label">Categorical cols</div><div class="m-value">{n_cat}</div><div class="m-sub">rest: numeric/bool</div></div>""", unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # Column profiler table
            rows = []
            for col, info in prof.items():
                kind_tag = {"numeric":"🔢","categorical":"🔤","boolean":"☑","datetime_str":"📅"}.get(info.get("kind","?"),"❓")
                enc_sug  = info.get("enc_suggest","—")
                rows.append({
                    "Column": col,
                    "Type": f"{kind_tag} {info.get('kind','?')}",
                    "Missing": f"{info['miss_pct']}%",
                    "Unique": info["n_uniq"],
                    "Encoding suggestion": enc_sug,
                    "Notes": ("⚠️ high missing" if info["miss_pct"] > 30 else
                               "🆔 likely ID" if enc_sug == "drop" and info["n_uniq"] > 100 else "")
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=280)

            with st.expander("Sample rows (first 5)"):
                st.dataframe(df.head(), use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Goal intake ───────────────────────────────────────────────
    st.markdown("""
    <div class="bubble-sys">
    <b>◈ Studio:</b> Before running any model, tell us what you're trying to achieve.
    This helps select the right technique, encoding, and evaluation metrics.
    </div>
    """, unsafe_allow_html=True)

    context = {}

    c1, c2 = st.columns(2)
    with c1:
        context["description"] = st.text_area(
            "What does this data represent?",
            placeholder="e.g., Audit responses (Yes/No) from two firms for detecting management fraud. Each row is one company.",
            height=110, key="ctx_desc"
        )
    with c2:
        context["goal"] = st.text_area(
            "What do you want to achieve?",
            placeholder="e.g., Compare which questionnaire detects fraud better, combine them into a composite score, and find the optimal classification threshold.",
            height=110, key="ctx_goal"
        )

    context["extra"] = st.text_input(
        "Any specific requirements or constraints? (optional)",
        placeholder="e.g., Sensitivity is more important than Accuracy. Want to see cutoff sweep. Use seed 12345.",
        key="ctx_extra"
    )

    # Sheet selection (if multiple)
    all_sheet_names = list(S["sheets"].keys())
    if len(all_sheet_names) > 1:
        context["selected_sheets"] = st.multiselect(
            "Which sheet(s) to include in the analysis?",
            all_sheet_names, default=all_sheet_names,
            key="ctx_sheets"
        )
    else:
        context["selected_sheets"] = all_sheet_names

    # Target column
    # Aggregate suggestions from selected sheets
    sel_sheets = context.get("selected_sheets") or all_sheet_names
    all_cols   = list({c for k in sel_sheets for c in S["sheets"].get(k, pd.DataFrame()).columns})
    all_prof   = {}
    for k in sel_sheets:
        all_prof.update(S["profile"].get(k, {}))
    suggestions = suggest_targets(S["sheets"].get(sel_sheets[0], pd.DataFrame()), S["profile"].get(sel_sheets[0], {}))

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        target_options = ["(auto-detect)"] + all_cols
        default_target = suggestions[0]["col"] if suggestions else all_cols[0] if all_cols else None
        default_idx    = target_options.index(default_target) if default_target in target_options else 0
        context["target"] = st.selectbox(
            "Target / output column:",
            target_options, index=default_idx,
            help="The column to predict (classification) or estimate (regression). For clustering/association, set 'none'."
        )
        if suggestions:
            s = suggestions[0]
            st.markdown(f"""<div class="card card-accent" style="font-size:.8rem;padding:.6rem .9rem">
            🎯 <b>Suggested:</b> <code>{s['col']}</code> — {s['reason']}</div>""",
            unsafe_allow_html=True)

    with c2:
        context["task"] = st.selectbox(
            "Primary task:",
            ["(auto-detect)", "Classification", "Regression",
             "Clustering only", "Association Rules only", "Full analysis"],
            key="ctx_task"
        )

    # Check if ready
    ready = bool(context["description"].strip() or context["goal"].strip())

    if not ready:
        st.markdown("""<div class="card card-yellow" style="font-size:.83rem">
        ⚠️ Please describe your data and goal so the studio can configure the analysis correctly.
        </div>""", unsafe_allow_html=True)

    if st.button("→ Review & Configure", type="primary", disabled=not ready):
        S["context"] = context
        # Auto-resolve task
        desc_full = (context["description"] + " " + context["goal"] + " " + context["extra"]).lower()
        task = context["task"]
        if task == "(auto-detect)":
            if any(k in desc_full for k in ["classify","fraud","detect","label","categor","flag"]):
                task = "Classification"
            elif any(k in desc_full for k in ["predict","forecast","regress","value","revenue","price"]):
                task = "Regression"
            elif any(k in desc_full for k in ["cluster","segment","group"]):
                task = "Clustering only"
            elif any(k in desc_full for k in ["basket","association","apriori","co-occur","market"]):
                task = "Association Rules only"
            else:
                task = "Classification"  # safe default
        S["context"]["task_resolved"] = task

        # Auto-resolve target
        tgt = context["target"]
        if tgt == "(auto-detect)" and suggestions:
            tgt = suggestions[0]["col"]
        elif tgt == "(auto-detect)":
            tgt = None
        S["context"]["target_resolved"] = tgt

        S["stage"] = "confirm"
        st.rerun()
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — CONFIGURE
# ═══════════════════════════════════════════════════════════════════════════
if S["stage"] == "confirm":
    ctx  = S["context"]
    task = ctx["task_resolved"]
    tgt  = ctx.get("target_resolved")

    st.markdown("""
    <div class="step-header">
      <div class="step-num">3</div> Configure & Launch
    </div>
    """, unsafe_allow_html=True)

    # Show what was understood
    st.markdown(f"""
    <div class="bubble-user">
    <b>You said:</b> {ctx.get('description','')}<br>
    <b>Goal:</b> {ctx.get('goal','')}<br>
    {f'<b>Extra:</b> {ctx["extra"]}' if ctx.get("extra") else ""}
    </div>
    <div class="bubble-sys">
    <b>◈ Studio understood:</b><br>
    Task → <code>{task}</code> &nbsp;|&nbsp; Target → <code>{tgt or 'none'}</code> &nbsp;|&nbsp;
    Sheets → {", ".join(f"<code>{s}</code>" for s in ctx['selected_sheets'])}
    </div>
    """, unsafe_allow_html=True)

    # ── Work out which sheet + df to use ─────────────────────────
    sel_sheets = ctx["selected_sheets"]
    # For multi-sheet same-structure → merge on common col
    if len(sel_sheets) == 1:
        work_df   = S["sheets"][sel_sheets[0]].copy()
        work_prof = S["profile"][sel_sheets[0]]
    else:
        # try joining on common index columns
        dfs = [S["sheets"][s].copy() for s in sel_sheets]
        common_cols = set(dfs[0].columns)
        for d in dfs[1:]: common_cols &= set(d.columns)
        join_candidates = [c for c in common_cols if "id" in c.lower() or "audit" in c.lower() or "key" in c.lower()]
        if join_candidates:
            jk = join_candidates[0]
            merged = dfs[0]
            for i, d in enumerate(dfs[1:]):
                suffix = f"_{sel_sheets[i+1].split('›')[-1].strip()[:6]}"
                merged = merged.merge(d, on=jk, how="outer", suffixes=("", suffix))
            work_df = merged
        else:
            # stack vertically if same columns
            if all(set(d.columns) == set(dfs[0].columns) for d in dfs):
                work_df = pd.concat(dfs, ignore_index=True)
            else:
                work_df = dfs[0]
        work_prof = profile_dataframe(work_df)

    all_cols = list(work_df.columns)

    # ── Encoding overrides ────────────────────────────────────────
    with st.expander("🔤 Encoding overrides (optional — defaults are auto)", expanded=False):
        enc_overrides = {}
        enc_opts = ["auto", "passthrough", "label", "onehot", "target", "ordinal", "drop"]
        cat_cols = [c for c, inf in work_prof.items() if inf.get("kind") in ("categorical","boolean")]
        if cat_cols:
            sub_cols = st.columns(min(4, len(cat_cols)))
            for i, col in enumerate(cat_cols[:12]):
                with sub_cols[i % len(sub_cols)]:
                    sug = work_prof[col].get("enc_suggest","auto")
                    override = st.selectbox(f"`{col}`", enc_opts,
                                             index=enc_opts.index(sug) if sug in enc_opts else 0,
                                             key=f"enc_{col}")
                    if override != "auto":
                        enc_overrides[col] = override
        else:
            st.info("No categorical columns detected — encoding overrides not needed.")

    # ── Drop columns ──────────────────────────────────────────────
    drop_cols = st.multiselect(
        "Columns to exclude from analysis:",
        [c for c in all_cols if c != tgt],
        placeholder="Select columns to drop…",
        key="drop_cols"
    )

    # ── Data split & balancing ────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        test_size  = st.slider("Validation set size", 0.15, 0.45, 0.30, 0.05,
                                help="Fraction held out for validation/testing")
    with c2:
        seed = int(st.number_input("Random seed", value=42, step=1))
    with c3:
        impute = st.selectbox("Missing value strategy", ["median","mean","mode"],
                               help="How to fill numeric missing values")
    with c4:
        if task == "Classification" and tgt:
            balance_method = st.selectbox(
                "Class balancing",
                ["none","oversample_random","smote","undersample"],
                index=1,
                help="none=off · oversample_random=duplicate minority · smote=synthetic · undersample=trim majority"
            )
        else:
            balance_method = "none"

    # ── Model selection ───────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if task == "Classification":
        clf_all = ["LDA","Logistic Regression","KNN","Decision Tree","Naïve Bayes",
                   "Neural Network","SVM","Random Forest"]
        selected_models = st.multiselect("Classification models to run:", clf_all,
                                          default=["LDA","Logistic Regression","Decision Tree",
                                                   "Naïve Bayes","Neural Network"],
                                          key="sel_clf")
    elif task == "Regression":
        reg_all = ["Linear Regression","Ridge","Lasso","Decision Tree","Neural Network","Random Forest"]
        selected_models = st.multiselect("Regression models to run:", reg_all,
                                          default=["Linear Regression","Decision Tree","Neural Network"],
                                          key="sel_reg")
    else:
        selected_models = []

    run_clustering  = task in ("Clustering only","Full analysis") or (task == "Classification" and st.checkbox("Also run K-Means clustering", value=False))
    run_association = task in ("Association Rules only","Full analysis") or (task != "Clustering only" and st.checkbox("Also run Association Rules", value=False))

    # ── Advanced model params ─────────────────────────────────────
    with st.expander("⚙️ Model hyperparameters", expanded=False):
        mcols = st.columns(3)
        params = {}
        with mcols[0]:
            st.markdown("**KNN**")
            params["k"]       = st.slider("k (neighbors)", 1, 21, 5, 2)
            params["weights"] = st.selectbox("Weights", ["uniform","distance"])
        with mcols[1]:
            st.markdown("**Neural Network**")
            hl_str = st.text_input("Hidden layers (comma-sep neurons)", "64,32",
                                    help="e.g. 128,64,32 = three hidden layers")
            try:
                params["hidden_layers"] = [int(x.strip()) for x in hl_str.split(",") if x.strip()]
            except Exception:
                params["hidden_layers"] = [64, 32]
            params["activation"]   = st.selectbox("Activation", ["relu","tanh","logistic"])
            params["max_iter_nn"]  = st.slider("Max epochs", 100, 2000, 500, 100)
        with mcols[2]:
            st.markdown("**Decision Tree / RF**")
            params["max_depth"]       = st.slider("Max depth (0=unlimited)", 0, 20, 6)
            if params["max_depth"] == 0: params["max_depth"] = None
            params["min_samples_leaf"]= st.slider("Min samples / leaf", 1, 30, 4)
            params["n_estimators"]    = st.slider("RF: n trees", 50, 500, 100, 50)

        st.markdown("**Logistic Regression / SVM**")
        c1p, c2p = st.columns(2)
        with c1p:
            params["C"]     = st.select_slider("LR: Regularisation C", [0.001,0.01,0.1,1,10,100], value=1.0)
        with c2p:
            params["svm_C"] = st.select_slider("SVM: C", [0.01,0.1,1,10,100], value=1.0)
            params["kernel"]= st.selectbox("SVM Kernel", ["rbf","linear","poly"])

        if run_clustering:
            st.markdown("**K-Means**")
            params["k_clusters"] = st.slider("Number of clusters (K)", 2, 12, 4)

        if run_association:
            st.markdown("**Apriori**")
            c1a, c2a = st.columns(2)
            with c1a: params["min_support"]    = st.slider("Min support",    0.01, 0.30, 0.05, 0.01)
            with c2a: params["min_confidence"] = st.slider("Min confidence", 0.10, 0.90, 0.30, 0.05)

    # ── Cutoff ────────────────────────────────────────────────────
    if task == "Classification" and tgt:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""<div class="card card-yellow" style="font-size:.84rem">
        🎛️ <b>Classification threshold (cutoff):</b>
        The probability above which a sample is classified as positive.
        Default 0.5. Lower → more positives detected (higher Sensitivity, more False Positives).
        Set to <code>auto</code> (0) to sweep all cutoffs and show the trade-off chart.
        </div>""", unsafe_allow_html=True)
        cutoff_val = st.slider("Cutoff (0 = show sweep only)", 0.0, 0.95, 0.50, 0.02)
    else:
        cutoff_val = 0.5

    # ── Feature selection ─────────────────────────────────────────
    top_k_feat = st.slider(
        "Top-K correlated features to use (0 = all)",
        0, min(40, len(all_cols)), 0,
        help="Select only K features most correlated with target. 0 = use all."
    )

    # ── Run ───────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if not (selected_models or run_clustering or run_association):
        st.warning("Select at least one model / analysis type above.")
        st.stop()

    if st.button("▶  Run Analysis", type="primary", use_container_width=True):
        S["cfg"] = {
            "work_df":        work_df,
            "work_prof":      work_prof,
            "task":           task,
            "target":         tgt,
            "test_size":      test_size,
            "seed":           seed,
            "impute":         impute,
            "balance":        balance_method,
            "enc_overrides":  enc_overrides,
            "drop_cols":      drop_cols,
            "selected_models":selected_models,
            "run_clustering": run_clustering,
            "run_association":run_association,
            "params":         params,
            "cutoff":         cutoff_val if cutoff_val > 0 else 0.5,
            "sweep_cutoff":   cutoff_val == 0,
            "top_k_feat":     top_k_feat,
        }
        S["stage"] = "run"
        st.rerun()
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3 — RUN
# ═══════════════════════════════════════════════════════════════════════════
if S["stage"] == "run":
    cfg = S["cfg"]
    st.markdown("""
    <div class="step-header">
      <div class="step-num">⏳</div> Running analysis…
    </div>
    """, unsafe_allow_html=True)

    prog = st.progress(0)
    log_area = st.empty()
    log_lines = []

    def log(msg):
        log_lines.append(msg)
        log_area.markdown("\n\n".join(log_lines[-8:]))

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    R = {}  # results dict

    # ── Pre-split indices for leakage-free target encoding ────────
    _df_tmp = cfg["work_df"]
    _tgt_tmp = cfg["target"] if cfg["target"] else list(_df_tmp.columns)[-1]
    _y_tmp = _df_tmp[_tgt_tmp]
    try:
        _train_idx, _val_idx = train_test_split(
            np.arange(len(_df_tmp)), test_size=cfg["test_size"],
            random_state=cfg["seed"],
            stratify=_y_tmp if cfg["task"]=="Classification" and _y_tmp.nunique()<=20 else None
        )
    except Exception:
        _train_idx, _val_idx = train_test_split(
            np.arange(len(_df_tmp)), test_size=cfg["test_size"], random_state=cfg["seed"])

    # ── Preprocess ────────────────────────────────────────────────
    log("🔧 Preprocessing data…")
    prog.progress(5)

    X_df, y_ser, feat_names, enc_log, label_maps = smart_preprocess(
        cfg["work_df"], cfg["work_prof"],
        cfg["target"] if cfg["target"] else list(cfg["work_df"].columns)[-1],
        cfg["enc_overrides"], cfg["impute"],
        cfg["drop_cols"], cfg["task"],
        train_idx=_train_idx
    )

    R["enc_log"]    = enc_log
    R["label_maps"] = label_maps
    R["feat_names"] = feat_names
    R["X_df"]       = X_df
    R["y_ser"]      = y_ser

    for line in enc_log:
        log(line)

    # ── Feature selection ─────────────────────────────────────────
    prog.progress(12)
    top_k = cfg["top_k_feat"]
    if top_k > 0 and len(feat_names) > top_k:
        log(f"🔑 Selecting top-{top_k} features by correlation…")
        corr_with_y = X_df.corrwith(y_ser.astype(float)).abs().sort_values(ascending=False)
        selected_feats = list(corr_with_y.head(top_k).index)
        X_df = X_df[selected_feats]
        feat_names = selected_feats
        R["feat_names"] = feat_names
        R["corr_with_y"] = corr_with_y
        log(f"   Selected: {selected_feats[:6]}{'…' if len(selected_feats)>6 else ''}")
    else:
        R["corr_with_y"] = X_df.corrwith(y_ser.astype(float)).abs().sort_values(ascending=False)

    # ── Train / Val split ─────────────────────────────────────────
    prog.progress(18)
    X_arr = X_df.values.astype(float)
    y_arr = y_ser.values

    try:
        Xtr_r, Xval, ytr_r, yval = train_test_split(
            X_arr, y_arr, test_size=cfg["test_size"],
            random_state=cfg["seed"],
            stratify=y_arr if cfg["task"]=="Classification" and len(np.unique(y_arr))<=20 else None
        )
    except Exception:
        Xtr_r, Xval, ytr_r, yval = train_test_split(
            X_arr, y_arr, test_size=cfg["test_size"], random_state=cfg["seed"])

    R["Xval"] = Xval; R["yval"] = yval
    R["Xtr_raw"] = Xtr_r; R["ytr_raw"] = ytr_r
    log(f"✂ Train: {len(Xtr_r):,}  Validation: {len(Xval):,}  Features: {len(feat_names)}")

    # ── Balance ───────────────────────────────────────────────────
    prog.progress(22)
    Xtr, ytr, bal_log = balance_classes(Xtr_r, ytr_r, cfg["balance"], cfg["seed"])
    R["bal_log"] = bal_log
    for l in bal_log: log(l)

    # ── Scale ─────────────────────────────────────────────────────
    sc = StandardScaler()
    Xtr_s  = sc.fit_transform(Xtr)
    Xval_s = sc.transform(Xval)
    R["scaler"] = sc

    # ── Classification models ─────────────────────────────────────
    clf_results = {}
    if cfg["task"] == "Classification" and cfg["selected_models"]:
        n_models = len(cfg["selected_models"])
        for i, mname in enumerate(cfg["selected_models"]):
            prog.progress(25 + int(40 * i / n_models))
            log(f"🤖 Training {mname}…")
            # KNN & Neural Network & SVM use scaled data
            use_scaled = mname in ("KNN","Neural Network","SVM","Logistic Regression","LDA")
            Xt = Xtr_s if use_scaled else Xtr
            Xv = Xval_s if use_scaled else Xval
            try:
                clf = build_clf(mname, cfg["params"], cfg["seed"])

                # KNN: find best k if not locked
                if mname == "KNN":
                    best_k_acc = -1; best_k = cfg["params"].get("k",5)
                    for k_try in [3,5,7,9,11,15]:
                        from sklearn.neighbors import KNeighborsClassifier
                        tmp = KNeighborsClassifier(n_neighbors=k_try,
                                                    weights=cfg["params"].get("weights","uniform"))
                        tmp.fit(Xt, ytr)
                        acc = accuracy_score(yval, tmp.predict(Xv))
                        if acc > best_k_acc:
                            best_k_acc = acc; best_k = k_try
                    cfg["params"]["k"] = best_k
                    clf = build_clf("KNN", cfg["params"], cfg["seed"])
                    log(f"   KNN best k={best_k} (val acc={best_k_acc:.1%})")

                res = train_classify(clf, Xt, ytr, Xv, yval, cfg["cutoff"])
                clf_results[mname] = res
                log(f"   {mname}: val_acc={res['val_acc']:.1%}  sensitivity={res.get('sensitivity',0):.1%}")
            except Exception as e:
                log(f"   ⚠️ {mname} failed: {e}")

        R["clf_results"] = clf_results

    # ── Regression models ─────────────────────────────────────────
    reg_results = {}
    if cfg["task"] == "Regression" and cfg["selected_models"]:
        n_models = len(cfg["selected_models"])
        for i, mname in enumerate(cfg["selected_models"]):
            prog.progress(25 + int(40 * i / n_models))
            log(f"📈 Training {mname}…")
            use_scaled = mname in ("Neural Network","Logistic Regression","Ridge","Lasso","SVM")
            Xt = Xtr_s if use_scaled else Xtr
            Xv = Xval_s if use_scaled else Xval
            try:
                reg = build_reg(mname, cfg["params"], cfg["seed"])
                res = train_regress(reg, Xt, ytr.astype(float), Xv, yval.astype(float))
                reg_results[mname] = res
                log(f"   {mname}: R²={res['val_r2']:.3f}  MAE={res['mae']:.4g}")
            except Exception as e:
                log(f"   ⚠️ {mname} failed: {e}")
        R["reg_results"] = reg_results

    # ── Cutoff sweep ──────────────────────────────────────────────
    prog.progress(68)
    if cfg.get("sweep_cutoff") and clf_results:
        log("📈 Running cutoff sweep on best LR / Logistic model…")
        # Use logistic if available, else best model with proba
        sweep_model = clf_results.get("Logistic Regression") or next(
            (v for v in clf_results.values() if v.get("prob_pos") is not None), None)
        if sweep_model and sweep_model.get("prob_pos") is not None:
            cuts, accs, senss, specs = cutoff_sweep(sweep_model["prob_pos"], yval)
            R["cutoff_sweep"] = {"cuts": cuts, "accs": accs, "senss": senss, "specs": specs}

    # ── Clustering ────────────────────────────────────────────────
    prog.progress(72)
    if cfg["run_clustering"]:
        log("🔵 Running K-Means clustering…")
        try:
            from sklearn.cluster import KMeans
            k = cfg["params"].get("k_clusters", 4)
            km = KMeans(n_clusters=k, random_state=cfg["seed"], n_init=10)
            sc2 = StandardScaler()
            X_cl = sc2.fit_transform(X_arr[:min(5000, len(X_arr))])
            labels = km.fit_predict(X_cl)
            R["cluster_labels"]   = labels
            R["cluster_centers"]  = pd.DataFrame(
                sc2.inverse_transform(km.cluster_centers_), columns=feat_names)
            R["cluster_data"]     = pd.DataFrame(X_arr[:min(5000, len(X_arr))], columns=feat_names)
            R["cluster_data"]["Cluster"] = labels.astype(str)
            R["cluster_inertia"]  = float(km.inertia_)
            log(f"   K-Means (k={k}): inertia={km.inertia_:.2f}")

            # Elbow (quick)
            ks   = list(range(2, min(11, len(np.unique(y_arr))+5)))
            ines = []
            for ki in ks:
                km_i = KMeans(n_clusters=ki, random_state=cfg["seed"], n_init=5).fit(X_cl)
                ines.append(km_i.inertia_)
            R["elbow"] = {"ks": ks, "inertia": ines}
        except Exception as e:
            log(f"   ⚠️ Clustering failed: {e}")

    # ── Association Rules ─────────────────────────────────────────
    prog.progress(82)
    if cfg["run_association"]:
        log("🔗 Running Apriori association rules…")
        try:
            from mlxtend.frequent_patterns import apriori, association_rules
            from mlxtend.preprocessing import TransactionEncoder

            # Use categorical / binary columns from original df
            orig_df = cfg["work_df"].copy()
            cat_cols = [c for c, inf in cfg["work_prof"].items()
                        if inf.get("kind") in ("categorical","boolean") and inf["n_uniq"] <= 30
                        and c != cfg["target"]][:12]

            if cat_cols:
                transactions = orig_df[cat_cols].astype(str).values.tolist()
                te = TransactionEncoder()
                arr = te.fit_transform(transactions)
                df_te = pd.DataFrame(arr, columns=te.columns_)
                freq = apriori(df_te, min_support=cfg["params"]["min_support"], use_colnames=True)
                if not freq.empty:
                    rules = association_rules(freq, metric="confidence",
                                               min_threshold=cfg["params"]["min_confidence"])
                    rules = rules.sort_values("lift", ascending=False)
                    R["assoc_rules"] = rules
                    log(f"   Found {len(rules)} rules (min_support={cfg['params']['min_support']:.2f})")
                else:
                    log("   No frequent itemsets at current min_support — try lowering it")
            else:
                log("   No suitable categorical columns for association rules")
        except Exception as e:
            log(f"   ⚠️ Association rules failed: {e}")

    prog.progress(95)
    log("✅ All analyses complete.")
    prog.progress(100)

    S["results"] = R
    S["stage"]   = "results"
    import time; time.sleep(0.4)
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4 — RESULTS
# ═══════════════════════════════════════════════════════════════════════════
if S["stage"] == "results":
    R   = S["results"]
    cfg = S["cfg"]
    ctx = S["context"]

    st.markdown("""
    <div class="step-header">
      <div class="step-num">4</div> Results & Insights
    </div>
    """, unsafe_allow_html=True)

    task = cfg["task"]
    tgt  = cfg["target"]

    # ── Top KPIs ──────────────────────────────────────────────────
    kpi_cols = st.columns(5)

    clf_res = R.get("clf_results", {})
    reg_res = R.get("reg_results", {})

    if clf_res:
        best_clf = max(clf_res, key=lambda m: clf_res[m]["val_acc"])
        br       = clf_res[best_clf]
        kpi_cols[0].markdown(f"""<div class="metric-tile"><div class="m-label">Best model</div><div class="m-value" style="font-size:1.05rem">{best_clf}</div><div class="m-badge badge-blue">Accuracy {br['val_acc']:.1%}</div></div>""", unsafe_allow_html=True)
        kpi_cols[1].markdown(f"""<div class="metric-tile"><div class="m-label">Sensitivity</div><div class="m-value">{br.get('sensitivity',0):.1%}</div><div class="m-sub">TP/(TP+FN)</div></div>""", unsafe_allow_html=True)
        kpi_cols[2].markdown(f"""<div class="metric-tile"><div class="m-label">Specificity</div><div class="m-value">{br.get('specificity',0):.1%}</div><div class="m-sub">TN/(TN+FP)</div></div>""", unsafe_allow_html=True)
        kpi_cols[3].markdown(f"""<div class="metric-tile"><div class="m-label">F1-Score</div><div class="m-value">{br['f1']:.3f}</div><div class="m-sub">harmonic mean</div></div>""", unsafe_allow_html=True)
        kpi_cols[4].markdown(f"""<div class="metric-tile"><div class="m-label">Models trained</div><div class="m-value">{len(clf_res)}</div><div class="m-sub">val cutoff={cfg['cutoff']:.2f}</div></div>""", unsafe_allow_html=True)

    elif reg_res:
        best_reg = max(reg_res, key=lambda m: reg_res[m]["val_r2"])
        br_r     = reg_res[best_reg]
        kpi_cols[0].markdown(f"""<div class="metric-tile"><div class="m-label">Best model</div><div class="m-value" style="font-size:1rem">{best_reg}</div></div>""", unsafe_allow_html=True)
        kpi_cols[1].markdown(f"""<div class="metric-tile"><div class="m-label">R²</div><div class="m-value">{br_r['val_r2']:.4f}</div><div class="m-sub">variance explained</div></div>""", unsafe_allow_html=True)
        kpi_cols[2].markdown(f"""<div class="metric-tile"><div class="m-label">MAE</div><div class="m-value">{br_r['mae']:.4g}</div></div>""", unsafe_allow_html=True)
        kpi_cols[3].markdown(f"""<div class="metric-tile"><div class="m-label">RMSE</div><div class="m-value">{br_r['rmse']:.4g}</div></div>""", unsafe_allow_html=True)
        kpi_cols[4].markdown(f"""<div class="metric-tile"><div class="m-label">Models</div><div class="m-value">{len(reg_res)}</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Main tabs ─────────────────────────────────────────────────
    tab_names = []
    if clf_res: tab_names += ["📊 Model Comparison","🎯 Confusion Matrix","📈 Cutoff Analysis","🔑 Feature Importance"]
    if reg_res: tab_names += ["📈 Regression Results","🔑 Feature Importance"]
    if R.get("cluster_labels") is not None: tab_names.append("🔵 Clustering")
    if R.get("assoc_rules") is not None:    tab_names.append("🔗 Association Rules")
    tab_names += ["🧬 Data Profile","⚙️ Pipeline Log","🗺️ Methodology"]

    tabs = st.tabs(tab_names)
    tab_idx = 0

    # ── Tab: Model Comparison ─────────────────────────────────────
    if clf_res:
        with tabs[tab_idx]:
            tab_idx += 1
            rows = []
            for mname, res in clf_res.items():
                rows.append({
                    "Model":         mname,
                    "Train Acc":     res["train_acc"],
                    "Val Accuracy":  res["val_acc"],
                    "Sensitivity":   res.get("sensitivity",0),
                    "Specificity":   res.get("specificity",0),
                    "F1":            res["f1"],
                    "Precision":     res["precision"],
                })
            df_cmp = pd.DataFrame(rows).sort_values("Val Accuracy", ascending=False)

            # comparison bars
            fcols = st.columns(2)
            with fcols[0]:
                st.plotly_chart(compare_bar(df_cmp.rename(columns={"Val Accuracy":"Val Acc","Model":"Model"}),
                                             "Val Acc", "Validation Accuracy"),
                                use_container_width=True)
            with fcols[1]:
                st.plotly_chart(compare_bar(df_cmp.rename(columns={"Model":"Model"}),
                                             "Sensitivity", "Sensitivity (recall for positive class)"),
                                use_container_width=True)

            # Overfitting indicator
            df_cmp["Overfit Gap"] = df_cmp["Train Acc"] - df_cmp["Val Accuracy"]
            df_cmp["Overfit?"]    = df_cmp["Overfit Gap"].apply(
                lambda x: "⚠️ High" if x > 0.12 else ("✓ Moderate" if x > 0.05 else "✅ Low"))

            # Banner warning for suspicious perfect/near-perfect scores
            suspicious = df_cmp[df_cmp["Val Accuracy"] >= 0.99]
            if len(suspicious) > 0:
                models_list = ", ".join(suspicious["Model"].tolist())
                st.markdown(f"""<div class="card card-yellow" style="font-size:.88rem;margin:.6rem 0">
                ⚠️ <b>Suspicious accuracy alert:</b> {models_list} scored ≥99% on validation.
                This often indicates the dataset is very simple/small, the target leaks from a feature,
                or the same data was used for training and validation.
                Check the <b>Overfit?</b> column — a low gap alongside 100% usually means the task is trivially easy,
                not that the model is truly powerful. Consider using a held-out test set.
                </div>""", unsafe_allow_html=True)

            high_overfit = df_cmp[df_cmp["Overfit Gap"] > 0.12]
            if len(high_overfit) > 0:
                models_list2 = ", ".join(high_overfit["Model"].tolist())
                st.markdown(f"""<div class="card card-red" style="font-size:.88rem;margin:.4rem 0">
                🔴 <b>Overfitting detected:</b> {models_list2} show a Train vs Val gap &gt;12%.
                Consider limiting tree depth, increasing regularisation, or adding more training data.
                </div>""", unsafe_allow_html=True)

            st.dataframe(
                df_cmp[["Model","Train Acc","Val Accuracy","Sensitivity","Specificity","F1","Precision","Overfit?"]].style
                .format({"Train Acc":"{:.1%}","Val Accuracy":"{:.1%}","Sensitivity":"{:.1%}",
                          "Specificity":"{:.1%}","F1":"{:.3f}","Precision":"{:.1%}"})
                .background_gradient(subset=["Val Accuracy","Sensitivity"], cmap="Blues"),
                use_container_width=True
            )

            # Interactive: select model and show detail
            sel_model = st.selectbox("Inspect model:", list(clf_res.keys()), key="inspect_model")
            sel_res   = clf_res[sel_model]

            # Train vs Val bar chart
            fig_tv = go.Figure([
                go.Bar(name="Train", x=[sel_model], y=[sel_res["train_acc"]],
                       marker_color="#58a6ff"),
                go.Bar(name="Val",   x=[sel_model], y=[sel_res["val_acc"]],
                       marker_color="#3fb950"),
            ])
            fig_tv.update_layout(barmode="group", title=f"{sel_model}: Train vs Validation",
                                  yaxis_range=[0,1.05])
            st.plotly_chart(dark_fig(fig_tv, 300), use_container_width=True)

        # ── Tab: Confusion Matrix ────────────────────────────────
        with tabs[tab_idx]:
            tab_idx += 1
            sel_cm = st.selectbox("Model:", list(clf_res.keys()), key="cm_model")
            res_cm = clf_res[sel_cm]
            cm     = res_cm["cm"]
            n_cls  = cm.shape[0]

            # Class labels from label map
            target_map = R["label_maps"].get(f"__target__{tgt}", {})
            if target_map:
                inv_map = {v: k for k, v in target_map.items()}
                class_labels = [str(inv_map.get(i, i)) for i in range(n_cls)]
            else:
                class_labels = [str(i) for i in range(n_cls)]

            c1, c2 = st.columns([3,2])
            with c1:
                st.plotly_chart(cm_plotly(cm, f"Confusion Matrix — {sel_cm}", class_labels),
                                use_container_width=True)
            with c2:
                st.markdown(f"""
                <div class="card card-accent">
                <div class="tag tag-blue">Sensitivity</div><br>
                <b style="font-size:1.5rem">{res_cm.get('sensitivity',0):.1%}</b>
                <p style="color:#8b949e;font-size:.82rem">TP/(TP+FN) — share of actual positives correctly identified</p>
                </div>
                <div class="card card-green" style="margin-top:.5rem">
                <div class="tag tag-green">Specificity</div><br>
                <b style="font-size:1.5rem">{res_cm.get('specificity',0):.1%}</b>
                <p style="color:#8b949e;font-size:.82rem">TN/(TN+FP) — share of actual negatives correctly identified</p>
                </div>
                <div class="card card-yellow" style="margin-top:.5rem">
                <div class="tag tag-yellow">F1 Score</div><br>
                <b style="font-size:1.5rem">{res_cm['f1']:.3f}</b>
                <p style="color:#8b949e;font-size:.82rem">Harmonic mean of Precision and Recall</p>
                </div>
                """, unsafe_allow_html=True)

        # ── Tab: Cutoff Analysis ─────────────────────────────────
        with tabs[tab_idx]:
            tab_idx += 1
            # Interactive cutoff slider
            st.markdown("""<div class="card card-yellow" style="font-size:.84rem">
            Change the cutoff to explore the Sensitivity / Specificity trade-off in real time.
            Lower cutoff → higher Sensitivity (catch more positives) but more False Positives.
            </div>""", unsafe_allow_html=True)

            live_cutoff = st.slider("Live cutoff", 0.05, 0.95,
                                     cfg["cutoff"], 0.02, key="live_co")

            # Recompute metrics at live cutoff
            live_rows = []
            for mname, res in clf_res.items():
                if res.get("prob_pos") is not None:
                    from sklearn.metrics import accuracy_score, confusion_matrix
                    yp_live = (res["prob_pos"] >= live_cutoff).astype(int)
                    acc_l   = accuracy_score(R["yval"], yp_live)
                    cm_l    = confusion_matrix(R["yval"], yp_live)
                    if cm_l.size == 4:
                        tn,fp,fn,tp = cm_l.ravel()
                        sens_l = tp/(tp+fn) if (tp+fn)>0 else 0
                        spec_l = tn/(tn+fp) if (tn+fp)>0 else 0
                    else:
                        sens_l = spec_l = 0
                    live_rows.append({"Model":mname, "Accuracy":acc_l,
                                       "Sensitivity":sens_l,"Specificity":spec_l})

            if live_rows:
                df_live = pd.DataFrame(live_rows).sort_values("Accuracy", ascending=False)
                st.dataframe(df_live.style.format({
                    "Accuracy":"{:.1%}","Sensitivity":"{:.1%}","Specificity":"{:.1%}"
                }).background_gradient(subset=["Sensitivity"], cmap="RdYlGn"),
                use_container_width=True)

            if R.get("cutoff_sweep"):
                sw = R["cutoff_sweep"]
                # Find optimal cutoff per criterion
                opt_acc_idx  = int(np.argmax(sw["accs"]))
                opt_sens_idx = int(np.argmax(sw["senss"]))
                opt_f1_idx   = int(np.argmax([2*s*a/(s+a+1e-9) for s,a in zip(sw["senss"],sw["accs"])]))

                st.plotly_chart(
                    cutoff_chart(sw["cuts"], sw["accs"], sw["senss"], sw["specs"],
                                  live_cutoff, "Accuracy / Sensitivity / Specificity vs Cutoff"),
                    use_container_width=True
                )
                c1,c2,c3 = st.columns(3)
                with c1: st.markdown(f"""<div class="metric-tile"><div class="m-label">Best Accuracy</div><div class="m-value">{sw['accs'][opt_acc_idx]:.1%}</div><div class="m-sub">cutoff={sw['cuts'][opt_acc_idx]:.2f}</div></div>""", unsafe_allow_html=True)
                with c2: st.markdown(f"""<div class="metric-tile"><div class="m-label">Best Sensitivity</div><div class="m-value">{sw['senss'][opt_sens_idx]:.1%}</div><div class="m-sub">cutoff={sw['cuts'][opt_sens_idx]:.2f}</div></div>""", unsafe_allow_html=True)
                with c3: st.markdown(f"""<div class="metric-tile"><div class="m-label">Best F1 trade-off</div><div class="m-value">{sw['accs'][opt_f1_idx]:.1%} acc</div><div class="m-sub">cutoff={sw['cuts'][opt_f1_idx]:.2f}</div></div>""", unsafe_allow_html=True)
            else:
                st.info("Set cutoff = 0 on the config page to generate the full sweep chart.")

        # ── Tab: Feature Importance ──────────────────────────────
        with tabs[tab_idx]:
            tab_idx += 1
            fi_model = st.selectbox("Model:", list(clf_res.keys()), key="fi_model_clf")
            fi = feature_importances(clf_res[fi_model], R["feat_names"])
            if fi is not None:
                fi_sorted = fi.sort_values(ascending=False).head(20)
                fig_fi = px.bar(fi_sorted.reset_index(), x="index", y=0,
                                 labels={"index":"Feature","0":"Importance"},
                                 title=f"Feature importance — {fi_model}",
                                 color=0, color_continuous_scale="Teal")
                fig_fi.update_coloraxes(showscale=False)
                st.plotly_chart(dark_fig(fig_fi, 420), use_container_width=True)

                # Correlation chart
                if R.get("corr_with_y") is not None:
                    cwy = R["corr_with_y"].reindex(fi_sorted.index).dropna()
                    fig_co = px.bar(cwy.reset_index(), x="index", y=0,
                                     labels={"index":"Feature","0":"|Corr with target|"},
                                     title="Feature-target correlation",
                                     color=0, color_continuous_scale="Purples")
                    fig_co.update_coloraxes(showscale=False)
                    st.plotly_chart(dark_fig(fig_co, 380), use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")

    # ── Tab: Regression ───────────────────────────────────────────
    if reg_res:
        with tabs[tab_idx]:
            tab_idx += 1
            rows_r = [{"Model":m,"Train R²":r["train_r2"],"Val R²":r["val_r2"],
                        "MAE":r["mae"],"RMSE":r["rmse"]} for m,r in reg_res.items()]
            df_reg = pd.DataFrame(rows_r).sort_values("Val R²",ascending=False)
            st.dataframe(df_reg.style.format({"Train R²":"{:.4f}","Val R²":"{:.4f}",
                                               "MAE":"{:.4g}","RMSE":"{:.4g}"})
                          .background_gradient(subset=["Val R²"],cmap="Greens"),
                          use_container_width=True)

            sel_reg_m = st.selectbox("Inspect:", list(reg_res.keys()), key="reg_insp")
            rr = reg_res[sel_reg_m]
            st.plotly_chart(scatter_avp(R["yval"].astype(float), rr["yp_val"],
                                         f"Actual vs Predicted — {sel_reg_m}"),
                            use_container_width=True)

        with tabs[tab_idx]:
            tab_idx += 1
            fi_model_r = st.selectbox("Model:", list(reg_res.keys()), key="fi_model_reg")
            fi_r = feature_importances(reg_res[fi_model_r], R["feat_names"])
            if fi_r is not None:
                fi_rs = fi_r.sort_values(ascending=False).head(20)
                fig_fir = px.bar(fi_rs.reset_index(), x="index", y=0,
                                  labels={"index":"Feature","0":"Importance"},
                                  title=f"Feature importance — {fi_model_r}",
                                  color=0, color_continuous_scale="Teal")
                fig_fir.update_coloraxes(showscale=False)
                st.plotly_chart(dark_fig(fig_fir, 420), use_container_width=True)
            else:
                st.info("Feature importance not available.")

    # ── Tab: Clustering ───────────────────────────────────────────
    if R.get("cluster_labels") is not None:
        with tabs[tab_idx]:
            tab_idx += 1
            cl_data = R["cluster_data"]
            num_feats = [c for c in cl_data.columns if c != "Cluster"]

            c1, c2 = st.columns(2)
            with c1:
                xf = st.selectbox("X axis:", num_feats, index=0, key="cl_x")
            with c2:
                yf = st.selectbox("Y axis:", num_feats, index=min(1,len(num_feats)-1), key="cl_y")

            fig_cl = px.scatter(cl_data, x=xf, y=yf, color="Cluster",
                                 title=f"K-Means (k={cfg['params'].get('k_clusters',4)})",
                                 color_discrete_sequence=px.colors.qualitative.Bold,
                                 opacity=0.7)
            st.plotly_chart(dark_fig(fig_cl, 420), use_container_width=True)

            # Elbow chart
            if R.get("elbow"):
                el = R["elbow"]
                fig_elbow = px.line(x=el["ks"], y=el["inertia"],
                                     title="Elbow method — choose K at the 'elbow'",
                                     labels={"x":"K","y":"Inertia"}, markers=True,
                                     color_discrete_sequence=["#58a6ff"])
                st.plotly_chart(dark_fig(fig_elbow, 300), use_container_width=True)

            st.markdown("**Cluster centres (original scale)**")
            st.dataframe(R["cluster_centers"].style.background_gradient(cmap="Blues"),
                          use_container_width=True)

    # ── Tab: Association Rules ────────────────────────────────────
    if R.get("assoc_rules") is not None:
        with tabs[tab_idx]:
            tab_idx += 1
            rules = R["assoc_rules"]
            rules_disp = rules.copy()
            rules_disp["antecedents"] = rules_disp["antecedents"].apply(lambda x: ", ".join(list(x)))
            rules_disp["consequents"] = rules_disp["consequents"].apply(lambda x: ", ".join(list(x)))

            min_lift_filter = st.slider("Filter: min lift", 1.0, float(rules_disp["lift"].max()), 1.0, 0.1)
            filtered = rules_disp[rules_disp["lift"] >= min_lift_filter].head(30)

            st.dataframe(filtered[["antecedents","consequents","support","confidence","lift"]]
                          .rename(columns={"antecedents":"If…","consequents":"Then…"})
                          .style.background_gradient(subset=["lift","confidence"],cmap="Greens")
                          .format({"support":"{:.3f}","confidence":"{:.3f}","lift":"{:.3f}"}),
                          use_container_width=True)

            fig_rules = px.scatter(filtered, x="support", y="confidence",
                                    size="lift", color="lift",
                                    hover_data=["antecedents","consequents"],
                                    title="Support vs Confidence (size = Lift)",
                                    color_continuous_scale="Teal")
            st.plotly_chart(dark_fig(fig_rules, 380), use_container_width=True)

    # ── Tab: Data Profile ─────────────────────────────────────────
    with tabs[tab_idx]:
        tab_idx += 1
        work_df = cfg["work_df"]
        c1, c2 = st.columns(2)
        with c1:
            num_cols = work_df.select_dtypes("number").columns.tolist()[:15]
            if num_cols:
                n_c = min(3, len(num_cols)); n_r = (len(num_cols)+n_c-1)//n_c
                fig_d = make_subplots(rows=n_r, cols=n_c, subplot_titles=num_cols[:n_r*n_c])
                for i, col in enumerate(num_cols[:n_r*n_c]):
                    r, c = divmod(i, n_c)
                    fig_d.add_trace(go.Histogram(x=work_df[col].dropna(), name=col,
                                                  showlegend=False,
                                                  marker_color="#58a6ff", opacity=0.7),
                                    row=r+1, col=c+1)
                fig_d.update_layout(**DARK, height=250*n_r, title="Numeric distributions",
                                     margin=dict(l=30,r=20,t=45,b=30))
                st.plotly_chart(fig_d, use_container_width=True)

        with c2:
            num_c10 = work_df.select_dtypes("number").columns.tolist()[:10]
            if len(num_c10) > 1:
                st.plotly_chart(corr_heatmap(work_df[num_c10], "Correlation matrix"),
                                use_container_width=True)

        # Target distribution
        if tgt and tgt in work_df.columns:
            vc = work_df[tgt].value_counts().reset_index()
            vc.columns = ["Value","Count"]
            fig_tgt = px.bar(vc, x="Value", y="Count", title=f"Target distribution: {tgt}",
                              color="Count", color_continuous_scale="Blues")
            fig_tgt.update_coloraxes(showscale=False)
            st.plotly_chart(dark_fig(fig_tgt, 300), use_container_width=True)

    # ── Tab: Pipeline Log ─────────────────────────────────────────
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("**Preprocessing steps**")
        for line in R.get("enc_log", []):
            st.markdown(f"<div class='card' style='font-size:.82rem;padding:.5rem .8rem'>{line}</div>", unsafe_allow_html=True)
        st.markdown("**Balancing**")
        for line in R.get("bal_log", []):
            st.markdown(f"<div class='card card-green' style='font-size:.82rem;padding:.5rem .8rem'>{line}</div>", unsafe_allow_html=True)
        st.markdown("**Label maps**")
        for k, v in R.get("label_maps", {}).items():
            st.markdown(f"`{k}`: {v}")

    # ── Tab: Methodology Flowcharts ───────────────────────────────
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown("""<div class="card card-accent" style="margin-bottom:1rem;font-size:.93rem">
        Each flowchart below illustrates how a method works — from raw data to final output.
        These are general explanations independent of the current dataset.
        </div>""", unsafe_allow_html=True)

        meth_tabs = st.tabs(["📐 Decision Tree","📊 Logistic Regression","🧠 Neural Network",
                              "📏 LDA","🌀 K-Means","🔗 Association Rules",
                              "🌲 Random Forest","🎯 Naïve Bayes"])

        # ── Decision Tree ────────────────────────────────────────
        with meth_tabs[0]:
            st.markdown("""### Decision Tree — How it works""")
            st.markdown("""
A Decision Tree learns a hierarchy of **if/else rules** on features to split data into pure groups.

**Key concepts:**
- **Gini impurity / Entropy** — measures how mixed a node's classes are; the algorithm picks the split that reduces this most
- **Max depth** — limits how many splits deep the tree can go (prevents overfitting — default here is **6**)
- **Min samples per leaf** — a node must have at least this many samples to be a leaf (prevents over-specific rules)
- **No feature scaling needed** — trees are invariant to monotonic transformations
            """)
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Cart_tree_kyphosis.png/440px-Cart_tree_kyphosis.png",
                     caption="Example decision tree structure", width=420)
            st.markdown("""
```
Raw Data
   │
   ▼
[Pick best feature & threshold]  ← minimises Gini / Entropy
   │
   ├─ Feature ≤ threshold ──► Left subtree (recurse)
   │
   └─ Feature > threshold ──► Right subtree (recurse)
                                    │
                              [Leaf node: majority class]
```
**Overfitting risk:** Without depth limits, a tree can memorise every training row (100% train acc, poor val acc).
This app caps `max_depth=6` by default. Increase with caution.
            """)

        # ── Logistic Regression ──────────────────────────────────
        with meth_tabs[1]:
            st.markdown("""### Logistic Regression — How it works""")
            st.markdown("""
Logistic Regression fits a **linear decision boundary** and outputs a **probability** via the sigmoid function.

**Key concepts:**
- **Log-odds / Logit** — the model predicts `log(p/(1-p)) = w₀ + w₁x₁ + … + wₙxₙ`
- **Sigmoid** — converts log-odds to a probability between 0 and 1
- **Cutoff** — you choose the threshold above which a probability is labelled "positive"
- **Regularisation (C)** — smaller C = stronger penalty on large weights = simpler model

**Flowchart:**
```
Features (scaled)
   │
   ▼
Linear combination:  z = w·x + b
   │
   ▼
Sigmoid:  P(y=1) = 1 / (1 + e^{-z})
   │
   ▼
P(y=1) ≥ cutoff ?  ──► Positive class
                   No ──► Negative class
```
**Assumes:** Linear separability; requires feature scaling (done automatically).
            """)

        # ── Neural Network ───────────────────────────────────────
        with meth_tabs[2]:
            st.markdown("""### Neural Network (MLP) — How it works""")
            st.markdown("""
A Multilayer Perceptron (MLP) learns **non-linear** patterns by stacking layers of weighted sums and activations.

**Key concepts:**
- **Hidden layers** — intermediate transformations; default here is `[64, 32]` neurons
- **Activation (ReLU)** — `max(0, x)` introduces non-linearity so the network can model complex boundaries
- **Backpropagation** — error is propagated backwards to update weights via gradient descent
- **Alpha (L2 regularisation)** — penalises large weights to prevent overfitting

**Flowchart:**
```
Input features (scaled)
   │
   ▼
Hidden Layer 1 (64 neurons) → ReLU activation
   │
   ▼
Hidden Layer 2 (32 neurons) → ReLU activation
   │
   ▼
Output layer → Softmax (multiclass) / Sigmoid (binary)
   │
   ▼
Predicted class / probability
```
**Watch out for:** Needs many iterations (epochs) to converge; sensitive to feature scale.
            """)

        # ── LDA ──────────────────────────────────────────────────
        with meth_tabs[3]:
            st.markdown("""### Linear Discriminant Analysis (LDA) — How it works""")
            st.markdown("""
LDA finds a **linear projection** of features that maximises the separation between classes while minimising spread within classes.

**Key concepts:**
- **Between-class scatter** — how far apart are the class means?
- **Within-class scatter** — how spread out is each class?
- LDA maximises the ratio: Between-class / Within-class
- Also works as **dimensionality reduction** — projects to at most `n_classes - 1` dimensions

**Flowchart:**
```
Compute class means (μ₁, μ₂, …)
   │
   ▼
Compute within-class scatter matrix Sw
   │
   ▼
Compute between-class scatter matrix Sb
   │
   ▼
Find projection W = argmax |Sb| / |Sw|
   │
   ▼
Project data: z = W^T · x
   │
   ▼
Assign to nearest class centroid in projected space
```
**Assumes:** Normally distributed features; equal covariance per class. Fast and interpretable.
            """)

        # ── K-Means ──────────────────────────────────────────────
        with meth_tabs[4]:
            st.markdown("""### K-Means Clustering — How it works""")
            st.markdown("""
K-Means partitions data into **K groups** by iteratively assigning points to the nearest centroid and updating centroids.

**Key concepts:**
- **Centroid** — the mean position of all points in a cluster
- **Inertia** — total within-cluster sum of squared distances (lower = tighter clusters)
- **Elbow method** — plot inertia vs K; pick the K where improvement flattens
- **K is set by you** — unlike classification, there is no "correct" K; domain knowledge helps

**Flowchart:**
```
Choose K
   │
   ▼
Initialise K centroids (K-Means++ method)
   │
   ▼
┌──────────────────────────────────────┐
│  Assign each point to nearest        │
│  centroid (Euclidean distance)       │
│        │                             │
│  Update centroid = mean of cluster   │
│        │                             │
│  Converged? ──No──► repeat           │
└──────────────────────────────────────┘
   │ Yes
   ▼
Final cluster assignments + centres
```
**Requires feature scaling** (done automatically). Sensitive to outliers.
            """)

        # ── Association Rules ────────────────────────────────────
        with meth_tabs[5]:
            st.markdown("""### Association Rules (Apriori) — How it works""")
            st.markdown("""
Association rule mining finds **if → then patterns** in transactional data (e.g. "customers who buy X also buy Y").

**Key metrics:**
| Metric | Formula | Meaning |
|---|---|---|
| **Support** | freq(X∪Y) / N | How often does {X,Y} appear together? |
| **Confidence** | freq(X∪Y) / freq(X) | Given X, how often does Y appear? |
| **Lift** | Confidence / P(Y) | Is the rule better than random? Lift > 1 = yes |

**Flowchart:**
```
Transaction data (binary columns)
   │
   ▼
Find frequent itemsets with Support ≥ min_support
(Apriori: prune any superset of an infrequent set)
   │
   ▼
Generate candidate rules from frequent itemsets
   │
   ▼
Filter rules with Confidence ≥ min_confidence
   │
   ▼
Rank by Lift → display top rules
```
**Lift > 1** means the items co-occur more than by chance — actionable patterns.
            """)

        # ── Random Forest ────────────────────────────────────────
        with meth_tabs[6]:
            st.markdown("""### Random Forest — How it works""")
            st.markdown("""
Random Forest is an **ensemble** of many Decision Trees, each trained on a random subset of data and features.
The final prediction is a **majority vote** (classification) or **mean** (regression).

**Key concepts:**
- **Bagging (Bootstrap Aggregating)** — each tree sees a random sample with replacement
- **Feature randomness** — at each split, only a random subset of features is considered
- **Variance reduction** — averaging many weak learners cancels out individual errors
- **Feature importance** — average impurity decrease across all trees per feature

**Flowchart:**
```
For each of N trees:
  ├─ Sample rows with replacement (bootstrap)
  ├─ At each split: sample √p features randomly
  └─ Grow full tree (no pruning needed — diversity handles it)

Prediction:
  Input ──► Tree₁ ──► vote₁ ─┐
  Input ──► Tree₂ ──► vote₂ ─┼──► Majority vote ──► Final class
  Input ──► TreeN ──► voteN ─┘
```
**More robust to overfitting** than a single tree. Slower but usually more accurate.
            """)

        # ── Naïve Bayes ──────────────────────────────────────────
        with meth_tabs[7]:
            st.markdown("""### Naïve Bayes — How it works""")
            st.markdown("""
Naïve Bayes applies **Bayes' theorem** with the "naïve" assumption that features are conditionally independent given the class.

**Bayes' theorem:**
```
P(class | features) ∝ P(class) × P(x₁|class) × P(x₂|class) × … × P(xₙ|class)
```

**Key concepts:**
- **Prior P(class)** — how frequent is each class in training data?
- **Likelihood P(xᵢ|class)** — for Gaussian NB: modelled as a normal distribution per feature per class
- **Posterior** — multiply prior × all likelihoods; assign the class with highest posterior
- **Var smoothing** — adds a tiny variance floor to prevent zero-probability issues

**Flowchart:**
```
Training:
  For each class:
    Estimate mean and variance of each feature
    Estimate class prior

Prediction:
  For each class:
    Compute log P(class) + Σ log P(xᵢ | class)  ← sum of log-likelihoods
  
  Assign class with highest score
```
**Strengths:** Very fast; works well on high-dimensional data; good baseline.
**Weakness:** Independence assumption rarely holds perfectly in practice.
            """)

    # ── Re-run with different params ──────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    with st.expander("🔄 Adjust parameters & re-run"):
        st.markdown("""<div class="card" style="font-size:.83rem">
        Change any parameter below and click <b>Re-run</b> to generate a new scenario.
        Results will replace the current view.
        </div>""", unsafe_allow_html=True)

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            new_cutoff = st.slider("New cutoff", 0.05, 0.95, cfg["cutoff"], 0.02, key="rer_co")
        with rc2:
            new_balance = st.selectbox("Balancing", ["none","oversample_random","smote","undersample"],
                                        index=["none","oversample_random","smote","undersample"].index(cfg["balance"]),
                                        key="rer_bal")
        with rc3:
            new_topk = st.slider("Top-K features (0=all)", 0, min(40, len(R.get("feat_names",[]))), cfg["top_k_feat"], key="rer_topk")

        # Neural network hidden layers
        new_hl = st.text_input("Neural Network hidden layers", ",".join(str(x) for x in cfg["params"].get("hidden_layers",[64,32])), key="rer_hl")
        new_k_clf = st.slider("KNN: k", 1, 21, cfg["params"].get("k",5), 2, key="rer_k")
        new_max_depth = st.slider("Tree max depth (0=unlimited)", 0, 20, cfg["params"].get("max_depth") or 0, key="rer_md")

        if st.button("▶ Re-run with these settings", type="primary", key="rerun_btn"):
            try:
                hl = [int(x.strip()) for x in new_hl.split(",") if x.strip()]
            except Exception:
                hl = [64, 32]
            S["cfg"]["cutoff"]       = new_cutoff
            S["cfg"]["sweep_cutoff"] = False
            S["cfg"]["balance"]      = new_balance
            S["cfg"]["top_k_feat"]   = new_topk
            S["cfg"]["params"]["hidden_layers"] = hl
            S["cfg"]["params"]["k"] = new_k_clf
            S["cfg"]["params"]["max_depth"] = new_max_depth if new_max_depth > 0 else None
            S["stage"] = "run"
            st.rerun()
