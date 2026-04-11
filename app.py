import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import re as _re
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataMine AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }
.main  { background: #0d1117; }
.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); color: #e6edf3; }
h1,h2,h3 { font-family: 'Space Mono', monospace; color: #e6edf3; }

[data-testid="stMetricValue"] { color: #e6edf3 !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; }
.stDataFrame td, .stDataFrame th { color: #e6edf3 !important; background: #161b22 !important; }
.stSelectbox div[data-baseweb="select"] { background: #161b22 !important; color: #e6edf3 !important; }
.stMultiSelect div[data-baseweb="select"] { background: #161b22 !important; color: #e6edf3 !important; }
div[data-baseweb="option"]  { background: #161b22 !important; color: #e6edf3 !important; }
div[data-baseweb="popover"] { background: #161b22 !important; }
.stTextInput input, .stTextArea textarea { color: #e6edf3 !important; background: #161b22 !important; }
.stInfo, .stSuccess, .stWarning, .stError { color: #e6edf3 !important; }
button[data-baseweb="tab"]                   { color: #8b949e !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #58a6ff !important; }
div[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #30363d; }

.hero-title {
    font-family:'Space Mono',monospace; font-size:2.4rem; font-weight:700;
    background:linear-gradient(90deg,#58a6ff,#bc8cff,#f778ba);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1.2;
}
.hero-sub { color:#8b949e; font-size:1rem; margin-bottom:1.5rem; }

/* Step pill nav */
.step-nav {
    display:flex; gap:8px; margin-bottom:1.5rem; flex-wrap:wrap;
}
.step-pill {
    padding:6px 16px; border-radius:20px; font-size:0.8rem;
    font-family:'Space Mono',monospace; font-weight:700; border:1px solid #30363d;
    background:#161b22; color:#8b949e; cursor:default;
}
.step-pill.active  { background:#1f3a5f; color:#58a6ff; border-color:#58a6ff; }
.step-pill.done    { background:#1a3a2a; color:#3fb950; border-color:#3fb950; }

.section-hdr {
    font-family:'Space Mono',monospace; font-size:0.72rem; text-transform:uppercase;
    letter-spacing:2px; color:#58a6ff; margin-bottom:0.8rem;
    border-bottom:1px solid #21262d; padding-bottom:0.5rem;
}
.card { background:#161b22; border:1px solid #30363d; border-radius:12px; padding:1.2rem 1.4rem; margin-bottom:1rem; }
.card-accent { border-left:4px solid #58a6ff; }

.badge { display:inline-block; padding:2px 10px; border-radius:20px; font-size:0.72rem;
         font-weight:600; font-family:'Space Mono',monospace; margin-right:4px; }
.badge-blue   { background:#1f3a5f; color:#58a6ff; }
.badge-purple { background:#2d1f5f; color:#bc8cff; }
.badge-pink   { background:#3d1f35; color:#f778ba; }
.badge-green  { background:#1a3a2a; color:#3fb950; }
.badge-yellow { background:#3a2d10; color:#d29922; }

.ai-box {
    background:#1c2333; border:1px solid #58a6ff; border-radius:10px;
    padding:1.1rem 1.3rem; margin-bottom:0.8rem; color:#ffffff !important;
}
.ai-box * { color:#ffffff !important; }
.ai-text {
    color:#e6edf3 !important; font-size:0.89rem; line-height:1.75;
    white-space:pre-wrap; word-break:break-word;
    font-family:'DM Sans',sans-serif;
}

.prob-row {
    display:flex; align-items:flex-start; gap:10px; padding:7px 0;
    border-bottom:1px solid #21262d; color:#e6edf3; font-size:0.85rem;
}
.prob-row:last-child { border-bottom:none; }
.chip { border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:700; }
.chip-err  { background:#3d1f1f; color:#f85149; }
.chip-warn { background:#3a2d10; color:#d29922; }
.chip-info { background:#1f3a5f; color:#58a6ff; }
.chip-ok   { background:#1a3a2a; color:#3fb950; }

.method-btn-selected { border:2px solid #58a6ff !important; background:#1a2332 !important; }
.method-btn-compare  { border:2px solid #3fb950 !important; background:#1a3a2a !important; }

.stButton>button {
    background:linear-gradient(90deg,#238636,#2ea043); color:white; border:none;
    border-radius:8px; font-family:'Space Mono',monospace; font-weight:700;
    padding:0.5rem 1.2rem; transition:opacity 0.2s;
}
.stButton>button:hover { opacity:0.85; }
.stSelectbox label, .stMultiselect label, .stTextArea label {
    color:#8b949e !important; font-size:0.85rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Imports ───────────────────────────────────────────────────────────────────
import google.generativeai as genai
import requests as _requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_squared_error, r2_score,
                             silhouette_score, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# ── AI config ─────────────────────────────────────────────────────────────────
_DEFAULT_GEMINI_KEY = ""
_GEMINI_CANDIDATES  = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
_OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_MODELS  = [
    "openrouter/auto",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1:free",
    "qwen/qwq-32b:free",
]

# ── Method catalogue ──────────────────────────────────────────────────────────
METHODS = {
    "Logistic Regression":                {"group":"classification","badge":"badge-blue",
        "vn":"Phân loại cơ bản",
        "desc":"Classifies data into binary or multi-class categories using a probabilistic S-curve. Interpretable coefficients. Great starting point."},
    "Linear Discriminant Analysis (LDA)": {"group":"classification","badge":"badge-blue",
        "vn":"Phân tích biệt thức",
        "desc":"Projects data onto axes that maximise class separation. Works best when classes have roughly equal covariance."},
    "K-Nearest Neighbors (KNN)":          {"group":"classification","badge":"badge-blue",
        "vn":"Phân loại láng giềng",
        "desc":"Assigns a label based on the majority class among the K closest training samples. Simple and interpretable."},
    "Classification Trees":               {"group":"classification","badge":"badge-blue",
        "vn":"Cây quyết định",
        "desc":"Builds human-readable IF-THEN rules to split data into classes. Highly interpretable output."},
    "Naive Bayes":                        {"group":"classification","badge":"badge-blue",
        "vn":"Dự báo xác suất",
        "desc":"Applies Bayes' theorem with naive independence assumption. Very fast. Strong on text and sparse data."},
    "Support Vector Machine (SVM)":       {"group":"classification","badge":"badge-blue",
        "vn":"Phân loại biên giới",
        "desc":"Finds the hyperplane with maximum margin between classes. Powerful in high-dimensional spaces."},
    "Random Forest":                      {"group":"classification","badge":"badge-blue",
        "vn":"Rừng ngẫu nhiên",
        "desc":"Ensemble of 100 decision trees on random subsets. Robust, accurate, provides feature importance."},
    "Neural Networks (MLP)":              {"group":"classification","badge":"badge-blue",
        "vn":"Mạng nơ-ron",
        "desc":"Multi-layer perceptron that learns complex non-linear patterns. Suitable for large complex datasets."},
    "Linear Regression":                  {"group":"prediction","badge":"badge-purple",
        "vn":"Hồi quy tuyến tính",
        "desc":"Models the linear relationship between features and a continuous target. Interpretable coefficients."},
    "Neural Networks Regression (MLP)":   {"group":"prediction","badge":"badge-purple",
        "vn":"Mạng nơ-ron hồi quy",
        "desc":"MLP applied to regression — predicts continuous values through stacked non-linear transformations."},
    "Association Rules (Apriori)":        {"group":"association","badge":"badge-pink",
        "vn":"Luật kết hợp",
        "desc":"Discovers IF-THEN patterns in transactional data. Uses support, confidence, lift metrics."},
    "K-Means Clustering":                 {"group":"association","badge":"badge-green",
        "vn":"Phân cụm K-Means",
        "desc":"Groups data into K clusters by minimising intra-cluster variance. Good for segmentation."},
    "Hierarchical Clustering":            {"group":"association","badge":"badge-green",
        "vn":"Phân cụm phân cấp",
        "desc":"Builds a dendrogram of nested clusters without specifying K upfront. Reveals natural hierarchy."},
    "Random Oversampling":                {"group":"balancing","badge":"badge-yellow",
        "vn":"Cân bằng ngẫu nhiên",
        "desc":"Replicates minority-class samples randomly to fix class imbalance before classification."},
    "SMOTE":                              {"group":"balancing","badge":"badge-yellow",
        "vn":"Cân bằng tổng hợp",
        "desc":"Generates synthetic minority samples by interpolating between existing ones. Richer than oversampling."},
}
GROUP_META = {
    "classification": {"label":"Classification / Phân loại",       "color":"#58a6ff","icon":"🔵"},
    "prediction":     {"label":"Prediction / Dự báo",               "color":"#bc8cff","icon":"🟣"},
    "association":    {"label":"Association & Clustering / Kết hợp","color":"#f778ba","icon":"🔴"},
    "balancing":      {"label":"Class Balancing / Cân bằng lớp",    "color":"#d29922","icon":"🟡"},
}

# ═════════════════════════════════════════════════════════════════════════════
# BACKEND HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _df_to_json(df: pd.DataFrame) -> str:
    buf = io.StringIO(); df.to_json(buf, orient="split"); return buf.getvalue()

def _json_to_df(s: str) -> pd.DataFrame:
    return pd.read_json(io.StringIO(s), orient="split")

def load_file(uploaded) -> dict[str, pd.DataFrame]:
    name = uploaded.name.lower(); sheets = {}
    if name.endswith(".csv"):
        sheets["Sheet1"] = pd.read_csv(uploaded)
    elif name.endswith((".xlsx", ".xls")):
        xf = pd.ExcelFile(uploaded)
        for s in xf.sheet_names:
            sheets[s] = xf.parse(s)
    elif name.endswith(".json"):
        data = json.load(uploaded)
        sheets["Sheet1"] = pd.DataFrame(data if isinstance(data, list) else [data])
    elif name.endswith(".txt"):
        sheets["Sheet1"] = pd.read_csv(uploaded, sep=None, engine="python")
    else:
        st.error("Unsupported file type.")
    return sheets

def df_summary(df: pd.DataFrame) -> str:
    return (f"Shape: {df.shape}\nColumns: {list(df.columns)}\n"
            f"Dtypes:\n{df.dtypes.to_string()}\n"
            f"Null counts:\n{df.isnull().sum().to_string()}\n"
            f"Sample (3 rows):\n{df.head(3).to_string()}\n"
            f"Describe:\n{df.describe(include='all').to_string()}")

def encode_df(df: pd.DataFrame):
    df = df.copy(); le = LabelEncoder()
    for col in df.select_dtypes(include=["object","string"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def fig_to_st(fig):
    st.pyplot(fig); plt.close(fig)

def _clean_ai(text: str) -> str:
    t = _re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    t = _re.sub(r"^#+\s*", "", t, flags=_re.MULTILINE)
    t = _re.sub(r"^[-•]\s+", "  ", t, flags=_re.MULTILINE)
    t = _re.sub(r"`{1,3}", "", t)
    t = _re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _safe_html(s: str) -> str:
    return s.replace("<", "&lt;").replace(">", "&gt;")

def render_ai_box(text: str, label: str = "🤖 AI Analysis"):
    clean = _clean_ai(text)
    st.markdown(
        f'<div class="ai-box"><b style="color:#58a6ff">{label}</b><br><br>'
        f'<div class="ai-text">{_safe_html(clean)}</div></div>',
        unsafe_allow_html=True,
    )

# ── AI calls ──────────────────────────────────────────────────────────────────
def _get_keys():
    g = st.session_state.get("gemini_key","").strip()
    o = st.session_state.get("openrouter_key","").strip()
    return g or _DEFAULT_GEMINI_KEY, o

def _call_openrouter(prompt, or_key):
    if not or_key: return ""
    headers = {"Authorization": f"Bearer {or_key}","Content-Type":"application/json",
               "HTTP-Referer":"https://datamine-ai.streamlit.app","X-Title":"DataMine AI"}
    for model in _OPENROUTER_MODELS:
        try:
            resp = _requests.post(_OPENROUTER_URL, headers=headers,
                json={"model":model,"messages":[{"role":"user","content":prompt}],
                      "max_tokens":2000,"temperature":0.7}, timeout=120)
            if resp.status_code == 401: return "__OR_FAIL__: Invalid key."
            data = resp.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
        except Exception:
            continue
    return ""

def ask_ai(prompt: str) -> str:
    gemini_key, or_key = _get_keys()
    for model_name in _GEMINI_CANDIDATES:
        try:
            genai.configure(api_key=gemini_key)
            mdl = genai.GenerativeModel(model_name)
            resp = mdl.generate_content(prompt)
            return resp.text
        except Exception:
            continue
    if or_key:
        result = _call_openrouter(prompt, or_key)
        if result and not result.startswith("__OR_FAIL__"):
            return result
    return ("⚠️ No AI key set or all models failed. "
            "Paste a Gemini key (aistudio.google.com) or OpenRouter key (openrouter.ai) in the sidebar.")

# ═════════════════════════════════════════════════════════════════════════════
# PREPROCESSING (cached)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def detect_problems(df_json: str) -> list[dict]:
    df = _json_to_df(df_json); n = len(df); problems = []
    for col in df.columns:
        pct = df[col].isnull().mean()
        if pct > 0:
            dtype = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
            problems.append({"col":col,"type":"missing","sev":"err" if pct>0.3 else "warn",
                "msg":f'Column "{col}" — {pct:.1%} missing ({int(pct*n)} rows)',
                "fix":f"Fill with {'mean' if dtype=='numeric' else 'mode'}","dtype":dtype})
    for col in df.select_dtypes(include=[np.number]).columns:
        q1,q3 = df[col].quantile(0.25),df[col].quantile(0.75); iqr = q3-q1
        if iqr>0:
            n_out = ((df[col]<q1-3*iqr)|(df[col]>q3+3*iqr)).sum()
            if n_out>0:
                problems.append({"col":col,"type":"outlier","sev":"warn",
                    "msg":f'Column "{col}" — {n_out} extreme outliers ({n_out/n:.1%})',
                    "fix":"Cap to 3×IQR (Winsorize)","dtype":"numeric"})
    for col in df.select_dtypes(include=["object","string"]).columns:
        problems.append({"col":col,"type":"encoding","sev":"info",
            "msg":f'Column "{col}" is text — needs encoding for ML',
            "fix":"Label Encoding","dtype":"categorical"})
    dups = df.duplicated().sum()
    if dups>0:
        problems.append({"col":"ALL","type":"duplicate","sev":"warn",
            "msg":f"{dups} duplicate rows ({dups/n:.1%})","fix":"Remove duplicates","dtype":"row"})
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols)>1:
        rng = df[num_cols].max()-df[num_cols].min()
        if rng.max()>0 and (rng.max()/(rng.min()+1e-9))>100:
            problems.append({"col":"NUMERIC","type":"scale","sev":"info",
                "msg":"Numeric columns have very different scales — scaling recommended",
                "fix":"StandardScaler (auto-applied at training)","dtype":"numeric"})
    if not problems:
        problems.append({"col":"","type":"ok","sev":"ok",
            "msg":"No major data problems detected. Dataset looks clean!","fix":"","dtype":""})
    return problems

@st.cache_data(show_spinner=False)
def fix_missing(df_json):
    df = _json_to_df(df_json); before = df.isnull().sum().sum()
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                m = df[col].mode(); df[col] = df[col].fillna(m[0] if len(m) else "Unknown")
    return _df_to_json(df), f"Fixed {before - df.isnull().sum().sum()} missing values."

@st.cache_data(show_spinner=False)
def fix_duplicates(df_json):
    df = _json_to_df(df_json); before = len(df); df = df.drop_duplicates()
    return _df_to_json(df), f"Removed {before-len(df)} duplicates. {len(df):,} rows remain."

@st.cache_data(show_spinner=False)
def fix_outliers(df_json):
    df = _json_to_df(df_json); capped = 0
    for col in df.select_dtypes(include=[np.number]).columns:
        q1,q3 = df[col].quantile(0.25),df[col].quantile(0.75); iqr=q3-q1
        if iqr>0:
            lo,hi = q1-3*iqr,q3+3*iqr
            capped += ((df[col]<lo)|(df[col]>hi)).sum()
            df[col] = df[col].clip(lower=lo,upper=hi)
    return _df_to_json(df), f"Capped {capped} extreme outlier values (3×IQR)."

@st.cache_data(show_spinner=False)
def fix_encode(df_json):
    df = _json_to_df(df_json)
    text_cols = df.select_dtypes(include=["object","string"]).columns.tolist()
    if not text_cols: return _df_to_json(df), "No text columns to encode.", {}
    mapping = {}; le = LabelEncoder()
    for col in text_cols:
        le.fit(df[col].astype(str))
        mapping[col] = {str(c):int(i) for i,c in enumerate(le.classes_)}
        df[col] = le.transform(df[col].astype(str))
    msg = f"Label-encoded {len(text_cols)} column(s): {', '.join(text_cols)}."
    return _df_to_json(df), msg, mapping

def suggest_target(df: pd.DataFrame) -> str | None:
    kws = ["label","target","class","churn","default","outcome","result",
           "status","output","dependent","response","predict","category","nhãn"]
    for col in df.columns:
        if any(k in col.lower().replace(" ","_") for k in kws):
            return col
    last = df.columns[-1]
    return last if df[last].nunique() <= 20 else None

# ═════════════════════════════════════════════════════════════════════════════
# ML RUNNERS  (return metrics dict for comparison table + export)
# ═════════════════════════════════════════════════════════════════════════════

def run_classification(method, df, target, features, test_size, balance):
    df_enc = encode_df(df[features+[target]].dropna())
    X = StandardScaler().fit_transform(df_enc[features].values)
    y = df_enc[target].values
    if balance == "Random Oversampling":
        X, y = RandomOverSampler(random_state=42).fit_resample(X, y)
        st.info("✅ Applied Random Oversampling.")
    elif balance == "SMOTE":
        try: X, y = SMOTE(random_state=42).fit_resample(X, y); st.info("✅ Applied SMOTE.")
        except Exception as e: st.warning(f"SMOTE skipped: {e}")
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=test_size,random_state=42)
    models = {
        "Logistic Regression":               LogisticRegression(max_iter=1000),
        "Linear Discriminant Analysis (LDA)":LinearDiscriminantAnalysis(),
        "K-Nearest Neighbors (KNN)":         KNeighborsClassifier(),
        "Classification Trees":              DecisionTreeClassifier(max_depth=5),
        "Naive Bayes":                       GaussianNB(),
        "Support Vector Machine (SVM)":      SVC(probability=True),
        "Random Forest":                     RandomForestClassifier(n_estimators=100,random_state=42),
        "Neural Networks (MLP)":             MLPClassifier(max_iter=500,random_state=42),
    }
    mdl = models[method]; mdl.fit(X_tr,y_tr); y_pred = mdl.predict(X_te)
    acc = accuracy_score(y_te,y_pred)
    is_bin = len(np.unique(y))==2; avg = "binary" if is_bin else "weighted"
    prec = precision_score(y_te,y_pred,average=avg,zero_division=0)
    rec  = recall_score(y_te,y_pred,average=avg,zero_division=0)
    f1   = f1_score(y_te,y_pred,average=avg,zero_division=0)
    try:
        auc = roc_auc_score(y_te, mdl.predict_proba(X_te)[:,1] if is_bin
              else mdl.predict_proba(X_te), multi_class="ovr" if not is_bin else "raise",
              average="weighted" if not is_bin else None)
    except Exception: auc = None

    metrics = {"Method":method,"Sheet":st.session_state.get("active_sheet",""),
               "Accuracy":f"{acc:.4f}","Precision":f"{prec:.4f}",
               "Recall":f"{rec:.4f}","F1-Score":f"{f1:.4f}",
               "AUC":f"{auc:.4f}" if auc else "N/A",
               "Train rows":len(X_tr),"Test rows":len(X_te)}

    # ── Metrics ───────────────────────────────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy",f"{acc:.2%}"); c2.metric("F1",f"{f1:.4f}")
    c3.metric("AUC",f"{auc:.4f}" if auc else "N/A"); c4.metric("Precision",f"{prec:.4f}")
    with st.expander("📋 Full classification report",expanded=False):
        st.text(classification_report(y_te,y_pred))

    col_a, col_b = st.columns(2)
    # ── Confusion Matrix ──────────────────────────────────────────────────────
    with col_a:
        st.markdown("##### 📊 Confusion Matrix")
        fig,ax = plt.subplots(figsize=(5,4))
        fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')
        sns.heatmap(confusion_matrix(y_te,y_pred),annot=True,fmt='d',cmap='Blues',ax=ax,
                    linewidths=0.5,linecolor='#30363d',annot_kws={"color":"#ffffff","size":12})
        ax.set_title("Confusion Matrix",color='#e6edf3')
        ax.set_xlabel("Predicted",color='#c9d1d9'); ax.set_ylabel("Actual",color='#c9d1d9')
        ax.tick_params(colors='#c9d1d9'); fig_to_st(fig)
        st.caption("Good: large numbers on the diagonal. Bad: large off-diagonal numbers = frequent misclassification.")

    # ── ROC or Feature Importance ─────────────────────────────────────────────
    with col_b:
        if is_bin and hasattr(mdl,"predict_proba"):
            st.markdown("##### 📈 ROC Curve")
            try:
                fpr,tpr,_ = roc_curve(y_te,mdl.predict_proba(X_te)[:,1])
                fig2,ax2 = plt.subplots(figsize=(5,4))
                fig2.patch.set_facecolor('#0d1117'); ax2.set_facecolor('#161b22')
                ax2.plot(fpr,tpr,color='#58a6ff',lw=2,label=f"AUC={auc:.3f}" if auc else "ROC")
                ax2.plot([0,1],[0,1],'r--',lw=1,label="Random")
                ax2.set_xlabel("FPR",color='#c9d1d9'); ax2.set_ylabel("TPR",color='#c9d1d9')
                ax2.set_title("ROC Curve",color='#e6edf3'); ax2.tick_params(colors='#c9d1d9')
                ax2.legend(facecolor='#161b22',labelcolor='#c9d1d9'); fig_to_st(fig2)
                st.caption("Good: curve bows toward top-left, AUC near 1.0. Bad: curve near the dashed line (AUC~0.5).")
            except Exception: pass

    # ── Feature importance ────────────────────────────────────────────────────
    if hasattr(mdl,"feature_importances_") or hasattr(mdl,"coef_"):
        st.markdown("##### 🏅 Feature Importance / Coefficients")
        if hasattr(mdl,"feature_importances_"):
            fi = pd.Series(mdl.feature_importances_,index=features).sort_values(ascending=False)
        else:
            fi = pd.Series(np.abs(mdl.coef_[0] if mdl.coef_.ndim>1 else mdl.coef_),
                           index=features).sort_values(ascending=False)
        fig3,ax3 = plt.subplots(figsize=(7,max(3,min(15,len(fi))*0.35)))
        fig3.patch.set_facecolor('#0d1117'); ax3.set_facecolor('#161b22')
        fi.head(15).plot(kind='barh',ax=ax3,color='#58a6ff')
        ax3.set_title("Top 15 Features",color='#e6edf3')
        ax3.tick_params(colors='#c9d1d9',labelcolor='#c9d1d9'); ax3.invert_yaxis()
        plt.tight_layout(); fig_to_st(fig3)
        st.caption("Longer bars = more influence on prediction. Top features are where domain expertise matters most.")

    if method=="Classification Trees":
        with st.expander("🌿 Decision Tree Rules",expanded=False):
            st.code(export_text(mdl,feature_names=features,max_depth=4),language="")

    # ── Correlation matrix ────────────────────────────────────────────────────
    st.markdown("##### 🔗 Feature Correlation Matrix")
    corr = df_enc[features].corr()
    fig4,ax4 = plt.subplots(figsize=(max(5,len(features)*0.55),max(4,len(features)*0.5)))
    fig4.patch.set_facecolor('#0d1117'); ax4.set_facecolor('#161b22')
    sns.heatmap(corr,annot=len(features)<=15,fmt=".2f",cmap='coolwarm',ax=ax4,
                linewidths=0.3,linecolor='#21262d',annot_kws={"size":7,"color":"white"})
    ax4.set_title("Correlation Matrix",color='#e6edf3')
    ax4.tick_params(colors='#c9d1d9',labelcolor='#c9d1d9'); plt.tight_layout(); fig_to_st(fig4)
    st.caption("Values near +1 or -1 = strongly correlated. Highly correlated pairs may carry redundant information.")

    return metrics, mdl, X_te, y_te, y_pred


def run_regression(method, df, target, features, test_size):
    df_enc = encode_df(df[features+[target]].dropna())
    X = StandardScaler().fit_transform(df_enc[features].values)
    y = df_enc[target].values
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=test_size,random_state=42)
    mdl = LinearRegression() if method=="Linear Regression" else MLPRegressor(max_iter=500,random_state=42)
    mdl.fit(X_tr,y_tr); y_pred = mdl.predict(X_te)
    mse = mean_squared_error(y_te,y_pred); r2 = r2_score(y_te,y_pred)
    residuals = y_te - y_pred

    metrics = {"Method":method,"Sheet":st.session_state.get("active_sheet",""),
               "R²":f"{r2:.4f}","RMSE":f"{np.sqrt(mse):.4f}",
               "Train rows":len(X_tr),"Test rows":len(X_te)}

    c1,c2 = st.columns(2)
    c1.metric("R² Score",f"{r2:.4f}"); c2.metric("RMSE",f"{np.sqrt(mse):.4f}")
    if r2>=0.7: st.success(f"R² = {r2:.4f} — Good fit ({r2:.1%} variance explained).")
    elif r2>=0.4: st.warning(f"R² = {r2:.4f} — Moderate fit. Consider adding more features.")
    else: st.error(f"R² = {r2:.4f} — Weak fit.")

    col_a,col_b = st.columns(2)
    with col_a:
        st.markdown("##### 📊 Actual vs Predicted")
        fig,ax = plt.subplots(figsize=(5,4)); fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')
        ax.scatter(y_te,y_pred,alpha=0.5,color='#58a6ff',edgecolors='none',s=20)
        mn,mx = min(y_te.min(),y_pred.min()),max(y_te.max(),y_pred.max())
        ax.plot([mn,mx],[mn,mx],'r--',lw=1.5,label="Perfect")
        ax.set_xlabel("Actual",color='#c9d1d9'); ax.set_ylabel("Predicted",color='#c9d1d9')
        ax.set_title("Actual vs Predicted",color='#e6edf3'); ax.tick_params(colors='#c9d1d9')
        ax.legend(facecolor='#161b22',labelcolor='#c9d1d9',fontsize=8); fig_to_st(fig)
        st.caption("Good: dots cluster tightly along the red line. Bad: scattered widely = poor predictions.")
    with col_b:
        st.markdown("##### 📉 Residual Plot")
        fig2,ax2 = plt.subplots(figsize=(5,4)); fig2.patch.set_facecolor('#0d1117'); ax2.set_facecolor('#161b22')
        ax2.scatter(y_pred,residuals,alpha=0.5,color='#bc8cff',edgecolors='none',s=20)
        ax2.axhline(0,color='#f85149',linestyle='--',lw=1.5)
        ax2.set_xlabel("Predicted",color='#c9d1d9'); ax2.set_ylabel("Residual",color='#c9d1d9')
        ax2.set_title("Residual Plot",color='#e6edf3'); ax2.tick_params(colors='#c9d1d9'); fig_to_st(fig2)
        st.caption("Good: dots randomly scattered around zero. Bad: funnel or curve pattern = systematic error.")

    if method=="Linear Regression":
        coef = pd.Series(mdl.coef_,index=features).sort_values(key=abs,ascending=False)
        st.subheader("Coefficients")
        st.dataframe(coef.reset_index().rename(columns={"index":"Feature",0:"Coefficient"}),use_container_width=True)

    return metrics


def run_clustering(method, df, features, n_clusters):
    df_enc = encode_df(df[features].dropna())
    X = StandardScaler().fit_transform(df_enc.values)

    if method == "K-Means Clustering":
        st.markdown("##### 📊 Elbow Curve")
        mdl = KMeans(n_clusters=n_clusters,random_state=42,n_init=10)
        labels = mdl.fit_predict(X)
        k_range = range(2,min(11,len(X)))
        inertias = []
        for k in k_range:
            km = KMeans(n_clusters=k,random_state=42,n_init=10); km.fit(X); inertias.append(km.inertia_)
        fig,ax = plt.subplots(figsize=(6,3)); fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')
        ax.plot(list(k_range),inertias,'o-',color='#58a6ff',lw=2)
        ax.axvline(n_clusters,color='#f778ba',linestyle='--',lw=1.5,label=f"K={n_clusters}")
        ax.set_title("Elbow Curve",color='#e6edf3'); ax.set_xlabel("K",color='#c9d1d9')
        ax.set_ylabel("Inertia",color='#c9d1d9'); ax.tick_params(colors='#c9d1d9')
        ax.legend(facecolor='#161b22',labelcolor='#c9d1d9'); fig_to_st(fig)
        st.caption("Find the 'elbow' — where the curve bends sharply. That K is usually optimal.")
    else:
        mdl = AgglomerativeClustering(n_clusters=n_clusters); labels = mdl.fit_predict(X)
        st.markdown("##### 🌳 Dendrogram")
        linked = linkage(X[:min(200,len(X))],method='ward')
        fig,ax = plt.subplots(figsize=(8,4)); fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')
        dendrogram(linked,ax=ax,color_threshold=0,above_threshold_color='#58a6ff',leaf_font_size=6)
        ax.set_title("Dendrogram (sample 200 rows)",color='#e6edf3'); ax.tick_params(colors='#8b949e')
        plt.tight_layout(); fig_to_st(fig)
        st.caption("Cut the diagram horizontally — the number of vertical lines crossed = number of clusters.")

    try:
        sil = silhouette_score(X,labels)
        st.metric("Silhouette Score",f"{sil:.4f}")
        if sil>0.7: st.success(f"Silhouette = {sil:.4f} — Excellent separation!")
        elif sil>0.5: st.info(f"Silhouette = {sil:.4f} — Good separation.")
        elif sil>0.25: st.warning(f"Silhouette = {sil:.4f} — Moderate. Try different K.")
        else: st.error(f"Silhouette = {sil:.4f} — Weak. Clusters overlap.")
    except Exception: sil = None

    st.markdown("##### 🎨 2D PCA Cluster Scatter")
    n_comp = min(2,X.shape[1]); pca = PCA(n_components=n_comp,random_state=42)
    X_2d = pca.fit_transform(X); var_exp = pca.explained_variance_ratio_.sum()*100
    fig2,ax2 = plt.subplots(figsize=(7,5)); fig2.patch.set_facecolor('#0d1117'); ax2.set_facecolor('#161b22')
    pal = plt.cm.tab10.colors
    for c in np.unique(labels):
        mask = labels==c
        ax2.scatter(X_2d[mask,0],X_2d[mask,1] if n_comp>1 else np.zeros(mask.sum()),
                    color=pal[c%10],label=f"Cluster {c}",alpha=0.7,s=25,edgecolors='none')
    ax2.legend(facecolor='#161b22',labelcolor='#c9d1d9',fontsize=8)
    ax2.set_title(f"Clusters — PCA 2D ({var_exp:.1f}% variance)",color='#e6edf3')
    ax2.set_xlabel("PC1",color='#c9d1d9'); ax2.set_ylabel("PC2",color='#c9d1d9')
    ax2.tick_params(colors='#c9d1d9'); plt.tight_layout(); fig_to_st(fig2)
    st.caption(f"Good: clearly separated colour blobs. Bad: colours mixed together. ({var_exp:.1f}% of data variance shown.)")

    df_out = df[features].copy(); df_out["Cluster"] = labels
    metrics = {"Method":method,"Sheet":st.session_state.get("active_sheet",""),
               "K":n_clusters,"Silhouette":f"{sil:.4f}" if sil else "N/A",
               "Rows":len(df_out)}
    return metrics, df_out


def run_association(df, min_support, min_confidence, min_lift):
    records = []
    for _,row in df.iterrows():
        items = [str(v).strip() for v in row.dropna().values if str(v).strip()]
        if items: records.append(items)
    if not records: st.error("Could not parse transactional data."); return None, None
    te = TransactionEncoder(); te_arr = te.fit_transform(records)
    df_bool = pd.DataFrame(te_arr,columns=te.columns_)
    freq = apriori(df_bool,min_support=min_support,use_colnames=True)
    if freq.empty: st.warning("No frequent itemsets found. Lower min support."); return None, None
    rules = association_rules(freq,metric="lift",min_threshold=min_lift)
    rules = rules[rules["confidence"]>=min_confidence].sort_values("lift",ascending=False)
    st.success(f"Found {len(rules)} rules from {len(freq)} frequent itemsets.")
    display = rules[["antecedents","consequents","support","confidence","lift"]].head(20).copy()
    display["antecedents"] = display["antecedents"].apply(lambda x:", ".join(list(x)))
    display["consequents"] = display["consequents"].apply(lambda x:", ".join(list(x)))
    st.dataframe(display,use_container_width=True)
    if not rules.empty:
        fig,ax = plt.subplots(figsize=(7,4)); fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')
        sc = ax.scatter(rules["support"],rules["confidence"],c=rules["lift"],cmap="plasma",alpha=0.8,s=60)
        plt.colorbar(sc,ax=ax,label="Lift")
        ax.set_xlabel("Support",color='#c9d1d9'); ax.set_ylabel("Confidence",color='#c9d1d9')
        ax.set_title("Support vs Confidence (colour=Lift)",color='#e6edf3'); ax.tick_params(colors='#c9d1d9')
        fig_to_st(fig)
        st.caption("Top-right = high support & confidence rules. Darker colour = higher lift = stronger association.")
    metrics = {"Method":"Association Rules","Sheet":st.session_state.get("active_sheet",""),
               "Rules found":len(rules),"Min Support":min_support,
               "Min Confidence":min_confidence,"Min Lift":min_lift}
    return metrics, display


def export_to_excel(results_dict: dict) -> bytes:
    """Pack all result DataFrames into a multi-sheet Excel file."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet_name, df in results_dict.items():
            safe_name = str(sheet_name)[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    return buf.getvalue()

# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════
DEFAULTS = {
    "sheets": {}, "active_sheet": None,
    "wizard_step": 1,
    "ai_blueprint": "",          # Step 2 AI output
    "prep_transforms": {},        # sheet_name -> list of (label, json) for undo
    "prep_log": {},               # sheet_name -> list of (name, msg)
    "enc_mapping": {},            # sheet_name -> mapping dict
    "user_goal": "",
    "selected_methods": [],
    "run_results": [],            # list of result metric dicts
    "run_exports": {},            # label -> DataFrame for Excel export
    "comparison_ai": "",
    "gemini_key": "", "openrouter_key": "", "ai_vn": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p style="font-family:Space Mono;font-size:0.72rem;text-transform:uppercase;'
                'letter-spacing:2px;color:#58a6ff;border-bottom:1px solid #21262d;padding-bottom:6px">'
                '📁 Data Upload</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload CSV, Excel, JSON, TXT (multiple files OK)",
        accept_multiple_files=True, type=["csv","xlsx","xls","json","txt"],
    )
    if uploaded_files:
        all_sheets = {}
        for uf in uploaded_files:
            loaded = load_file(uf)
            for sh, df in loaded.items():
                key = f"{uf.name} › {sh}" if len(loaded)>1 else uf.name
                all_sheets[key] = df
        st.session_state["sheets"] = all_sheets

        chosen = st.selectbox("Active dataset", list(all_sheets.keys()))
        st.session_state["active_sheet"] = chosen
        df_preview = all_sheets[chosen]
        st.markdown(
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;'
            f'padding:8px 12px;color:#8b949e;font-size:0.8rem">'
            f'<b style="color:#58a6ff">{chosen}</b><br>'
            f'{df_preview.shape[0]:,} rows × {df_preview.shape[1]} cols</div>',
            unsafe_allow_html=True,
        )

        # ── Smart merge ───────────────────────────────────────────────────────
        if len(all_sheets) > 1:
            st.markdown("---")
            st.markdown('<p style="font-family:Space Mono;font-size:0.7rem;text-transform:uppercase;'
                        'letter-spacing:2px;color:#58a6ff">🔗 Merge Files</p>', unsafe_allow_html=True)
            col_freq = Counter(c for cols in (set(d.columns) for d in all_sheets.values()) for c in cols)
            shared_all  = [c for c,n in col_freq.items() if n==len(all_sheets)]
            shared_some = [c for c,n in col_freq.items() if 1<n<len(all_sheets)]
            if shared_all:
                st.success(f"✅ Shared in ALL files: {', '.join(shared_all[:5])}")
            elif shared_some:
                st.warning(f"⚠️ Partial match: {', '.join(shared_some[:5])} — some files will be skipped.")
            else:
                st.error("❌ No shared columns. Files cannot be merged by key. Analyse separately.")

            candidates = shared_all + shared_some
            if candidates:
                merge_on = st.selectbox("Merge key column", candidates)
            else:
                merge_on = st.text_input("Merge key (manual)", "")
            merge_how = st.selectbox("Merge type",
                ["outer (keep all rows)","inner (matching rows only)","left","right"]).split(" ")[0]

            if st.button("🔗 Merge all files"):
                if not merge_on:
                    st.error("Specify a merge key.")
                else:
                    eligible = {n:d for n,d in all_sheets.items() if merge_on in d.columns}
                    skipped  = [n for n in all_sheets if n not in eligible]
                    if len(eligible)<2:
                        st.error(f'Key "{merge_on}" found in only {len(eligible)} file(s).')
                    else:
                        merged = None
                        for name,df in eligible.items():
                            merged = df if merged is None else pd.merge(
                                merged, df, on=merge_on, how=merge_how,
                                suffixes=("",f"__{name[:8]}"))
                        if merged is not None:
                            key = "🔗 Merged Dataset"
                            st.session_state["sheets"][key] = merged
                            st.session_state["active_sheet"] = key
                            msg = (f"✅ Merged {len(eligible)} files → "
                                   f"{merged.shape[0]:,} rows × {merged.shape[1]} cols")
                            if skipped: msg += f"\n⚠️ Skipped: {', '.join(skipped)}"
                            st.success(msg); st.rerun()

    st.markdown("---")
    st.markdown('<p style="font-family:Space Mono;font-size:0.7rem;text-transform:uppercase;'
                'letter-spacing:2px;color:#58a6ff">🔑 AI Keys</p>', unsafe_allow_html=True)
    st.text_input("Gemini API Key", key="gemini_key", type="password", placeholder="AIzaSy…",
                  help="Free at aistudio.google.com")
    st.text_input("OpenRouter API Key (free fallback)", key="openrouter_key", type="password",
                  placeholder="sk-or-…", help="Free at openrouter.ai")
    g = bool(st.session_state.get("gemini_key","").strip())
    o = bool(st.session_state.get("openrouter_key","").strip())
    if g and o: st.success("✅ Both AI keys set")
    elif g: st.info("🔵 Gemini key set")
    elif o: st.info("🟣 OpenRouter key set")
    else: st.warning("⚠️ No AI key — ML still works without one")
    st.markdown("---")
    st.markdown('<p style="color:#8b949e;font-size:0.72rem;text-align:center">DataMine AI · sklearn + Gemini</p>',
                unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">🧠 DataMine AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">5-step guided data mining · Tải dữ liệu · Hỏi AI · Làm sạch · Phân tích · Xuất kết quả</div>',
            unsafe_allow_html=True)

# ── Step nav pills ────────────────────────────────────────────────────────────
ws = st.session_state["wizard_step"]
steps = ["1 Upload","2 AI Blueprint","3 Prepare Data","4 Run Analysis","5 Results & Export"]
def pill_cls(i):
    n = i+1
    if n < ws: return "done"
    if n == ws: return "active"
    return ""
pills_html = '<div class="step-nav">' + "".join(
    f'<div class="step-pill {pill_cls(i)}">{s}</div>' for i,s in enumerate(steps)
) + '</div>'
st.markdown(pills_html, unsafe_allow_html=True)

# Guard: need data to proceed past step 1
if not st.session_state["sheets"]:
    st.markdown("""
    <div class="card card-accent">
    <b style="color:#58a6ff">👋 Welcome to DataMine AI</b><br><br>
    <ol style="color:#c9d1d9;line-height:2.2">
      <li><b>Upload</b> one or more data files in the sidebar (CSV, Excel, JSON, TXT, multiple sheets OK).</li>
      <li><b>Describe your goal</b> — AI reads your data and gives a concise blueprint: what to clean, what technique to use, what to expect.</li>
      <li><b>Prepare</b> — merge files, fix missing values, encode text, remove duplicates with one click.</li>
      <li><b>Run</b> — pick one or many methods; results appear side by side in a comparison table.</li>
      <li><b>Export</b> — download predictions, cluster assignments, association rules, and AI explanations as Excel.</li>
    </ol>
    </div>""", unsafe_allow_html=True)
    st.stop()

# Active dataframe (with any prep applied)
active_name = st.session_state["active_sheet"]
raw_df = st.session_state["sheets"][active_name]

# Retrieve cleaned version if prep was done
_prep_log = st.session_state["prep_log"].get(active_name, [])
_prep_json_stack = st.session_state["prep_transforms"].get(active_name, [])
if _prep_json_stack:
    df_active = _json_to_df(_prep_json_stack[-1][1])
else:
    df_active = raw_df.copy()

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA PREVIEW
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">📂 Step 1 — Data Overview</div>', unsafe_allow_html=True)

with st.expander(f"🔍 Preview: {active_name}  ({df_active.shape[0]:,} rows × {df_active.shape[1]} cols)",
                 expanded=True):
    tab_t, tab_s, tab_c = st.tabs(["Table","Statistics","Column Types"])
    with tab_t: st.dataframe(df_active.head(50), use_container_width=True)
    with tab_s: st.dataframe(df_active.describe(include="all"), use_container_width=True)
    with tab_c:
        dt = df_active.dtypes.reset_index(); dt.columns = ["Column","Type"]
        dt["Nulls"] = df_active.isnull().sum().values
        dt["Unique"] = df_active.nunique().values
        dt["Sample"] = [str(df_active[c].dropna().iloc[0]) if df_active[c].dropna().shape[0]>0 else "" for c in df_active.columns]
        st.dataframe(dt, use_container_width=True)

if ws == 1:
    if st.button("▶ Continue to AI Blueprint →", type="primary"):
        st.session_state["wizard_step"] = 2; st.rerun()

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — AI BLUEPRINT
# ──────────────────────────────────────────────────────────────────────────────
if ws >= 2:
    st.markdown('<div class="section-hdr">🤖 Step 2 — AI Blueprint: What Should You Do?</div>',
                unsafe_allow_html=True)

    user_goal = st.text_area(
        "Describe your goal in any language:",
        value=st.session_state.get("user_goal",""),
        placeholder="e.g. 'I want to predict which employees are likely to leave', "
                    "'Find groups of customers with similar behaviour', "
                    "'Discover which maintenance topics are done together'…",
        height=90, key="user_goal_input",
    )
    st.session_state["user_goal"] = user_goal

    g_key = st.session_state.get("gemini_key","").strip()
    o_key = st.session_state.get("openrouter_key","").strip()
    if not g_key and not o_key:
        st.warning("⚠️ Paste an AI key in the sidebar to unlock the Blueprint. You can still run ML methods without one.")

    col_btn1, col_btn2 = st.columns([2,1])
    with col_btn1:
        analyse_clicked = st.button("🔎 Generate AI Blueprint", type="primary")
    with col_btn2:
        if st.session_state["ai_blueprint"]:
            if st.button("🗑️ Clear & re-run"):
                st.session_state["ai_blueprint"] = ""; st.rerun()

    if analyse_clicked and user_goal.strip():
        all_loaded = st.session_state.get("sheets",{})
        summaries = "\n\n".join(
            f"=== DATASET: {n} ===\n{df_summary(d)}" for n,d in list(all_loaded.items())[:4]
        )
        prompt = f"""You are an expert data scientist. The user has uploaded {len(all_loaded)} dataset(s).

{summaries}

USER GOAL: {user_goal}

Provide a CONCISE, STRUCTURED blueprint covering exactly these sections:

1. DATASET UNDERSTANDING
What each dataset contains. What the columns represent. Note any date/ID/text columns.

2. DATA QUALITY ISSUES
List specific problems: missing values (which columns, what %), outliers (which columns), text columns needing encoding, duplicate rows, scale differences. Be specific with column names.

3. RECOMMENDED TRANSFORMATIONS (in order)
Step-by-step: what to fix first, what encoding is needed, whether to merge files and on which key, any feature engineering that would help.

4. BEST TECHNIQUES FOR THIS GOAL
Recommend 2-4 specific methods from: Logistic Regression, Random Forest, Linear Regression, Neural Networks (MLP), K-Means Clustering, Hierarchical Clustering, Association Rules (Apriori), LDA, KNN, Naive Bayes, SVM. Explain WHY each is suitable. If multiple methods, explain which to compare.

5. TARGET & FEATURE COLUMNS
For supervised methods: which column is the target/dependent variable and why. Which columns are features. Reference actual column names.

6. EXPECTED CHALLENGES
Class imbalance? Too many features? Correlated columns? Small dataset? Warn the user.

FORMATTING: No markdown symbols (no **, no *, no #). Write in numbered paragraphs only. Be direct and specific. Maximum 500 words."""
        with st.spinner("AI is reading your data and building a blueprint…"):
            result = ask_ai(prompt)
        st.session_state["ai_blueprint"] = result
        st.session_state["wizard_step"] = 2

    if st.session_state["ai_blueprint"]:
        render_ai_box(st.session_state["ai_blueprint"], "🤖 AI Blueprint")

        col_vn, _ = st.columns([1,2])
        with col_vn:
            if st.button("🇻🇳 Dịch sang Tiếng Việt", key="translate_blueprint"):
                with st.spinner("Đang dịch..."):
                    vn = ask_ai(f"Translate this to natural Vietnamese. No markdown symbols. Plain paragraphs only.\n\n{st.session_state['ai_blueprint'][:3000]}")
                st.session_state["ai_vn"] = vn
        if st.session_state.get("ai_vn"):
            render_ai_box(st.session_state["ai_vn"], "🇻🇳 Phân tích Tiếng Việt")

        if ws == 2:
            if st.button("▶ Continue to Prepare Data →", type="primary"):
                st.session_state["wizard_step"] = 3; st.rerun()

    st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — PREPARE DATA
# ──────────────────────────────────────────────────────────────────────────────
if ws >= 3:
    st.markdown('<div class="section-hdr">🧹 Step 3 — Prepare Data (Transform & Engineer)</div>',
                unsafe_allow_html=True)

    prep_log_key = active_name
    if prep_log_key not in st.session_state["prep_log"]:
        st.session_state["prep_log"][prep_log_key] = []
    if prep_log_key not in st.session_state["prep_transforms"]:
        st.session_state["prep_transforms"][prep_log_key] = []

    stack = st.session_state["prep_transforms"][prep_log_key]
    log   = st.session_state["prep_log"][prep_log_key]
    current_json = stack[-1][1] if stack else _df_to_json(raw_df)
    current_df   = _json_to_df(current_json)

    # ── Auto-detected problems ────────────────────────────────────────────────
    st.markdown("**🔍 Auto-detected Data Problems:**")
    with st.spinner("Scanning…"):
        problems = detect_problems(current_json)

    icon_map = {"err":"🔴","warn":"🟡","info":"🔵","ok":"✅"}
    chip_map = {"err":"chip-err","warn":"chip-warn","info":"chip-info","ok":"chip-ok"}
    for p in problems:
        chip = f'<span class="chip {chip_map.get(p["sev"],"chip-info")}">{p["type"].upper()}</span>'
        fix  = f' <span style="color:#8b949e;font-size:0.77rem">→ {p["fix"]}</span>' if p["fix"] else ""
        st.markdown(f'<div class="prob-row">{icon_map.get(p["sev"],"ℹ️")} {chip} <span>{p["msg"]}</span>{fix}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🔧 Apply Fixes (click in order as suggested above):**")

    has_missing  = current_df.isnull().any().any()
    has_dups     = current_df.duplicated().any()
    has_text     = len(current_df.select_dtypes(include=["object","string"]).columns) > 0
    num_c = current_df.select_dtypes(include=[np.number]).columns
    has_outliers = any(
        ((current_df[c]<current_df[c].quantile(0.25)-3*(current_df[c].quantile(0.75)-current_df[c].quantile(0.25)))|
         (current_df[c]>current_df[c].quantile(0.75)+3*(current_df[c].quantile(0.75)-current_df[c].quantile(0.25)))).any()
        for c in num_c if (current_df[c].quantile(0.75)-current_df[c].quantile(0.25))>0
    )

    b1,b2,b3,b4 = st.columns(4)
    with b1:
        lbl = "🩹 Fix Missing Values" if has_missing else "✅ No Missing Values"
        if st.button(lbl, disabled=not has_missing, key=f"b_miss_{active_name}"):
            nj, msg = fix_missing(current_json)
            stack.append(("🩹 Fix Missing", nj)); log.append(("🩹 Missing Values", msg))
            detect_problems.clear(); st.rerun()
    with b2:
        lbl = "🗑️ Remove Duplicates" if has_dups else "✅ No Duplicates"
        if st.button(lbl, disabled=not has_dups, key=f"b_dup_{active_name}"):
            nj, msg = fix_duplicates(current_json)
            stack.append(("🗑️ Remove Dups", nj)); log.append(("🗑️ Duplicates", msg))
            detect_problems.clear(); st.rerun()
    with b3:
        lbl = "📐 Cap Outliers (3×IQR)" if has_outliers else "✅ No Extreme Outliers"
        if st.button(lbl, disabled=not has_outliers, key=f"b_out_{active_name}"):
            nj, msg = fix_outliers(current_json)
            stack.append(("📐 Cap Outliers", nj)); log.append(("📐 Outliers", msg))
            detect_problems.clear(); st.rerun()
    with b4:
        lbl = "🔤 Encode Text Columns" if has_text else "✅ No Text Columns"
        if st.button(lbl, disabled=not has_text, key=f"b_enc_{active_name}"):
            nj, msg, mapping = fix_encode(current_json)
            stack.append(("🔤 Encode Text", nj)); log.append(("🔤 Encoding", msg))
            st.session_state["enc_mapping"][active_name] = mapping
            detect_problems.clear(); st.rerun()

    # ── Encoding mapping viewer ───────────────────────────────────────────────
    if st.session_state["enc_mapping"].get(active_name):
        with st.expander("🗂️ Encoding Mapping — what each number means", expanded=False):
            for col, vm in st.session_state["enc_mapping"][active_name].items():
                st.markdown(f"**`{col}`**")
                mdf = pd.DataFrame(list(vm.items()), columns=["Original","Encoded"]).sort_values("Encoded")
                st.dataframe(mdf, use_container_width=True, height=min(200,35*len(mdf)+38))

    # ── Change log + before/after ─────────────────────────────────────────────
    if log:
        st.markdown("**📋 Change Log:**")
        for name, msg in log:
            st.markdown(
                f'<div style="background:#1a3a2a;border-left:3px solid #3fb950;'
                f'padding:5px 12px;border-radius:4px;margin-bottom:3px;color:#e6edf3;font-size:0.83rem">'
                f'<b style="color:#3fb950">{name}:</b> {msg}</div>',
                unsafe_allow_html=True,
            )
        current_df2 = _json_to_df(stack[-1][1])
        ca, cb = st.columns(2)
        with ca:
            st.markdown('<b style="color:#f85149">Before</b>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:#1f1414;border:1px solid #f85149;border-radius:8px;'
                f'padding:8px 12px;color:#e6edf3;font-size:0.82rem">'
                f'📏 {raw_df.shape[0]:,} rows × {raw_df.shape[1]} cols<br>'
                f'❓ {raw_df.isnull().sum().sum():,} missing &nbsp;|&nbsp; '
                f'📋 {raw_df.duplicated().sum():,} duplicates</div>', unsafe_allow_html=True)
            st.dataframe(raw_df.head(5), use_container_width=True, height=170)
        with cb:
            st.markdown('<b style="color:#3fb950">After (cleaned)</b>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:#0d1f17;border:1px solid #3fb950;border-radius:8px;'
                f'padding:8px 12px;color:#e6edf3;font-size:0.82rem">'
                f'📏 {current_df2.shape[0]:,} rows × {current_df2.shape[1]} cols<br>'
                f'❓ {current_df2.isnull().sum().sum():,} missing &nbsp;|&nbsp; '
                f'📋 {current_df2.duplicated().sum():,} duplicates</div>', unsafe_allow_html=True)
            st.dataframe(current_df2.head(5), use_container_width=True, height=170)

        # Download cleaned dataset
        cleaned_csv = current_df2.to_csv(index=False).encode()
        st.download_button("⬇️ Download Cleaned Dataset (CSV)", cleaned_csv,
                           file_name=f"cleaned_{active_name.replace(' ','_')[:30]}.csv",
                           mime="text/csv")

        col_undo, col_reset = st.columns(2)
        with col_undo:
            if st.button("↩️ Undo Last Fix"):
                stack.pop(); log.pop(); detect_problems.clear(); st.rerun()
        with col_reset:
            if st.button("🔄 Reset All"):
                st.session_state["prep_transforms"][active_name] = []
                st.session_state["prep_log"][active_name] = []
                if active_name in st.session_state["enc_mapping"]:
                    del st.session_state["enc_mapping"][active_name]
                detect_problems.clear(); st.rerun()
    else:
        st.info("👆 No fixes applied yet. Click buttons above to clean your data, or proceed directly if data is already clean.")

    if ws == 3:
        if st.button("▶ Continue to Run Analysis →", type="primary"):
            st.session_state["wizard_step"] = 4; st.rerun()

    st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — RUN ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
if ws >= 4:
    st.markdown('<div class="section-hdr">⚡ Step 4 — Choose Methods & Run Analysis</div>',
                unsafe_allow_html=True)

    # Re-derive df_active from latest prep state
    stack4 = st.session_state["prep_transforms"].get(active_name, [])
    df_active = _json_to_df(stack4[-1][1]) if stack4 else raw_df.copy()

    numeric_cols = df_active.select_dtypes(include=[np.number]).columns.tolist()
    all_cols     = df_active.columns.tolist()

    # ── Method selection grid ─────────────────────────────────────────────────
    st.markdown("**Select one or more methods (blue = run, green = also add to comparison):**")

    for group_id, gmeta in GROUP_META.items():
        st.markdown(f"**{gmeta['icon']} {gmeta['label']}**")
        methods_in = [(n,m) for n,m in METHODS.items() if m["group"]==group_id]
        cols = st.columns(min(4, len(methods_in)))
        for i,(name,meta) in enumerate(methods_in):
            with cols[i % len(cols)]:
                in_sel = name in st.session_state["selected_methods"]
                bg  = "#1a3a2a" if in_sel else "#161b22"
                brd = "2px solid #3fb950" if in_sel else "1px solid #30363d"
                st.markdown(
                    f'<div style="background:{bg};border:{brd};border-radius:10px;'
                    f'padding:0.7rem;margin-bottom:0.5rem;">'
                    f'<span class="badge {meta["badge"]}">{group_id[:4].upper()}</span><br>'
                    f'<b style="color:#e6edf3;font-size:0.85rem">{name}</b><br>'
                    f'<small style="color:#8b949e">{meta["vn"]}</small><br>'
                    f'<small style="color:#c9d1d9;font-size:0.77rem;line-height:1.4">{meta["desc"]}</small>'
                    f'</div>', unsafe_allow_html=True)
                if in_sel:
                    if st.button(f"✓ Remove", key=f"rm_{name}"):
                        st.session_state["selected_methods"].remove(name); st.rerun()
                else:
                    if st.button(f"＋ Select", key=f"add_{name}"):
                        st.session_state["selected_methods"].append(name); st.rerun()
        st.markdown("")

    if not st.session_state["selected_methods"]:
        st.info("👆 Select at least one method above.")
    else:
        sel_html = " · ".join(
            f'<span style="background:#1f3a5f;color:#58a6ff;padding:2px 8px;border-radius:4px;'
            f'font-size:0.8rem">{m}</span>' for m in st.session_state["selected_methods"])
        st.markdown(f'<div style="margin-bottom:0.5rem">Selected: {sel_html}</div>', unsafe_allow_html=True)
        if st.button("🗑️ Clear all selections"):
            st.session_state["selected_methods"] = []; st.rerun()

    st.markdown("---")

    # ── Shared config ─────────────────────────────────────────────────────────
    sel = st.session_state["selected_methods"]
    has_clf  = any(METHODS[m]["group"]=="classification" for m in sel)
    has_pred = any(METHODS[m]["group"]=="prediction" for m in sel)
    has_clus = any(METHODS[m]["group"]=="association" and "Cluster" in m for m in sel)
    has_assoc= "Association Rules (Apriori)" in sel
    has_bal  = any(METHODS[m]["group"]=="balancing" for m in sel)

    target_col = None; feature_cols = []; test_size = 0.2; balance_opt = "None"; n_clusters = 3

    if has_clf or has_pred or has_bal:
        st.markdown("**⚙️ Configure Supervised Methods:**")
        ca, cb = st.columns(2)
        with ca:
            auto_t = suggest_target(df_active)
            def_idx = all_cols.index(auto_t) if auto_t and auto_t in all_cols else 0
            if auto_t:
                st.markdown(f'<small style="color:#3fb950">✅ Auto-suggested target: <b>{auto_t}</b></small>',
                            unsafe_allow_html=True)
            target_col = st.selectbox("🎯 Target / Dependent column", all_cols, index=def_idx)
        with cb:
            feature_cols = st.multiselect(
                "📐 Feature columns",
                [c for c in all_cols if c != target_col],
                default=[c for c in numeric_cols if c != target_col][:10],
            )
        test_size = st.slider(
            "Test split % ℹ️ (data held out for evaluation — not used in training)",
            10, 40, 20, help="20% = model trains on 80%, evaluated on 20%."
        ) / 100
        if has_clf:
            balance_opt = st.selectbox("Class balancing (optional)", ["None","Random Oversampling","SMOTE"])

    if has_clus:
        st.markdown("**⚙️ Configure Clustering:**")
        cc1, cc2 = st.columns(2)
        with cc1:
            clus_features = st.multiselect("Clustering features", numeric_cols, default=numeric_cols[:8])
        with cc2:
            n_clusters = st.slider("Number of clusters (K)", 2, 10, 3)

    if has_assoc:
        st.markdown("**⚙️ Configure Association Rules:**")
        ra1, ra2, ra3 = st.columns(3)
        with ra1: min_sup  = st.slider("Min Support", 0.01, 0.5, 0.05, 0.01)
        with ra2: min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05)
        with ra3: min_lift = st.slider("Min Lift", 1.0, 10.0, 1.0, 0.1)

    # ── RUN button ────────────────────────────────────────────────────────────
    if sel and st.button("🚀 Run Selected Methods", type="primary"):
        st.session_state["run_results"] = []
        st.session_state["run_exports"] = {}

        for method in sel:
            group = METHODS[method]["group"]
            st.markdown(f'<div class="section-hdr">Running: {method}</div>', unsafe_allow_html=True)

            try:
                if group == "classification":
                    if not feature_cols: st.error(f"{method}: select feature columns."); continue
                    metrics, mdl, X_te, y_te, y_pred = run_classification(
                        method, df_active, target_col, feature_cols, test_size, balance_opt)
                    st.session_state["run_results"].append(metrics)
                    # Export predictions
                    pred_df = df_active.iloc[:len(y_te)][feature_cols].copy()
                    pred_df["Actual"] = y_te; pred_df["Predicted"] = y_pred
                    st.session_state["run_exports"][f"{method}_predictions"] = pred_df

                elif group == "prediction":
                    if not feature_cols: st.error(f"{method}: select feature columns."); continue
                    metrics = run_regression(method, df_active, target_col, feature_cols, test_size)
                    st.session_state["run_results"].append(metrics)

                elif group == "association":
                    if "Cluster" in method:
                        fc = clus_features if 'clus_features' in dir() and clus_features else numeric_cols[:6]
                        metrics, df_out = run_clustering(method, df_active, fc, n_clusters)
                        st.session_state["run_results"].append(metrics)
                        st.session_state["run_exports"][f"{method}_clusters"] = df_out
                    else:
                        metrics, rules_df = run_association(df_active,
                            min_sup if 'min_sup' in dir() else 0.05,
                            min_conf if 'min_conf' in dir() else 0.3,
                            min_lift if 'min_lift' in dir() else 1.0)
                        if metrics:
                            st.session_state["run_results"].append(metrics)
                            if rules_df is not None:
                                st.session_state["run_exports"]["Association_Rules"] = rules_df

                elif group == "balancing":
                    if not feature_cols: st.error(f"{method}: select feature columns."); continue
                    df_enc = encode_df(df_active[feature_cols+[target_col]].dropna())
                    X = df_enc[feature_cols].values; y = df_enc[target_col].values
                    orig = pd.Series(y).value_counts()
                    if method == "Random Oversampling":
                        Xr,yr = RandomOverSampler(random_state=42).fit_resample(X,y)
                    else:
                        Xr,yr = SMOTE(random_state=42).fit_resample(X,y)
                    new = pd.Series(yr).value_counts()
                    bc1,bc2 = st.columns(2)
                    with bc1: st.subheader("Before"); st.bar_chart(orig)
                    with bc2: st.subheader("After");  st.bar_chart(new)
                    st.success(f"{method}: {len(y):,} → {len(yr):,} samples.")
                    metrics = {"Method":method,"Sheet":active_name,
                               "Before rows":len(y),"After rows":len(yr)}
                    st.session_state["run_results"].append(metrics)

            except Exception as e:
                st.error(f"Error running {method}: {e}")

        st.session_state["wizard_step"] = 5; st.rerun()

    st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — RESULTS, AI EXPLANATION & EXPORT
# ──────────────────────────────────────────────────────────────────────────────
if ws >= 5 and st.session_state["run_results"]:
    st.markdown('<div class="section-hdr">📊 Step 5 — Results, AI Explanation & Export</div>',
                unsafe_allow_html=True)

    results = st.session_state["run_results"]

    # ── Comparison table ──────────────────────────────────────────────────────
    st.markdown("#### 📋 Methods Comparison Table")
    cmp_df = pd.DataFrame(results)
    col_order = ["Sheet","Method","Accuracy","Precision","Recall","F1-Score","AUC",
                 "R²","RMSE","K","Silhouette","Rules found","Train rows","Test rows","Rows"]
    col_order = [c for c in col_order if c in cmp_df.columns]
    cmp_df = cmp_df[col_order]
    st.dataframe(cmp_df, use_container_width=True)

    # Highlight best classification by F1 if available
    if "F1-Score" in cmp_df.columns:
        try:
            best = cmp_df.loc[cmp_df["F1-Score"].astype(float).idxmax()]
            st.success(f"🏆 Best classification result: **{best.get('Method','')}** — F1 = {best['F1-Score']}")
        except Exception: pass

    st.markdown("---")

    # ── AI explanation across all methods ─────────────────────────────────────
    st.markdown("#### 🤖 AI Explanation & Recommendations")

    if st.button("🔎 Generate AI Explanation for All Results", type="primary"):
        results_text = cmp_df.to_string(index=False)
        prompt = f"""You are a data mining expert. The user ran the following methods on their dataset.

DATASET: {active_name} ({df_active.shape[0]} rows x {df_active.shape[1]} cols)
USER GOAL: {st.session_state.get('user_goal','(not specified)')}

RESULTS TABLE:
{results_text}

Please provide:

1. OVERALL PERFORMANCE SUMMARY
Summarise what each method achieved. Which performed best and why.

2. METHOD-BY-METHOD ANALYSIS
For each method: what the metrics mean, whether the performance is good or poor, any limitations.

3. COMPARISON INSIGHTS
If multiple methods ran: which is the winner and why. What the differences reveal about the data.

4. PRACTICAL RECOMMENDATIONS
What the user should do next. Which model to deploy or trust. What to improve.

5. RED FLAGS
Any concerning signs: overfitting, class imbalance, low AUC, weak silhouette, etc.

FORMATTING: No **, no *, no #, no dashes as bullets. Plain numbered paragraphs only. Max 500 words."""
        with st.spinner("AI is analysing all your results…"):
            ai_exp = ask_ai(prompt)
        st.session_state["comparison_ai"] = ai_exp

    if st.session_state.get("comparison_ai"):
        render_ai_box(st.session_state["comparison_ai"], "🤖 AI Analysis of All Results")

        col_vn2, _ = st.columns([1,2])
        with col_vn2:
            if st.button("🇻🇳 Dịch kết quả sang Tiếng Việt", key="translate_results"):
                with st.spinner("Đang dịch..."):
                    vn2 = ask_ai(f"Translate to natural Vietnamese. No markdown. Plain paragraphs.\n\n{st.session_state['comparison_ai'][:3000]}")
                st.session_state["ai_vn"] = vn2
        if st.session_state.get("ai_vn"):
            render_ai_box(st.session_state["ai_vn"], "🇻🇳 Giải thích Tiếng Việt")

    st.markdown("---")

    # ── Export to Excel ───────────────────────────────────────────────────────
    st.markdown("#### ⬇️ Export Results to Excel")
    st.caption("Download all results — predictions, cluster assignments, association rules, comparison table — in one Excel file.")

    export_sheets = {"Comparison Table": cmp_df}
    for label, df_exp in st.session_state.get("run_exports",{}).items():
        export_sheets[label[:31]] = df_exp

    # Add AI explanation as a text sheet
    if st.session_state.get("comparison_ai"):
        ai_lines = st.session_state["comparison_ai"].split("\n")
        ai_sheet = pd.DataFrame({"AI Explanation": ai_lines})
        export_sheets["AI Explanation"] = ai_sheet

    excel_bytes = export_to_excel(export_sheets)
    fname = f"DataMineAI_Results_{active_name[:20].replace(' ','_')}.xlsx"
    st.download_button(
        "⬇️ Download Full Results (Excel)",
        data=excel_bytes,
        file_name=fname,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("---")
    if st.button("🔄 Start New Analysis (reset all)"):
        for key in ["wizard_step","ai_blueprint","prep_transforms","prep_log","enc_mapping",
                    "user_goal","selected_methods","run_results","run_exports","comparison_ai","ai_vn"]:
            st.session_state[key] = DEFAULTS.get(key, [] if "list" in str(type(DEFAULTS.get(key,[]))) else {})
        st.session_state["wizard_step"] = 1; st.rerun()
