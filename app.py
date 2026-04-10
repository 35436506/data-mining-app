import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataMine AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0d1117; }
.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); color: #e6edf3; }

h1,h2,h3 { font-family: 'Space Mono', monospace; color: #e6edf3; }

/* Force white text in all Streamlit widgets on dark bg */
.stMarkdown, .stText, p, span, div { color: inherit; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; }
[data-testid="stDataFrame"] { color: #e6edf3 !important; }
.stDataFrame td, .stDataFrame th { color: #e6edf3 !important; background: #161b22 !important; }
.stSelectbox div[data-baseweb="select"] { background: #161b22 !important; color: #e6edf3 !important; }
.stMultiSelect div[data-baseweb="select"] { background: #161b22 !important; color: #e6edf3 !important; }
div[data-baseweb="option"] { background: #161b22 !important; color: #e6edf3 !important; }
div[data-baseweb="popover"] { background: #161b22 !important; }
.stTextInput input, .stTextArea textarea { color: #e6edf3 !important; background: #161b22 !important; }
.stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] { color: #c9d1d9 !important; }
.stInfo, .stSuccess, .stWarning, .stError { color: #e6edf3 !important; }
/* Tab text */
button[data-baseweb="tab"] { color: #8b949e !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #58a6ff !important; }

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #58a6ff, #bc8cff, #f778ba);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}
.hero-sub {
    color: #8b949e;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.card-accent {
    border-left: 4px solid #58a6ff;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
    margin-right: 4px;
}
.badge-blue  { background:#1f3a5f; color:#58a6ff; }
.badge-purple{ background:#2d1f5f; color:#bc8cff; }
.badge-pink  { background:#3d1f35; color:#f778ba; }
.badge-green { background:#1a3a2a; color:#3fb950; }
.badge-yellow{ background:#3a2d10; color:#d29922; }

.method-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    cursor: pointer;
    transition: border-color 0.2s;
}
.method-card:hover { border-color: #58a6ff; }
.method-card.selected { border-color: #58a6ff; background: #1a2332; }

.result-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #8b949e;
}
.ai-bubble {
    background: #1c2333;
    border: 1px solid #58a6ff;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    color: #ffffff !important;
}
.ai-bubble * { color: #ffffff !important; }
.ai-bubble pre, .ai-bubble .stText {
    color: #e6edf3 !important;
    background: transparent !important;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 0.9rem;
    line-height: 1.7;
}
.preprocess-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
}
.problem-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
    border-bottom: 1px solid #21262d;
    color: #e6edf3;
    font-size: 0.88rem;
}
.problem-row:last-child { border-bottom: none; }
.chip-warn  { background:#3a2d10; color:#d29922; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; }
.chip-err   { background:#3d1f1f; color:#f85149; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; }
.chip-info  { background:#1f3a5f; color:#58a6ff; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; }
.chip-ok    { background:#1a3a2a; color:#3fb950; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; }
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #58a6ff;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.5rem;
}
div[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
}
.stButton>button {
    background: linear-gradient(90deg, #238636, #2ea043);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    padding: 0.5rem 1.2rem;
    transition: opacity 0.2s;
}
.stButton>button:hover { opacity: 0.85; }
.stSelectbox label, .stMultiselect label, .stTextArea label {
    color: #8b949e !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Imports (lazy to avoid cold-start errors) ─────────────────────────────────
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
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_squared_error, r2_score,
                             silhouette_score, confusion_matrix)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# ── AI setup ─────────────────────────────────────────────────────────────────
# No hardcoded keys — users must enter their own in the sidebar
_DEFAULT_GEMINI_KEY = ""  # intentionally blank — leaked keys are auto-revoked by Google
_GEMINI_CANDIDATES  = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
# OpenRouter free endpoint — no billing needed, many models available
_OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
# "openrouter/free" auto-picks the best available free model — future-proof
# Specific fallbacks are current working free models as of April 2026
_OPENROUTER_MODELS  = [
    "openrouter/auto",            # auto-router: picks best available
    "google/gemma-3-27b-it:free", # Google Gemma 3 27B
    "meta-llama/llama-3.3-70b-instruct:free",  # Meta Llama 3.3 70B
    "nvidia/llama-3.1-nemotron-ultra-253b:free", # NVIDIA Nemotron
    "deepseek/deepseek-r1:free",  # DeepSeek R1
    "qwen/qwq-32b:free",          # Qwen 32B
]

def _make_model(name, key):
    genai.configure(api_key=key)
    return genai.GenerativeModel(name)

# ─────────────────────────────────────────────────────────────────────────────
# METHOD CATALOGUE
# ─────────────────────────────────────────────────────────────────────────────
METHODS = {
    # ── Classification ────────────────────────────────────────────────────────
    "Logistic Regression": {
        "group": "classification",
        "badge": "badge-blue",
        "desc": "Classifies data into binary or multi-class categories using a probabilistic S-curve. Great starting point for classification tasks.",
        "vn": "Phân loại cơ bản",
    },
    "Linear Discriminant Analysis (LDA)": {
        "group": "classification",
        "badge": "badge-blue",
        "desc": "Projects data onto axes that maximise class separation. Works best when classes have roughly equal covariance.",
        "vn": "Phân tích biệt thức",
    },
    "K-Nearest Neighbors (KNN)": {
        "group": "classification",
        "badge": "badge-blue",
        "desc": "Assigns a label based on the majority class among the K closest training samples. Simple and interpretable.",
        "vn": "Phân loại theo láng giềng",
    },
    "Classification Trees": {
        "group": "classification",
        "badge": "badge-blue",
        "desc": "Builds a human-readable tree of IF-THEN rules to split data into classes. Highly interpretable.",
        "vn": "Cây quyết định",
    },
    "Naive Bayes": {
        "group": "classification",
        "badge": "badge-blue",
        "desc": "Applies Bayes' theorem with a naive independence assumption. Fast and strong on text and probability problems.",
        "vn": "Dự báo xác suất",
    },
    "Support Vector Machine (SVM)": {
        "group": "classification",
        "badge": "badge-blue",
        "desc": "Finds the hyperplane with maximum margin between classes. Powerful in high-dimensional spaces.",
        "vn": "Phân loại biên giới",
    },
    "Random Forest": {
        "group": "classification",
        "badge": "badge-blue",
        "desc": "Ensemble of decision trees trained on random subsets. Robust, accurate, and handles missing values well.",
        "vn": "Rừng ngẫu nhiên",
    },
    "Neural Networks (MLP)": {
        "group": "classification",
        "badge": "badge-blue",
        "desc": "Multi-layer perceptron that can learn complex non-linear patterns. Suitable for large, complex datasets.",
        "vn": "Mạng nơ-ron",
    },
    # ── Prediction ────────────────────────────────────────────────────────────
    "Linear Regression": {
        "group": "prediction",
        "badge": "badge-purple",
        "desc": "Models the linear relationship between features and a continuous target. Interpretable coefficients show feature impact.",
        "vn": "Hồi quy tuyến tính",
    },
    "Neural Networks Regression (MLP)": {
        "group": "prediction",
        "badge": "badge-purple",
        "desc": "MLP applied to regression tasks — predicts continuous values through stacked non-linear transformations.",
        "vn": "Mạng nơ-ron hồi quy",
    },
    # ── Association / Clustering / Balancing ──────────────────────────────────
    "Association Rules (Apriori)": {
        "group": "association",
        "badge": "badge-pink",
        "desc": "Discovers IF-THEN patterns in transactional data (e.g. 'customers who buy X also buy Y'). Uses support, confidence, lift.",
        "vn": "Luật kết hợp / Combo mua hàng",
    },
    "K-Means Clustering": {
        "group": "association",
        "badge": "badge-green",
        "desc": "Groups data into K clusters by minimising intra-cluster variance. Good for customer segmentation and anomaly detection.",
        "vn": "Phân cụm K-Means",
    },
    "Hierarchical Clustering": {
        "group": "association",
        "badge": "badge-green",
        "desc": "Builds a dendrogram of nested clusters without specifying K upfront. Helps reveal natural data hierarchy.",
        "vn": "Phân cụm phân cấp",
    },
    "Random Oversampling": {
        "group": "association",
        "badge": "badge-yellow",
        "desc": "Replicates minority-class samples randomly to fix class imbalance before classification.",
        "vn": "Cân bằng ngẫu nhiên",
    },
    "SMOTE": {
        "group": "association",
        "badge": "badge-yellow",
        "desc": "Generates synthetic minority-class samples by interpolating between existing ones, creating richer training data.",
        "vn": "Cân bằng tổng hợp (SMOTE)",
    },
}

GROUP_META = {
    "classification": {"label": "Classification / Phân loại", "color": "#58a6ff", "icon": "🔵"},
    "prediction":     {"label": "Prediction / Dự báo", "color": "#bc8cff", "icon": "🟣"},
    "association":    {"label": "Association, Clustering & Balancing / Kết hợp, Phân cụm & Cân bằng", "color": "#f778ba", "icon": "🔴"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_file(uploaded) -> dict[str, pd.DataFrame]:
    """Return {sheet_name: DataFrame} for any supported file type."""
    name = uploaded.name.lower()
    sheets = {}
    if name.endswith(".csv"):
        sheets["Sheet1"] = pd.read_csv(uploaded)
    elif name.endswith((".xlsx", ".xls")):
        xf = pd.ExcelFile(uploaded)
        for s in xf.sheet_names:
            sheets[s] = xf.parse(s)
    elif name.endswith(".json"):
        data = json.load(uploaded)
        if isinstance(data, list):
            sheets["Sheet1"] = pd.DataFrame(data)
        else:
            sheets["Sheet1"] = pd.DataFrame([data])
    elif name.endswith(".txt"):
        sheets["Sheet1"] = pd.read_csv(uploaded, sep=None, engine="python")
    else:
        st.error("Unsupported file type.")
    return sheets


def df_summary(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return (
        f"Shape: {df.shape}\n"
        f"Columns: {list(df.columns)}\n"
        f"Dtypes:\n{df.dtypes.to_string()}\n"
        f"Null counts:\n{df.isnull().sum().to_string()}\n"
        f"Sample (3 rows):\n{df.head(3).to_string()}\n"
        f"Describe:\n{df.describe(include='all').to_string()}"
    )


def _get_keys():
    """Read keys from session state — always fresh, never stale."""
    g = st.session_state.get("gemini_key", "").strip()
    o = st.session_state.get("openrouter_key", "").strip()
    # If user never touched the fields, fall back to compiled defaults
    if not g:
        g = _DEFAULT_GEMINI_KEY
    return g, o


def _call_openrouter(prompt: str, or_key: str) -> str:
    """Call OpenRouter API — free tier, no billing required."""
    if not or_key:
        return ""
    headers = {
        "Authorization": f"Bearer {or_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://datamine-ai.streamlit.app",
        "X-Title": "DataMine AI",
    }
    errors = []
    for model in _OPENROUTER_MODELS:
        try:
            resp = _requests.post(
                _OPENROUTER_URL,
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.7,
                },
                timeout=120,
            )
            if resp.status_code == 401:
                return "__OR_FAIL__: Invalid OpenRouter API key (401). Please check your key."
            data = resp.json()
            # Success path
            if "choices" in data and data["choices"]:
                txt = data["choices"][0]["message"]["content"]
                used_model = data.get("model", model)  # OR tells us actual model used
                return f"*(AI via OpenRouter · {used_model})*\n\n{txt}"
            # Error path
            err_obj = data.get("error", {})
            err_msg = err_obj.get("message", str(data)) if isinstance(err_obj, dict) else str(err_obj)
            errors.append(f"{model}: {err_msg}")
        except _requests.exceptions.Timeout:
            errors.append(f"{model}: timeout (120s)")
        except Exception as exc:
            errors.append(f"{model}: {exc}")
    return f"__OR_FAIL__: {' | '.join(errors)}"


def ask_gemini(prompt: str) -> str:
    """
    Try Google Gemini across all candidate models, then OpenRouter, then help text.
    Reads keys fresh every call so sidebar changes are always picked up.
    """
    gemini_key, or_key = _get_keys()
    gemini_errors = []

    # ── 1. Try every Gemini model ─────────────────────────────────────────────
    for model_name in _GEMINI_CANDIDATES:
        try:
            mdl = _make_model(model_name, gemini_key)
            resp = mdl.generate_content(prompt)
            return resp.text                        # ✅ success
        except Exception as e:
            gemini_errors.append(f"{model_name}: {e}")
            continue                                # always try next model

    # ── 2. Fallback to OpenRouter ─────────────────────────────────────────────
    if or_key:
        result = _call_openrouter(prompt, or_key)
        if result and not result.startswith("__OR_FAIL__"):
            return result
        or_debug = result  # save for diagnostics
    else:
        or_debug = "(no OpenRouter key provided)"

    # ── 3. Show diagnostics + help ────────────────────────────────────────────
    gemini_summary = "\n".join(f"  • {e}" for e in gemini_errors)
    return f"""⚠️ **Both AI providers failed.** Debug info:

**Gemini errors:**
{gemini_summary}

**OpenRouter:** {or_debug}

---
**How to fix:**

🔑 **Get a fresh Gemini key** (takes 2 min, free):
1. Open [aistudio.google.com](https://aistudio.google.com) in a **private/incognito window**
2. Sign in with a **different Google account** than before
3. Click **Get API Key → Create API key**
4. Paste into **"Khoa Gemini API / Gemini API Key"** in the sidebar and press Enter

🔑 **Get a free OpenRouter key** (no card needed):
1. Go to [openrouter.ai](https://openrouter.ai) → Sign up → **Keys → Create Key**
2. Paste into **"OpenRouter API Key"** in the sidebar

> All **ML methods below work without any AI key** — only this goal-analysis step needs one.
"""


def ask_gemini_multisheet(sheets: dict, user_goal: str) -> str:
    """Deep multi-sheet analysis — understands relationships between datasets."""
    ctx_parts = []
    for sheet_name, df in sheets.items():
        ctx_parts.append(
            f"=== DATASET: {sheet_name} ===\n"
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
            f"Columns: {list(df.columns)}\n"
            f"Types:\n{df.dtypes.to_string()}\n"
            f"Missing values:\n{df.isnull().sum().to_string()}\n"
            f"Sample (top 3 rows):\n{df.head(3).to_string()}\n"
            f"Statistics:\n{df.describe(include='all').to_string()}\n"
        )
    all_ctx = "\n\n".join(ctx_parts)

    prompt = f"""You are an expert data scientist and fraud analytics consultant.

The user has provided {len(sheets)} dataset(s)/sheet(s):
{all_ctx}

USER'S GOAL / QUESTION:
{user_goal}

Please provide a comprehensive analysis covering:

1. **Understanding of the datasets**: What does each sheet contain? What do columns represent? Are they questionnaire scores, binary answers, categorical items?

2. **Key columns identified**: Which column is most likely the fraud label/target? Which columns are features (questionnaire items)?

3. **Cross-dataset comparison** (if multiple sheets): How do the datasets relate? Are the questionnaires measuring the same constructs? Do they have overlapping or complementary columns?

4. **Recommended approach**: Given the goal, what specific data mining strategy do you recommend?
   - Which methods from [Logistic Regression, Random Forest, SVM, Neural Networks (MLP), LDA, KNN, Classification Trees, Naive Bayes] would be best?
   - Should datasets be merged or compared separately first?
   - How to handle class imbalance (if fraud cases are rare — use SMOTE or Random Oversampling)?
   - Feature importance: which questionnaire items matter most?

5. **Actionable next steps**: Step-by-step instructions tailored to this specific situation.

6. **Important data quality issues**: Missing values, encoding needs, scaling requirements.

Be specific, practical, and refer to actual column names you see in the data.
IMPORTANT FORMATTING RULES:
- Do NOT use any markdown: no **, no *, no #, no bullet points starting with -
- Write in numbered sections and plain paragraphs
- Use simple words that are clear without any formatting symbols
- End your response with a SHORT SUMMARY section (max 5 sentences) that tells the user EXACTLY which 2-3 methods to try first and why
Respond in the same language the user used."""

    return ask_gemini(prompt)


def encode_df(df: pd.DataFrame):
    df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def suggest_target_column(df: pd.DataFrame) -> str | None:
    """Heuristically guess the most likely target/dependent column."""
    TARGET_KEYWORDS = [
        "label", "target", "class", "churn", "fraud", "default", "outcome",
        "result", "status", "y", "output", "dependent", "response",
        "predict", "category", "nhãn", "kết_quả", "mục_tiêu",
    ]
    cols_lower = {c: c.lower().replace(" ", "_") for c in df.columns}
    # 1. Exact or substring keyword match
    for col, col_l in cols_lower.items():
        for kw in TARGET_KEYWORDS:
            if kw in col_l:
                return col
    # 2. Last column heuristic (common ML convention)
    # Only use if it's numeric or binary
    last = df.columns[-1]
    if df[last].nunique() <= 20:
        return last
    return None


def show_algorithm_flowchart(method: str):
    """Display a simple text-based flowchart and parameter explanation for each method."""
    FLOWCHARTS = {
        "Logistic Regression": {
            "steps": ["📥 Input features (X)", "⚖️ Multiply by learned weights + bias", "📈 Apply Sigmoid function → probability 0–1", "🏷️ Threshold at 0.5 → Class 0 or Class 1"],
            "params": "**max_iter=1000** — max training iterations; **C** — regularisation strength (default 1.0)",
            "note": "Best for: binary outcomes, interpretable coefficients. Assumes linear decision boundary.",
        },
        "Linear Discriminant Analysis (LDA)": {
            "steps": ["📥 Input features", "📊 Compute mean & variance per class", "📐 Find axes that maximise class separation", "🏷️ Project & classify by nearest centroid"],
            "params": "No key hyperparameters. Assumes equal covariance per class (Gaussian).",
            "note": "Best for: well-separated classes, small datasets. Also used for dimensionality reduction.",
        },
        "K-Nearest Neighbors (KNN)": {
            "steps": ["📥 New data point arrives", "📏 Calculate distance to ALL training points", f"👥 Pick K nearest neighbours (default K=5)", "🗳️ Majority vote → predicted class"],
            "params": "**n_neighbors=5** — number of neighbours K; **metric** — distance measure (Euclidean default).",
            "note": "Best for: small–medium datasets. No training phase. Slow at prediction on large data.",
        },
        "Classification Trees": {
            "steps": ["📥 All training data at root node", "✂️ Find best feature + threshold to split (Gini/Entropy)", "🌿 Recurse on each branch until max_depth or pure leaf", "🏷️ Leaf node → majority class label"],
            "params": "**max_depth=5** — limits tree depth to prevent overfitting.",
            "note": "Best for: interpretability — you can read the IF-THEN rules directly.",
        },
        "Naive Bayes": {
            "steps": ["📥 Input features", "📊 Compute P(class) and P(feature|class) from training data", "✖️ Multiply probabilities (naive independence)", "🏷️ Pick class with highest posterior probability"],
            "params": "No key hyperparameters for Gaussian NB. Assumes features are normally distributed.",
            "note": "Best for: text classification, high-dimensional sparse data. Very fast.",
        },
        "Support Vector Machine (SVM)": {
            "steps": ["📥 Input features (scaled)", "📐 Find hyperplane with maximum margin between classes", "🔲 Support vectors = closest points define the margin", "🏷️ New points classified by which side of hyperplane"],
            "params": "**kernel='rbf'** — maps to higher dimensions; **C** — margin softness; **gamma** — kernel width.",
            "note": "Best for: high-dimensional data, text, images. Slow on very large datasets.",
        },
        "Random Forest": {
            "steps": ["📥 Training data", "🌲×100 Build 100 decision trees on random data subsets + random features", "🗳️ Each tree votes on new sample", "🏆 Majority vote → final prediction + feature importance scores"],
            "params": "**n_estimators=100** — number of trees; **max_features='sqrt'** — features per split.",
            "note": "Best for: most datasets. Robust, handles missing values, gives feature importance.",
        },
        "Neural Networks (MLP)": {
            "steps": ["📥 Input layer (one node per feature)", "🔗 Hidden layers apply weights + ReLU activation", "🔁 Backpropagation adjusts weights to minimise loss", "📤 Output layer → class probabilities (Softmax)"],
            "params": "**hidden_layer_sizes=(100,)** — neurons per layer; **max_iter=500** — training epochs; **activation='relu'**.",
            "note": "Best for: complex non-linear patterns, large datasets. Less interpretable.",
        },
        "Linear Regression": {
            "steps": ["📥 Input features (X)", "⚖️ Fit weights to minimise Sum of Squared Errors", "📈 Output = w₁x₁ + w₂x₂ + … + b (continuous value)", "📊 Evaluate with R² and RMSE"],
            "params": "No key hyperparameters. Assumes linear relationship between features and target.",
            "note": "Best for: continuous target, interpretable coefficients showing feature impact.",
        },
        "Neural Networks Regression (MLP)": {
            "steps": ["📥 Input features", "🔗 Hidden layers apply non-linear transformations", "🔁 Backpropagation minimises Mean Squared Error", "📤 Output layer → single continuous value"],
            "params": "**hidden_layer_sizes=(100,)** — neurons; **max_iter=500** — epochs; **activation='relu'**.",
            "note": "Best for: non-linear regression with complex patterns. Needs more data than Linear Regression.",
        },
        "Association Rules (Apriori)": {
            "steps": ["📥 Transactional data (basket of items per row)", "🔍 Find all itemsets with support ≥ min_support", "📏 Compute confidence & lift for each rule", "📋 Filter rules by min_confidence and min_lift thresholds"],
            "params": "**min_support** — how often itemset appears; **min_confidence** — rule reliability; **min_lift** — improvement over random.",
            "note": "Best for: market basket analysis, recommendation systems.",
        },
        "K-Means Clustering": {
            "steps": ["📥 Input data (no labels)", "🎯 Randomly place K centroids", "🔁 Assign each point to nearest centroid → recompute centroids", "✅ Repeat until centroids stop moving"],
            "params": "**n_clusters=K** — number of groups; **n_init=10** — random restarts to avoid local optima.",
            "note": "Best for: customer segmentation. Use Elbow Curve to choose K.",
        },
        "Hierarchical Clustering": {
            "steps": ["📥 Input data (no labels)", "📏 Compute pairwise distances between all points", "🌿 Merge closest pair into a cluster (Ward linkage)", "🌲 Build dendrogram — cut at desired level for K clusters"],
            "params": "**n_clusters** — where to cut dendrogram; **linkage='ward'** — minimises within-cluster variance.",
            "note": "Best for: exploring natural groupings without specifying K upfront.",
        },
        "Random Oversampling": {
            "steps": ["📥 Imbalanced dataset (e.g. 90% Class 0, 10% Class 1)", "🔍 Identify minority class samples", "🔁 Randomly duplicate minority samples with replacement", "✅ Output: balanced dataset for training"],
            "params": "**random_state=42** — reproducibility. No other parameters.",
            "note": "Use before classification when classes are heavily imbalanced (e.g. fraud detection).",
        },
        "SMOTE": {
            "steps": ["📥 Imbalanced dataset", "🔍 For each minority sample, find K nearest minority neighbours", "🧬 Generate synthetic point along the line between sample and a neighbour", "✅ Output: richer balanced dataset"],
            "params": "**k_neighbors=5** — neighbours for synthesis; **random_state=42** — reproducibility.",
            "note": "Better than Random Oversampling — creates new data rather than duplicates. Needs ≥6 minority samples.",
        },
    }

    info = FLOWCHARTS.get(method)
    if not info:
        return

    with st.expander(f"📊 Sơ đồ thuật toán / Algorithm Flowchart — {method}", expanded=False):
        st.markdown("**Các bước hoạt động / How it works:**")
        flow_html = '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-bottom:1rem;">'
        for i, step in enumerate(info["steps"]):
            flow_html += f'<div style="background:#1a2332;border:1px solid #30363d;border-radius:8px;padding:6px 12px;color:#c9d1d9;font-size:0.82rem">{step}</div>'
            if i < len(info["steps"]) - 1:
                flow_html += '<div style="color:#58a6ff;font-size:1.1rem">→</div>'
        flow_html += '</div>'
        st.markdown(flow_html, unsafe_allow_html=True)
        st.markdown(f"**Tham số chính / Key parameters:** {info['params']}")
        st.info(f"💡 {info['note']}")


def fig_to_st(fig):
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING MODULE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def detect_data_problems(df_json: str) -> list[dict]:
    """Detect data quality issues. Cached so it only re-runs when data changes."""
    df = pd.read_json(df_json, orient="split")
    problems = []
    n = len(df)

    for col in df.columns:
        null_pct = df[col].isnull().mean()
        if null_pct > 0:
            sev = "err" if null_pct > 0.3 else "warn"
            dtype = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
            problems.append({
                "col": col, "type": "missing", "severity": sev,
                "msg": f'Column "{col}" has {null_pct:.1%} missing values ({int(null_pct*n)} rows)',
                "fix": f"Fill with {'mean' if dtype == 'numeric' else 'mode'}",
                "dtype": dtype,
            })

    for col in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            outlier_mask = (df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)
            n_out = outlier_mask.sum()
            if n_out > 0:
                problems.append({
                    "col": col, "type": "outlier", "severity": "warn",
                    "msg": f'Column "{col}" has {n_out} extreme outliers (3×IQR rule, {n_out/n:.1%} of rows)',
                    "fix": "Cap to 3×IQR (Winsorize)",
                    "dtype": "numeric",
                })

    text_cols = df.select_dtypes(include="object").columns.tolist()
    for col in text_cols:
        problems.append({
            "col": col, "type": "encoding", "severity": "info",
            "msg": f'Column "{col}" is text — needs encoding for ML models',
            "fix": "Label Encoding (auto-applied at training)",
            "dtype": "categorical",
        })

    dups = df.duplicated().sum()
    if dups > 0:
        problems.append({
            "col": "ALL", "type": "duplicate", "severity": "warn",
            "msg": f"{dups} duplicate rows found ({dups/n:.1%} of data)",
            "fix": "Remove duplicates",
            "dtype": "row",
        })

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        ranges = df[numeric_cols].max() - df[numeric_cols].min()
        if ranges.max() > 0 and (ranges.max() / (ranges.min() + 1e-9)) > 100:
            problems.append({
                "col": "NUMERIC", "type": "scale", "severity": "info",
                "msg": "Numeric columns have very different scales — scaling recommended",
                "fix": "StandardScaler (auto-applied at training)",
                "dtype": "numeric",
            })

    if not problems:
        problems.append({
            "col": "", "type": "ok", "severity": "ok",
            "msg": "No major data problems detected. Your dataset looks clean!",
            "fix": "",
            "dtype": "",
        })

    return problems


@st.cache_data(show_spinner=False)
def apply_fix_missing(df_json: str) -> tuple[str, str]:
    """Impute missing values. Returns (new_df_json, summary_message)."""
    df = pd.read_json(df_json, orient="split")
    before_nulls = df.isnull().sum().sum()
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val[0] if len(mode_val) else "Unknown")
    after_nulls = df.isnull().sum().sum()
    msg = f"Fixed {before_nulls - after_nulls} missing values (numeric → mean, text → mode)."
    return df.to_json(orient="split"), msg


@st.cache_data(show_spinner=False)
def apply_remove_duplicates(df_json: str) -> tuple[str, str]:
    """Remove duplicate rows. Returns (new_df_json, summary_message)."""
    df = pd.read_json(df_json, orient="split")
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    msg = f"Removed {before - after} duplicate rows. Before: {before} rows → After: {after} rows."
    return df.to_json(orient="split"), msg


@st.cache_data(show_spinner=False)
def apply_winsorize(df_json: str) -> tuple[str, str]:
    """Cap extreme outliers at 3×IQR. Returns (new_df_json, summary_message)."""
    df = pd.read_json(df_json, orient="split")
    capped = 0
    for col in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
            n_cap = ((df[col] < lo) | (df[col] > hi)).sum()
            df[col] = df[col].clip(lower=lo, upper=hi)
            capped += n_cap
    msg = f"Capped {capped} extreme outlier values across all numeric columns (3×IQR Winsorization)."
    return df.to_json(orient="split"), msg


def show_preprocessing_section(df_active: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """
    Renders the full Step 1 preprocessing UI.
    Returns the (possibly modified) DataFrame for downstream use.
    """
    st.markdown('<div class="section-header">🧹 Bước 1 / Step 1 — Làm sạch & Tiền xử lý / Data Cleaning & Preprocessing</div>',
                unsafe_allow_html=True)

    prep_key   = f"prep_df_{sheet_name}"
    prep_log   = f"prep_log_{sheet_name}"

    # Initialise cleaned copy in session state
    if prep_key not in st.session_state:
        st.session_state[prep_key] = df_active.to_json(orient="split")
    if prep_log not in st.session_state:
        st.session_state[prep_log] = []

    current_df_json = st.session_state[prep_key]

    # ── Problem detection ─────────────────────────────────────────────────────
    with st.spinner("Đang phát hiện vấn đề dữ liệu..."):
        problems = detect_data_problems(current_df_json)

    sev_icon = {"err": "🔴", "warn": "🟡", "info": "🔵", "ok": "✅"}
    chip_cls  = {"err": "chip-err", "warn": "chip-warn", "info": "chip-info", "ok": "chip-ok"}

    st.markdown("**🔍 Vấn đề phát hiện tự động / Auto-detected Problems:**")
    for p in problems:
        icon  = sev_icon.get(p["severity"], "ℹ️")
        chip  = f'<span class="{chip_cls.get(p["severity"], "chip-info")}">{p["type"].upper()}</span>'
        fix   = f'<span style="color:#8b949e;font-size:0.78rem"> → {p["fix"]}</span>' if p["fix"] else ""
        st.markdown(
            f'<div class="problem-row">{icon} {chip} <span>{p["msg"]}</span>{fix}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Action buttons ────────────────────────────────────────────────────────
    st.markdown("**🔧 Áp dụng sửa lỗi / Apply Fixes:**")
    fix_cols = st.columns(3)

    with fix_cols[0]:
        if st.button("🩹 Fix Missing Values\n(Mean / Mode)", key=f"fix_miss_{sheet_name}",
                     help="Numeric columns → filled with mean; text columns → filled with mode"):
            new_json, msg = apply_fix_missing(current_df_json)
            st.session_state[prep_key] = new_json
            st.session_state[prep_log].append(("🩹 Missing Values", msg))
            detect_data_problems.clear()
            st.rerun()

    with fix_cols[1]:
        if st.button("🗑️ Remove Duplicates", key=f"fix_dup_{sheet_name}",
                     help="Drops rows that are identical across all columns"):
            new_json, msg = apply_remove_duplicates(current_df_json)
            st.session_state[prep_key] = new_json
            st.session_state[prep_log].append(("🗑️ Duplicates", msg))
            detect_data_problems.clear()
            st.rerun()

    with fix_cols[2]:
        if st.button("📐 Cap Outliers\n(3×IQR Winsorize)", key=f"fix_out_{sheet_name}",
                     help="Clips extreme values to 3×IQR boundary — preserves all rows"):
            new_json, msg = apply_winsorize(current_df_json)
            st.session_state[prep_key] = new_json
            st.session_state[prep_log].append(("📐 Outliers", msg))
            detect_data_problems.clear()
            st.rerun()

    # ── Change log ────────────────────────────────────────────────────────────
    log = st.session_state[prep_log]
    if log:
        st.markdown("**📋 Nhật ký thay đổi / Change Log:**")
        for step_name, step_msg in log:
            st.markdown(
                f'<div style="background:#1a3a2a;border-left:3px solid #3fb950;'
                f'padding:6px 12px;border-radius:4px;margin-bottom:4px;color:#e6edf3;font-size:0.85rem">'
                f'<b style="color:#3fb950">{step_name}:</b> {step_msg}</div>',
                unsafe_allow_html=True,
            )

    # ── Before / After comparison ─────────────────────────────────────────────
    current_df = pd.read_json(current_df_json, orient="split")
    if log:
        st.markdown("**📊 Trước / Sau — Before / After Comparison:**")
        bcol, acol = st.columns(2)
        with bcol:
            st.markdown('<b style="color:#f85149">Before (original)</b>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:#1f1414;border:1px solid #f85149;border-radius:8px;'
                f'padding:8px 14px;color:#e6edf3;font-size:0.85rem">'
                f'📏 {df_active.shape[0]:,} rows × {df_active.shape[1]} cols<br>'
                f'❓ {df_active.isnull().sum().sum():,} missing values<br>'
                f'📋 {df_active.duplicated().sum():,} duplicate rows</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(df_active.head(5), use_container_width=True, height=180)
        with acol:
            st.markdown('<b style="color:#3fb950">After (cleaned)</b>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:#0d1f17;border:1px solid #3fb950;border-radius:8px;'
                f'padding:8px 14px;color:#e6edf3;font-size:0.85rem">'
                f'📏 {current_df.shape[0]:,} rows × {current_df.shape[1]} cols<br>'
                f'❓ {current_df.isnull().sum().sum():,} missing values<br>'
                f'📋 {current_df.duplicated().sum():,} duplicate rows</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(current_df.head(5), use_container_width=True, height=180)

        if st.button("↩️ Reset to Original", key=f"reset_prep_{sheet_name}"):
            del st.session_state[prep_key]
            del st.session_state[prep_log]
            detect_data_problems.clear()
            st.rerun()
    else:
        st.info("👆 No fixes applied yet. The original data will be used. Click a Fix button above to clean your data.")

    return current_df


# ─────────────────────────────────────────────────────────────────────────────
# ML runners
# ─────────────────────────────────────────────────────────────────────────────

def run_classification(method, df, target, features, test_size, balance, show_ui=True):
    if show_ui:
        st.markdown('<div class="section-header">⚙️ Training & Evaluation</div>', unsafe_allow_html=True)
    df_enc = encode_df(df[features + [target]].dropna())
    X = df_enc[features].values
    y = df_enc[target].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if balance == "Random Oversampling":
        X, y = RandomOverSampler(random_state=42).fit_resample(X, y)
        if show_ui:
            st.info("✅ Applied Random Oversampling.")
    elif balance == "SMOTE":
        try:
            X, y = SMOTE(random_state=42).fit_resample(X, y)
            if show_ui:
                st.info("✅ Applied SMOTE.")
        except Exception as e:
            if show_ui:
                st.warning(f"SMOTE skipped: {e}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)

    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                  roc_auc_score, roc_curve)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear Discriminant Analysis (LDA)": LinearDiscriminantAnalysis(),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
        "Classification Trees": DecisionTreeClassifier(max_depth=5),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine (SVM)": SVC(probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Neural Networks (MLP)": MLPClassifier(max_iter=500, random_state=42),
    }
    mdl = models[method]
    mdl.fit(X_tr, y_tr)
    y_pred = mdl.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    is_binary = len(np.unique(y)) == 2
    avg = "binary" if is_binary else "weighted"
    prec  = precision_score(y_te, y_pred, average=avg, zero_division=0)
    rec   = recall_score(y_te, y_pred, average=avg, zero_division=0)
    f1    = f1_score(y_te, y_pred, average=avg, zero_division=0)
    try:
        if is_binary:
            auc = roc_auc_score(y_te, mdl.predict_proba(X_te)[:, 1])
        else:
            auc = roc_auc_score(y_te, mdl.predict_proba(X_te), multi_class="ovr", average="weighted")
    except Exception:
        auc = None

    metrics = {
        "Method": method,
        "Accuracy": f"{acc:.4f}",
        "Precision": f"{prec:.4f}",
        "Recall": f"{rec:.4f}",
        "F1-Score": f"{f1:.4f}",
        "AUC": f"{auc:.4f}" if auc is not None else "N/A",
        "Train rows": len(X_tr),
        "Test rows": len(X_te),
    }

    if not show_ui:
        return metrics

    # ── Metrics row ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{acc:.2%}")
    m2.metric("F1-Score",  f"{f1:.4f}")
    m3.metric("AUC",       f"{auc:.4f}" if auc is not None else "N/A")
    m4.metric("Precision", f"{prec:.4f}")
    st.text(classification_report(y_te, y_pred))

    # ── Plot 1: Confusion Matrix ───────────────────────────────────────────────
    st.markdown("##### 📊 Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    cm = confusion_matrix(y_te, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                linewidths=0.5, linecolor='#30363d',
                annot_kws={"color": "#ffffff", "size": 12})
    ax.set_title("Confusion Matrix", color='#e6edf3')
    ax.set_xlabel("Predicted", color='#c9d1d9')
    ax.set_ylabel("Actual", color='#c9d1d9')
    ax.tick_params(colors='#c9d1d9')
    fig_to_st(fig)
    st.caption(
        "📖 How to read this: Dark blue = many predictions in that cell. "
        "A GOOD model has large numbers on the diagonal (top-left to bottom-right = correct predictions) "
        "and small numbers off-diagonal (= mistakes). "
        "A BAD model has large off-diagonal numbers meaning it confuses classes often."
    )

    # ── Plot 2a: ROC Curve (binary) OR Feature Importance (multi-class) ───────
    if is_binary and hasattr(mdl, "predict_proba"):
        st.markdown("##### 📈 ROC Curve")
        try:
            fpr, tpr, _ = roc_curve(y_te, mdl.predict_proba(X_te)[:, 1])
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            fig3.patch.set_facecolor('#0d1117')
            ax3.set_facecolor('#161b22')
            ax3.plot(fpr, tpr, color='#58a6ff', lw=2, label=f"AUC = {auc:.4f}" if auc else "ROC")
            ax3.plot([0, 1], [0, 1], 'r--', lw=1, label="Random baseline")
            ax3.set_xlabel("False Positive Rate", color='#c9d1d9')
            ax3.set_ylabel("True Positive Rate", color='#c9d1d9')
            ax3.set_title("ROC Curve", color='#e6edf3')
            ax3.tick_params(colors='#c9d1d9')
            ax3.legend(facecolor='#161b22', labelcolor='#c9d1d9')
            fig_to_st(fig3)
            st.caption(
                "📖 How to read this: The curve shows the trade-off between catching true positives and avoiding false positives. "
                "A GOOD model has a curve that bows strongly toward the top-left corner — AUC close to 1.0. "
                "The red dashed line is a random-guess baseline (AUC = 0.5). "
                "A BAD model sits close to the dashed line."
            )
        except Exception:
            pass

    # ── Plot 2b: Feature Importance / Coefficients ────────────────────────────
    if hasattr(mdl, "feature_importances_"):
        st.markdown("##### 🏅 Feature Importances")
        fi = pd.Series(mdl.feature_importances_, index=features).sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(7, max(3, len(features[:15]) * 0.35)))
        fig2.patch.set_facecolor('#0d1117')
        ax2.set_facecolor('#161b22')
        fi.head(15).plot(kind='barh', ax=ax2, color='#58a6ff')
        ax2.set_title("Feature Importances (top 15)", color='#e6edf3')
        ax2.tick_params(colors='#c9d1d9', labelcolor='#c9d1d9')
        ax2.invert_yaxis()
        plt.tight_layout()
        fig_to_st(fig2)
        st.caption(
            "📖 How to read this: Longer bars = more important features for prediction. "
            "Focus your attention (and domain expertise) on the top-ranked features. "
            "Very short bars may be safe to drop from the model to simplify it."
        )
    elif hasattr(mdl, "coef_"):
        st.markdown("##### 🏅 Coefficient Magnitudes")
        coef = pd.Series(np.abs(mdl.coef_[0]) if mdl.coef_.ndim > 1 else np.abs(mdl.coef_),
                         index=features).sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(7, max(3, len(features[:15]) * 0.35)))
        fig2.patch.set_facecolor('#0d1117')
        ax2.set_facecolor('#161b22')
        coef.head(15).plot(kind='barh', ax=ax2, color='#bc8cff')
        ax2.set_title("Coefficient Magnitudes (top 15)", color='#e6edf3')
        ax2.tick_params(colors='#c9d1d9', labelcolor='#c9d1d9')
        ax2.invert_yaxis()
        plt.tight_layout()
        fig_to_st(fig2)
        st.caption(
            "📖 How to read this: Longer bars = stronger influence on the predicted class. "
            "Note this shows absolute magnitude — a feature can push the prediction in either direction. "
            "Positive coefficients push toward class 1; negative push toward class 0."
        )

    if method == "Classification Trees":
        with st.expander("🌿 Decision Tree Rules (text)", expanded=False):
            st.code(export_text(mdl, feature_names=features, max_depth=4), language="")

    return metrics


def run_regression(method, df, target, features, test_size):
    st.markdown('<div class="section-header">⚙️ Training & Evaluation</div>', unsafe_allow_html=True)
    df_enc = encode_df(df[features + [target]].dropna())
    X = df_enc[features].values
    y = df_enc[target].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)

    if method == "Linear Regression":
        mdl = LinearRegression()
    else:
        mdl = MLPRegressor(max_iter=500, random_state=42)

    mdl.fit(X_tr, y_tr)
    y_pred = mdl.predict(X_te)
    mse = mean_squared_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)
    residuals = y_te - y_pred

    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
        st.metric("RMSE", f"{np.sqrt(mse):.4f}")
    with col2:
        if r2 >= 0.7:
            st.success(f"R² = {r2:.4f} — Good fit! The model explains {r2:.1%} of variance.")
        elif r2 >= 0.4:
            st.warning(f"R² = {r2:.4f} — Moderate fit. Consider adding more features.")
        else:
            st.error(f"R² = {r2:.4f} — Weak fit. The model struggles to explain the target.")

    # ── Plot 1: Actual vs Predicted ───────────────────────────────────────────
    st.markdown("##### 📊 Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    ax.scatter(y_te, y_pred, alpha=0.5, color='#58a6ff', edgecolors='none', s=20)
    mn, mx = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual", color='#c9d1d9')
    ax.set_ylabel("Predicted", color='#c9d1d9')
    ax.set_title("Actual vs Predicted", color='#e6edf3')
    ax.tick_params(colors='#c9d1d9')
    ax.legend(facecolor='#161b22', labelcolor='#c9d1d9', fontsize=8)
    fig_to_st(fig)
    st.caption(
        "📖 How to read this: Each dot is one test sample. "
        "A GOOD model has dots clustering tightly along the red dashed line (predicted ≈ actual). "
        "A BAD model has dots scattered widely above or below the line — "
        "dots above mean the model under-predicts; dots below mean it over-predicts."
    )

    # ── Plot 2: Residual Plot ─────────────────────────────────────────────────
    st.markdown("##### 📉 Residual Plot")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    fig2.patch.set_facecolor('#0d1117')
    ax2.set_facecolor('#161b22')
    ax2.scatter(y_pred, residuals, alpha=0.5, color='#bc8cff', edgecolors='none', s=20)
    ax2.axhline(0, color='#f85149', linestyle='--', lw=1.5)
    ax2.set_xlabel("Predicted Value", color='#c9d1d9')
    ax2.set_ylabel("Residual (Actual − Predicted)", color='#c9d1d9')
    ax2.set_title("Residual Plot", color='#e6edf3')
    ax2.tick_params(colors='#c9d1d9')
    fig_to_st(fig2)
    st.caption(
        "📖 How to read this: Residuals are prediction errors (actual − predicted). "
        "A GOOD model has dots randomly scattered above and below the red zero line — no clear pattern. "
        "A BAD model shows a funnel shape (heteroscedasticity), a curve, or a systematic band "
        "which means the model is missing something important."
    )

    if method == "Linear Regression":
        coef = pd.Series(mdl.coef_, index=features).sort_values(key=abs, ascending=False)
        st.subheader("Coefficients")
        st.dataframe(coef.reset_index().rename(columns={"index": "Feature", 0: "Coefficient"}),
                     use_container_width=True)


def run_association(df, min_support, min_confidence, min_lift):
    st.markdown('<div class="section-header">⚙️ Association Rules</div>', unsafe_allow_html=True)
    # Try to detect transaction-style data
    records = []
    for _, row in df.iterrows():
        items = [str(v).strip() for v in row.dropna().values if str(v).strip()]
        if items:
            records.append(items)

    if not records:
        st.error("Could not parse transactional data.")
        return

    te = TransactionEncoder()
    te_arr = te.fit_transform(records)
    df_bool = pd.DataFrame(te_arr, columns=te.columns_)
    freq = apriori(df_bool, min_support=min_support, use_colnames=True)
    if freq.empty:
        st.warning("No frequent itemsets found. Try lowering min support.")
        return
    rules = association_rules(freq, metric="lift", min_threshold=min_lift)
    rules = rules[rules["confidence"] >= min_confidence].sort_values("lift", ascending=False)
    st.success(f"Found **{len(rules)}** rules from **{len(freq)}** frequent itemsets.")

    st.subheader("Top Rules")
    display = rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(20).copy()
    display["antecedents"] = display["antecedents"].apply(lambda x: ", ".join(list(x)))
    display["consequents"] = display["consequents"].apply(lambda x: ", ".join(list(x)))
    st.dataframe(display, use_container_width=True)

    if not rules.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        sc = ax.scatter(rules["support"], rules["confidence"], c=rules["lift"],
                        cmap="plasma", alpha=0.8, s=60)
        plt.colorbar(sc, ax=ax, label="Lift")
        ax.set_xlabel("Support", color='#8b949e')
        ax.set_ylabel("Confidence", color='#8b949e')
        ax.set_title("Support vs Confidence (colour = Lift)", color='#e6edf3')
        ax.tick_params(colors='#8b949e')
        fig_to_st(fig)


def run_clustering(method, df, features, n_clusters):
    st.markdown('<div class="section-header">⚙️ Clustering</div>', unsafe_allow_html=True)
    from sklearn.decomposition import PCA
    df_enc = encode_df(df[features].dropna())
    X = StandardScaler().fit_transform(df_enc.values)

    if method == "K-Means Clustering":
        # ── Plot 1: Elbow Curve ────────────────────────────────────────────────
        st.markdown("##### 📊 Elbow Curve (choose best K)")
        mdl = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = mdl.fit_predict(X)
        inertia_vals = []
        k_range = range(2, min(11, len(X)))
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertia_vals.append(km.inertia_)
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        ax.plot(list(k_range), inertia_vals, 'o-', color='#58a6ff', lw=2)
        ax.axvline(n_clusters, color='#f778ba', linestyle='--', lw=1.5, label=f"Current K={n_clusters}")
        ax.set_title("Elbow Curve", color='#e6edf3')
        ax.set_xlabel("Number of Clusters (K)", color='#c9d1d9')
        ax.set_ylabel("Inertia (WCSS)", color='#c9d1d9')
        ax.tick_params(colors='#c9d1d9')
        ax.legend(facecolor='#161b22', labelcolor='#c9d1d9')
        fig_to_st(fig)
        st.caption(
            "📖 How to read this: The y-axis shows total within-cluster variance (lower = tighter clusters). "
            "Look for the 'elbow' — the point where the curve bends and stops dropping steeply. "
            "That K value is usually the optimal number of clusters. "
            "The pink dashed line shows your currently selected K."
        )
    else:
        mdl = AgglomerativeClustering(n_clusters=n_clusters)
        labels = mdl.fit_predict(X)
        # ── Plot 1: Dendrogram ─────────────────────────────────────────────────
        st.markdown("##### 🌳 Dendrogram")
        linked = linkage(X[:min(200, len(X))], method='ward')
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        dendrogram(linked, ax=ax, color_threshold=0,
                   above_threshold_color='#58a6ff', leaf_font_size=6)
        ax.set_title("Dendrogram (sample of 200 rows)", color='#e6edf3')
        ax.tick_params(colors='#8b949e')
        plt.tight_layout()
        fig_to_st(fig)
        st.caption(
            "📖 How to read this: Each leaf at the bottom is a data point. "
            "Branches merge from the bottom up — points that are most similar merge first (lowest lines). "
            "The height of a merge represents how different the two groups were. "
            "To choose K clusters, draw a horizontal line across the diagram — the number of vertical lines it crosses = K."
        )

    # ── Silhouette Score ──────────────────────────────────────────────────────
    try:
        sil = silhouette_score(X, labels)
        sil_col, _ = st.columns([1, 2])
        with sil_col:
            st.metric("Silhouette Score", f"{sil:.4f}",
                      help="Range: -1 to +1. Above 0.5 = good separation. Above 0.7 = strong clusters.")
        if sil > 0.7:
            st.success(f"Silhouette = {sil:.4f} — Excellent cluster separation!")
        elif sil > 0.5:
            st.info(f"Silhouette = {sil:.4f} — Good cluster separation.")
        elif sil > 0.25:
            st.warning(f"Silhouette = {sil:.4f} — Moderate separation. Try a different K.")
        else:
            st.error(f"Silhouette = {sil:.4f} — Weak separation. Clusters may overlap significantly.")
    except Exception:
        pass

    # ── Plot 2: PCA 2D Scatter ────────────────────────────────────────────────
    st.markdown("##### 🎨 2D PCA Cluster Scatter")
    st.caption(
        f"Note: {len(features)} features compressed into 2 dimensions using PCA for visualization. "
        "Some information is lost — this is an approximation."
    )
    if X.shape[1] >= 2:
        n_comp = min(2, X.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        X_2d = pca.fit_transform(X)
        var_explained = pca.explained_variance_ratio_.sum() * 100

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        fig2.patch.set_facecolor('#0d1117')
        ax2.set_facecolor('#161b22')
        palette = plt.cm.tab10.colors
        for c in np.unique(labels):
            mask = labels == c
            ax2.scatter(X_2d[mask, 0], X_2d[mask, 1] if n_comp > 1 else np.zeros(mask.sum()),
                        color=palette[c % 10], label=f"Cluster {c}",
                        alpha=0.7, s=25, edgecolors='none')
        ax2.legend(facecolor='#161b22', labelcolor='#c9d1d9', fontsize=8)
        ax2.set_title(f"Clusters in 2D PCA Space ({var_explained:.1f}% variance explained)",
                      color='#e6edf3')
        ax2.set_xlabel("PCA Component 1", color='#c9d1d9')
        ax2.set_ylabel("PCA Component 2", color='#c9d1d9')
        ax2.tick_params(colors='#c9d1d9')
        plt.tight_layout()
        fig_to_st(fig2)
        st.caption(
            "📖 How to read this: Each dot is one data row, coloured by its assigned cluster. "
            "A GOOD clustering result shows clearly separated blobs of colour with little overlap. "
            "A BAD result has colours mixed together — the clusters are not well-defined. "
            f"(This 2D view captures {var_explained:.1f}% of the total variance in your {len(features)}-feature data.)"
        )

    df_out = df[features].copy()
    df_out["Cluster"] = labels
    with st.expander("📋 Cluster Assignments (first 50 rows)", expanded=False):
        st.dataframe(df_out.head(50), use_container_width=True)


def run_balancing(method, df, target, features):
    st.markdown('<div class="section-header">⚙️ Class Balancing</div>', unsafe_allow_html=True)
    df_enc = encode_df(df[features + [target]].dropna())
    X = df_enc[features].values
    y = df_enc[target].values

    orig_counts = pd.Series(y).value_counts()
    if method == "Random Oversampling":
        Xr, yr = RandomOverSampler(random_state=42).fit_resample(X, y)
    else:
        try:
            Xr, yr = SMOTE(random_state=42).fit_resample(X, y)
        except Exception as e:
            st.error(f"SMOTE failed: {e}")
            return
    new_counts = pd.Series(yr).value_counts()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Before")
        st.bar_chart(orig_counts)
    with col2:
        st.subheader("After")
        st.bar_chart(new_counts)

    st.success(f"Samples: {len(y)} → {len(yr)}")
    st.info("💡 Use the balanced dataset as input to a Classification method above.")


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in {
    "sheets": {},
    "active_sheet": None,
    "ai_suggestion": "",
    "chosen_method": None,
    "selected_methods": [],
    "comparison_results": [],
    "step": 1,
    "gemini_key": "",
    "openrouter_key": "",
    "ai_vn": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">📁 Data Upload</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Tải lên CSV, Excel, JSON hoặc TXT / Upload CSV, Excel, JSON, or TXT",
        accept_multiple_files=True,
        type=["csv", "xlsx", "xls", "json", "txt"],
    )

    if uploaded_files:
        all_sheets = {}
        for uf in uploaded_files:
            loaded = load_file(uf)
            for sh, df in loaded.items():
                key = f"{uf.name} › {sh}" if len(loaded) > 1 else uf.name
                all_sheets[key] = df
        st.session_state["sheets"] = all_sheets

        st.markdown('<p class="section-header">📊 Select Dataset</p>', unsafe_allow_html=True)
        chosen = st.selectbox("Bộ dữ liệu hiện tại / Active dataset", list(all_sheets.keys()))
        st.session_state["active_sheet"] = chosen

        df_active = all_sheets[chosen]
        st.markdown(f'<div class="card"><b style="color:#58a6ff">{chosen}</b><br>'
                    f'<span style="color:#8b949e">{df_active.shape[0]} rows × {df_active.shape[1]} cols</span></div>',
                    unsafe_allow_html=True)

        if len(all_sheets) > 1:
            st.markdown('<p class="section-header">🔗 Merge Datasets</p>', unsafe_allow_html=True)
            merge_on = st.text_input("Cột khóa chung (để gộp) / Common key column", "")
            if st.button("Tự động gộp tất cả / Auto-merge all") and merge_on:
                merged = None
                for df in all_sheets.values():
                    if merge_on in df.columns:
                        merged = df if merged is None else pd.merge(merged, df, on=merge_on, how="outer")
                if merged is not None:
                    st.session_state["sheets"]["🔗 Merged"] = merged
                    st.success(f"Merged → {merged.shape}")

    st.markdown("---")
    st.markdown('<p class="section-header">🔑 AI API Keys</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8b949e;font-size:0.78rem">'
        'Nhập ít nhất một khóa để bật phân tích AI. '
        'All ML methods work without a key.</p>',
        unsafe_allow_html=True,
    )

    # Use key= so Streamlit binds directly to session_state — no value= needed.
    # Pre-populate session state with defaults BEFORE the widget renders.
    if "gemini_key" not in st.session_state:
        st.session_state["gemini_key"] = ""
    if "openrouter_key" not in st.session_state:
        st.session_state["openrouter_key"] = ""

    st.text_input(
        "Gemini API Key",
        key="gemini_key",          # directly syncs with st.session_state["gemini_key"]
        type="password",
        placeholder="AIzaSy...",
        help="Get a free key at aistudio.google.com (15 req/min free tier)",
    )
    st.text_input(
        "Khóa OpenRouter API (miễn phí) / OpenRouter API Key",
        key="openrouter_key",      # directly syncs with st.session_state["openrouter_key"]
        type="password",
        placeholder="sk-or-...",
        help="100% free at openrouter.ai — no billing or card needed",
    )

    g_set = bool(st.session_state.get("gemini_key", "").strip())
    o_set = bool(st.session_state.get("openrouter_key", "").strip())
    if g_set and o_set:
        st.success("✅ Both keys set (Gemini + OpenRouter fallback)")
    elif g_set:
        st.info("🔵 Gemini key set")
    elif o_set:
        st.info("🟣 OpenRouter key set")
    else:
        st.warning("⚠️ No AI key — paste one above and press Enter")

    st.markdown("---")
    st.markdown('<p style="color:#8b949e;font-size:0.75rem;text-align:center;">DataMine AI · Powered by Gemini + sklearn</p>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 DataMine AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Tải dữ liệu lên · Mô tả mục tiêu · Để AI hướng dẫn phân tích</div>'
    '<div style="color:#6e7681;font-size:0.85rem;margin-bottom:1rem">'
    'Upload your data · Describe your goal · Let AI guide your analysis</div>',
    unsafe_allow_html=True,
)

if not st.session_state["sheets"]:
    st.markdown("""
    <div class="card card-accent">
    <b style="color:#58a6ff">👋 Chào mừng! / Welcome!</b><br><br>
    <ol style="color:#c9d1d9;line-height:2">
      <li>Tải lên một hoặc nhiều tệp dữ liệu ở thanh bên (CSV, Excel, JSON, TXT).<br><small>Upload one or more data files in the sidebar.</small></li>
      <li>Mô tả mục tiêu của bạn — AI sẽ đề xuất phương pháp phù hợp.<br><small>Describe your goal — AI will suggest a method.</small></li>
      <li>Cấu hình tham số và chạy kỹ thuật đã chọn.<br><small>Configure parameters and run the chosen technique.</small></li>
      <li>Xem kết quả, biểu đồ và giải thích từ AI.<br><small>View results, charts, and AI interpretation.</small></li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df_active_raw = st.session_state["sheets"][st.session_state["active_sheet"]]
active_sheet_name = st.session_state["active_sheet"]

# ── Data Preview ──────────────────────────────────────────────────────────────
with st.expander("🔍 Xem trước Dữ liệu / Data Preview & Profile", expanded=False):
    tab1, tab2, tab3 = st.tabs(["Table", "Statistics", "Column Types"])
    with tab1:
        st.dataframe(df_active_raw.head(50), use_container_width=True)
    with tab2:
        st.dataframe(df_active_raw.describe(include="all"), use_container_width=True)
    with tab3:
        dtypes = df_active_raw.dtypes.reset_index()
        dtypes.columns = ["Column", "Type"]
        dtypes["Nulls"] = df_active_raw.isnull().sum().values
        dtypes["Unique"] = df_active_raw.nunique().values
        st.dataframe(dtypes, use_container_width=True)

st.divider()

# ── Step 1 – Data Cleaning & Preprocessing ────────────────────────────────────
df_active = show_preprocessing_section(df_active_raw, active_sheet_name)

st.divider()

# ── Step 2 – AI Goal Understanding ───────────────────────────────────────────
st.markdown('<div class="section-header">🤖 Bước 2 / Step 2 — Mô tả Mục tiêu / Describe Your Goal</div>', unsafe_allow_html=True)

user_goal = st.text_area(
    "Bạn muốn đạt được điều gì? / What do you want to achieve? (in any language)",
    placeholder="e.g. 'I want to predict customer churn', 'Find which products are bought together', "
                "'Segment customers into groups', 'Classify emails as spam or not'…",
    height=80,
)

# Show key status warning inline if neither key is set
_g_key = st.session_state.get("gemini_key", "").strip()
_o_key = st.session_state.get("openrouter_key", "").strip()
if not _g_key and not _o_key:
    st.warning(
        "⚠️ **No AI key set.** Paste your Gemini or OpenRouter key in the sidebar to enable this step. "
        "You can skip this and go straight to choosing a method below.",
        icon="🔑",
    )

if st.button("🔎 Phân tích Mục tiêu với AI / Analyse Goal with AI"):
    with st.spinner("AI đang đọc dữ liệu và mục tiêu của bạn..."):
        # Use multi-sheet analysis if multiple datasets loaded, else single
        all_sheets_loaded = st.session_state.get("sheets", {})
        if len(all_sheets_loaded) > 1:
            st.session_state["ai_suggestion"] = ask_gemini_multisheet(all_sheets_loaded, user_goal)
        else:
            summary = df_summary(df_active)
            prompt = f"""You are a data mining expert assistant.

DATASET SUMMARY:
{summary}

USER GOAL:
{user_goal}

Tasks:
1. Identify the user's analytical purpose (classification, prediction/regression, association rules, clustering, class balancing, or a combination).
2. List the 2-3 most suitable data mining methods from this list: {list(METHODS.keys())}
   Give the method name EXACTLY as written above, then a brief reason.
3. Identify the most likely TARGET column (for supervised methods) and the best FEATURE columns. Reference actual column names.
4. Point out data quality issues (missing values, imbalanced classes, wrong types, scaling needed).
5. Suggest preprocessing steps.
6. If this looks like fraud detection, recommend SMOTE to handle class imbalance.

IMPORTANT FORMATTING RULES: Do NOT use **, *, #, or bullet - symbols. Write plain numbered paragraphs only.
End with a SHORT SUMMARY of the top 2-3 recommended methods.
Respond in the same language the user used."""
            st.session_state["ai_suggestion"] = ask_gemini(prompt)

if st.session_state["ai_suggestion"]:
    raw = st.session_state["ai_suggestion"]

    # ── Clean up markdown symbols so they read as plain prose ────────────────
    import re as _re
    # Remove bold/italic markers
    clean = _re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", raw)
    # Remove heading hashes
    clean = _re.sub(r"^#+\s*", "", clean, flags=_re.MULTILINE)
    # Remove bullet dashes that start lines
    clean = _re.sub(r"^[-•]\s+", "  ", clean, flags=_re.MULTILINE)
    # Collapse triple+ newlines
    clean = _re.sub(r"\n{3,}", "\n\n", clean)

    with st.expander("🤖 Phân tích AI (nhấn để mở rộng / click to expand)", expanded=True):
        col_lang1, col_lang2 = st.columns([1, 1])
        with col_lang1:
            st.markdown("**English Analysis**")
            st.text(clean[:3000] + ("..." if len(clean) > 3000 else ""))
        with col_lang2:
            st.markdown("**Phân tích (Tiếng Việt)**")
            if st.button("Dịch sang Tiếng Việt", key="translate_btn"):
                with st.spinner("Đang dịch..."):
                    vn_prompt = f"""Translate the following data mining analysis into clear, natural Vietnamese.
Keep the structure and all technical terms (Random Forest, SMOTE, Logistic Regression, etc.) in English but explain them in Vietnamese.
Remove all markdown symbols like *, #, ** from both input and output.
Write in plain numbered paragraphs, no bullet symbols.

Text to translate:
{clean[:3000]}"""
                    st.session_state["ai_vn"] = ask_gemini(vn_prompt)
            if st.session_state.get("ai_vn"):
                vn_text = st.session_state["ai_vn"]
                vn_clean = _re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", vn_text)
                vn_clean = _re.sub(r"^#+\s*", "", vn_clean, flags=_re.MULTILINE)
                vn_clean = _re.sub(r"^[-•]\s+", "  ", vn_clean, flags=_re.MULTILINE)
                st.text(vn_clean)
            else:
                st.info("Nhấn nút phía trên để dịch sang Tiếng Việt")

    # ── AI Quick Recommendation Summary ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Hướng dẫn nhanh / Quick Recommendation")
    st.markdown(
        "Dựa trên phân tích AI, hãy chọn một trong các phương pháp phù hợp nhất dưới đây "
        "(Based on the AI analysis, select the most suitable method below):"
    )

    # Parse suggested methods from the raw AI text
    suggested = []
    for m_name in METHODS.keys():
        if m_name.lower() in raw.lower():
            suggested.append(m_name)
    # Limit to top 4
    suggested = suggested[:4]

    if suggested:
        rec_cols = st.columns(len(suggested))
        for idx, m_name in enumerate(suggested):
            with rec_cols[idx]:
                m_meta = METHODS[m_name]
                badge_color = {"classification": "#1f3a5f", "prediction": "#2d1f5f", "association": "#3d1f35"}[m_meta["group"]]
                text_color  = {"classification": "#58a6ff", "prediction": "#bc8cff", "association": "#f778ba"}[m_meta["group"]]
                st.markdown(
                    f'<div style="background:{badge_color};border-radius:8px;padding:0.7rem;text-align:center;">' +
                    f'<b style="color:{text_color}">{m_name}</b><br>' +
                    f'<small style="color:#c9d1d9">{m_meta["vn"]}</small></div>',
                    unsafe_allow_html=True,
                )
                if st.button(f"Chọn / Select", key=f"rec_{m_name}"):
                    st.session_state["chosen_method"] = m_name
                    st.rerun()
    else:
        st.info("Không phát hiện phương pháp cụ thể. Hãy chọn từ danh sách phía dưới.")

st.divider()

# ── Step 3 – Method Selection ─────────────────────────────────────────────────
st.markdown('<div class="section-header">🛠️ Bước 3 / Step 3 — Chọn Phương pháp / Choose a Method</div>', unsafe_allow_html=True)

for group_id, gmeta in GROUP_META.items():
    st.markdown(f"**{gmeta['icon']} {gmeta['label']}**")
    cols = st.columns(3)
    methods_in_group = [(n, m) for n, m in METHODS.items() if m["group"] == group_id]
    for i, (name, meta) in enumerate(methods_in_group):
        with cols[i % 3]:
            selected_single  = st.session_state["chosen_method"] == name
            selected_multi   = name in st.session_state["selected_methods"]
            border = "2px solid #58a6ff" if selected_single else ("2px solid #3fb950" if selected_multi else "1px solid #30363d")
            bg     = "#1a2332" if selected_single else ("#1a3a2a" if selected_multi else "#161b22")
            st.markdown(
                f'<div style="background:{bg};border:{border};border-radius:10px;'
                f'padding:0.8rem;margin-bottom:0.6rem;">'
                f'<span class="badge {meta["badge"]}">{group_id.upper()}</span><br>'
                f'<b style="color:#e6edf3">{name}</b><br>'
                f'<small style="color:#8b949e">{meta["vn"]}</small><br>'
                f'<small style="color:#c9d1d9;font-size:0.8rem;line-height:1.5;display:block;margin-top:4px">{meta["desc"]}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )
            bc1, bc2 = st.columns(2)
            with bc1:
                if st.button("▶ Chọn", key=f"sel_{name}"):
                    st.session_state["chosen_method"] = name
                    st.rerun()
            with bc2:
                if selected_multi:
                    if st.button("✓ So sánh", key=f"cmp_{name}", help="Bỏ khỏi danh sách so sánh"):
                        st.session_state["selected_methods"].remove(name)
                        st.rerun()
                else:
                    if st.button("＋ So sánh", key=f"cmp_{name}", help="Thêm vào bảng so sánh"):
                        st.session_state["selected_methods"].append(name)
                        st.rerun()
    st.markdown("")

# Show comparison badge
if st.session_state["selected_methods"]:
    st.markdown(
        f'<div style="background:#1a3a2a;border:1px solid #3fb950;border-radius:8px;padding:0.6rem 1rem;color:#3fb950;">'
        f'📊 <b>So sánh đa phương pháp / Multi-method comparison:</b> '
        + " · ".join([f"<span style='background:#0d2b1a;padding:2px 8px;border-radius:4px'>{m}</span>" for m in st.session_state["selected_methods"]])
        + f' &nbsp;<small>(Chạy từng phương pháp để xem bảng so sánh / Run each method to populate the table)</small></div>',
        unsafe_allow_html=True,
    )
    if st.button("🗑️ Xóa danh sách so sánh / Clear comparison list"):
        st.session_state["selected_methods"] = []
        st.session_state["comparison_results"] = []
        st.rerun()

st.divider()

# ── Step 4 – Configure & Run ──────────────────────────────────────────────────
method = st.session_state["chosen_method"]
if not method:
    st.info("👆 Hãy chọn một phương pháp ở trên để cấu hình và chạy / Select a method above.")
    st.stop()

st.markdown(f'<div class="section-header">⚡ Bước 4 / Step 4 — Cấu hình & Chạy / Configure & Run: {method}</div>',
            unsafe_allow_html=True)

meta = METHODS[method]
st.markdown(f'<div class="card"><span class="badge {meta["badge"]}">{meta["group"].upper()}</span> '
            f'<b>{method}</b> — {meta["desc"]}</div>', unsafe_allow_html=True)

numeric_cols = df_active.select_dtypes(include=[np.number]).columns.tolist()
all_cols = df_active.columns.tolist()

group = meta["group"]

# ── Show algorithm flowchart ─────────────────────────────────────────────────
show_algorithm_flowchart(method)

# ── Classification & Prediction shared config ────────────────────────────────
if group in ("classification", "prediction") or method in ("Random Oversampling", "SMOTE"):
    col_a, col_b = st.columns(2)
    with col_a:
        # Auto-detect target column
        auto_target = suggest_target_column(df_active)
        default_idx = all_cols.index(auto_target) if auto_target and auto_target in all_cols else 0
        if auto_target:
            st.markdown(
                f'<small style="color:#3fb950">✅ Cột mục tiêu đề xuất tự động: <b>{auto_target}</b></small>',
                unsafe_allow_html=True,
            )
        target_col = st.selectbox("🎯 Cột mục tiêu / Target column", all_cols, index=default_idx)
    with col_b:
        feature_cols = st.multiselect(
            "📐 Các cột đặc trưng / Feature columns",
            [c for c in all_cols if c != target_col],
            default=[c for c in numeric_cols if c != target_col][:8],
        )

if group == "classification":
    test_size = st.slider(
        "Tỷ lệ kiểm tra % / Test split %  ℹ️  (Phần dữ liệu dùng để đánh giá — không dùng để huấn luyện. "
        "Ví dụ: 20% = 80% train, 20% test để đo độ chính xác thực tế.)",
        10, 40, 20
    ) / 100
    balance_opt = st.selectbox("Cân bằng lớp (tùy chọn) / Class balancing (optional)",
                               ["None", "Random Oversampling", "SMOTE"])
elif group == "prediction" and method != "Neural Networks Regression (MLP)":
    test_size = st.slider(
        "Test split %  ℹ️  (Percentage of data held out for evaluation — not used in training. "
        "E.g. 20% = model trains on 80%, is evaluated on the remaining 20%.)",
        10, 40, 20
    ) / 100
elif group == "prediction":
    test_size = st.slider(
        "Test split %  ℹ️  (Percentage of data held out for evaluation — not used in training.)",
        10, 40, 20
    ) / 100

if method == "Association Rules (Apriori)":
    c1, c2, c3 = st.columns(3)
    with c1: min_sup = st.slider("Min Support", 0.01, 0.5, 0.05, 0.01)
    with c2: min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05)
    with c3: min_lift = st.slider("Min Lift", 1.0, 10.0, 1.0, 0.1)

if method in ("K-Means Clustering", "Hierarchical Clustering"):
    c1, c2 = st.columns(2)
    with c1:
        feature_cols = st.multiselect("📐 Feature columns", numeric_cols, default=numeric_cols[:6])
    with c2:
        n_clusters = st.slider("Number of clusters (K)", 2, 10, 3)

if st.button(f"🚀 Run {method}", type="primary"):
    result_metrics = None
    if group == "classification" and method not in ("Random Oversampling", "SMOTE"):
        if not feature_cols:
            st.error("Select at least one feature column.")
        else:
            result_metrics = run_classification(method, df_active, target_col, feature_cols, test_size,
                               balance_opt if 'balance_opt' in dir() else "None")

    elif group == "prediction":
        if not feature_cols:
            st.error("Select at least one feature column.")
        else:
            run_regression(method, df_active, target_col, feature_cols, test_size)

    elif method == "Association Rules (Apriori)":
        run_association(df_active, min_sup, min_conf, min_lift)

    elif method in ("K-Means Clustering", "Hierarchical Clustering"):
        if not feature_cols:
            st.error("Select at least one feature column.")
        else:
            run_clustering(method, df_active, feature_cols, n_clusters)

    elif method in ("Random Oversampling", "SMOTE"):
        if not feature_cols:
            st.error("Select at least one feature column.")
        else:
            run_balancing(method, df_active, target_col, feature_cols)

    # ── Store comparison metrics if method is in selected_methods ────────────
    if result_metrics and method in st.session_state["selected_methods"]:
        result_metrics["Sheet"] = active_sheet_name
        # Use method + sheet as composite key so same method on different sheets both appear
        existing = [r for r in st.session_state["comparison_results"]
                    if not (r["Method"] == method and r.get("Sheet") == active_sheet_name)]
        existing.append(result_metrics)
        st.session_state["comparison_results"] = existing
        st.success(f"✅ Added {method} ({active_sheet_name}) to comparison table.")

    # ── AI interpretation ─────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">🤖 AI Result Interpretation</div>',
                unsafe_allow_html=True)

    metrics_context = ""
    if result_metrics:
        metrics_context = f"""
PERFORMANCE METRICS (calculated from this run):
  Accuracy:      {result_metrics['Accuracy']}
  Precision:     {result_metrics['Precision']}
  Recall:        {result_metrics['Recall']}
  F1-Score:      {result_metrics['F1-Score']}
  AUC:           {result_metrics['AUC']}
  Training rows: {result_metrics['Train rows']}
  Test rows:     {result_metrics['Test rows']}
"""

    with st.spinner("AI đang phân tích kết quả..."):
        interp_prompt = f"""You are a data mining expert providing a post-run analysis report.

METHOD USED: {method}

DATASET SUMMARY:
{df_summary(df_active)}

USER GOAL: {user_goal or '(not specified)'}
{metrics_context}
Your analysis must cover ALL of the following points:

1. Plain-language explanation of what the results mean given the metrics above.
2. Performance assessment: Is Accuracy / F1 / AUC strong or weak? Flag any AUC below 0.7 clearly.
3. Limitations of {method} specific to this dataset — for example: independence assumptions (Naive Bayes, Logistic Regression), interpretability constraints, sensitivity to class imbalance, overfitting risk.
4. Feature importance: if coefficients or importances were computed, discuss which features appear most influential.
5. Class imbalance: if one class is much rarer (e.g. fraud detection), recommend SMOTE.
6. Concrete next steps: what should the user do after this result?
7. One or two alternative methods and why they might perform better here.

MANDATORY FORMATTING RULES — violations will make the output unusable:
- Do NOT use asterisks, hash symbols, backticks, dashes, or any markdown symbols.
- Do NOT start lines with - or * or # or bullet points.
- Write ONLY in plain numbered paragraphs.
- No special characters of any kind.
- Maximum 400 words total.
Respond in the same language the user used (default English)."""

        interp = ask_gemini(interp_prompt)

    import re as _re2
    interp_clean = _re2.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", interp)
    interp_clean = _re2.sub(r"^#+\s*", "", interp_clean, flags=_re2.MULTILINE)
    interp_clean = _re2.sub(r"^[-•]\s+", "  ", interp_clean, flags=_re2.MULTILINE)
    interp_clean = _re2.sub(r"`{1,3}", "", interp_clean)
    interp_clean = _re2.sub(r"\n{3,}", "\n\n", interp_clean)

    st.markdown('<div class="ai-bubble">🤖 <b style="color:#58a6ff">AI Interpretation</b></div>',
                unsafe_allow_html=True)
    st.text(interp_clean)

# ── Comparison Table (always visible when populated) ─────────────────────────
if st.session_state.get("comparison_results"):
    st.divider()
    st.markdown('<div class="section-header">📊 Bảng So sánh Phương pháp / Method Comparison Table</div>',
                unsafe_allow_html=True)
    cmp_df = pd.DataFrame(st.session_state["comparison_results"])
    # Reorder columns — Sheet first for cross-sheet clarity
    col_order = ["Sheet", "Method", "Accuracy", "Precision", "Recall", "F1-Score", "AUC", "Train rows", "Test rows"]
    col_order = [c for c in col_order if c in cmp_df.columns]
    cmp_df = cmp_df[col_order]

    # Label rows as "Sheet — Method" for the index
    if "Sheet" in cmp_df.columns and "Method" in cmp_df.columns:
        cmp_df.index = cmp_df["Sheet"].str[:20] + " › " + cmp_df["Method"]
        display_df = cmp_df.drop(columns=["Sheet", "Method"])
    else:
        display_df = cmp_df.set_index("Method") if "Method" in cmp_df.columns else cmp_df

    st.dataframe(display_df, use_container_width=True)
    st.caption(
        "📖 Each row is one model run. The Sheet column shows which dataset was used — "
        "allowing you to compare the same algorithm across different sheets/datasets. "
        "F1-Score is usually the best single metric for imbalanced classification problems."
    )
    try:
        best_row = cmp_df.loc[cmp_df["F1-Score"].astype(float).idxmax()]
        label = f"{best_row.get('Sheet', '')} › {best_row.get('Method', best_row.name)}"
        st.success(f"🏆 Best by F1-Score: **{label}** (F1 = {best_row['F1-Score']})")
    except Exception:
        pass
