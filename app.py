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

h1,h2,h3 { font-family: 'Space Mono', monospace; }

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
    background: linear-gradient(135deg, #1f2d3d, #1a2235);
    border: 1px solid #58a6ff44;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
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
    "classification": {"label": "Classification / Phan loai", "color": "#58a6ff", "icon": "🔵"},
    "prediction":     {"label": "Prediction / Du bao", "color": "#bc8cff", "icon": "🟣"},
    "association":    {"label": "Association, Clustering & Balancing", "color": "#f778ba", "icon": "🔴"},
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


def fig_to_st(fig):
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# ML runners
# ─────────────────────────────────────────────────────────────────────────────

def run_classification(method, df, target, features, test_size, balance):
    st.markdown('<div class="section-header">⚙️ Training & Evaluation</div>', unsafe_allow_html=True)
    df_enc = encode_df(df[features + [target]].dropna())
    X = df_enc[features].values
    y = df_enc[target].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if balance == "Random Oversampling":
        X, y = RandomOverSampler(random_state=42).fit_resample(X, y)
        st.info("✅ Applied Random Oversampling.")
    elif balance == "SMOTE":
        try:
            X, y = SMOTE(random_state=42).fit_resample(X, y)
            st.info("✅ Applied SMOTE.")
        except Exception as e:
            st.warning(f"SMOTE skipped: {e}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)

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

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{acc:.2%}")
        st.text(classification_report(y_te, y_pred))
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        cm = confusion_matrix(y_te, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    linewidths=0.5, linecolor='#30363d')
        ax.set_title("Confusion Matrix", color='#e6edf3')
        ax.tick_params(colors='#8b949e')
        fig_to_st(fig)

    # Feature importance (where available)
    if hasattr(mdl, "feature_importances_"):
        fi = pd.Series(mdl.feature_importances_, index=features).sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        fig2.patch.set_facecolor('#0d1117')
        ax2.set_facecolor('#161b22')
        fi.head(15).plot(kind='bar', ax=ax2, color='#58a6ff')
        ax2.set_title("Feature Importances", color='#e6edf3')
        ax2.tick_params(colors='#8b949e')
        plt.tight_layout()
        fig_to_st(fig2)
    elif hasattr(mdl, "coef_"):
        coef = pd.Series(np.abs(mdl.coef_[0]) if mdl.coef_.ndim > 1 else np.abs(mdl.coef_),
                         index=features).sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        fig2.patch.set_facecolor('#0d1117')
        ax2.set_facecolor('#161b22')
        coef.head(15).plot(kind='bar', ax=ax2, color='#bc8cff')
        ax2.set_title("Coefficient Magnitudes", color='#e6edf3')
        ax2.tick_params(colors='#8b949e')
        plt.tight_layout()
        fig_to_st(fig2)

    if method == "Classification Trees":
        st.code(export_text(mdl, feature_names=features, max_depth=4), language="")


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

    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
        st.metric("RMSE", f"{np.sqrt(mse):.4f}")
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        ax.scatter(y_te, y_pred, alpha=0.6, color='#58a6ff', edgecolors='none')
        mn, mx = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5)
        ax.set_xlabel("Actual", color='#8b949e')
        ax.set_ylabel("Predicted", color='#8b949e')
        ax.set_title("Actual vs Predicted", color='#e6edf3')
        ax.tick_params(colors='#8b949e')
        fig_to_st(fig)

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
    df_enc = encode_df(df[features].dropna())
    X = StandardScaler().fit_transform(df_enc.values)

    if method == "K-Means Clustering":
        mdl = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = mdl.fit_predict(X)
        inertia_vals = []
        for k in range(2, min(11, len(X))):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertia_vals.append(km.inertia_)
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        ax.plot(range(2, min(11, len(X))), inertia_vals, 'o-', color='#58a6ff')
        ax.set_title("Elbow Curve", color='#e6edf3')
        ax.set_xlabel("K", color='#8b949e')
        ax.set_ylabel("Inertia", color='#8b949e')
        ax.tick_params(colors='#8b949e')
        fig_to_st(fig)
    else:
        mdl = AgglomerativeClustering(n_clusters=n_clusters)
        labels = mdl.fit_predict(X)
        linked = linkage(X[:min(200, len(X))], method='ward')
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        dendrogram(linked, ax=ax, color_threshold=0,
                   above_threshold_color='#58a6ff',
                   leaf_font_size=6)
        ax.set_title("Dendrogram", color='#e6edf3')
        ax.tick_params(colors='#8b949e')
        plt.tight_layout()
        fig_to_st(fig)

    df_out = df[features].copy()
    df_out["Cluster"] = labels
    st.dataframe(df_out.head(30), use_container_width=True)

    try:
        sil = silhouette_score(X, labels)
        st.metric("Silhouette Score", f"{sil:.4f}")
    except Exception:
        pass

    # 2-D scatter (first 2 features)
    if len(features) >= 2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor('#0d1117')
        ax2.set_facecolor('#161b22')
        palette = plt.cm.tab10.colors
        for c in np.unique(labels):
            mask = labels == c
            ax2.scatter(X[mask, 0], X[mask, 1],
                        color=palette[c % 10], label=f"Cluster {c}", alpha=0.7, s=30)
        ax2.legend(fontsize=7, labelcolor='#8b949e', facecolor='#161b22')
        ax2.set_title("Cluster Scatter (first 2 dims)", color='#e6edf3')
        ax2.tick_params(colors='#8b949e')
        fig_to_st(fig2)


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
    "step": 1,
    "gemini_key": "",  # user must provide their own fresh key
    "openrouter_key": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">📁 Data Upload</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Tai len CSV, Excel, JSON hoac TXT / Upload CSV, Excel, JSON, or TXT",
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
        chosen = st.selectbox("Bo du lieu hien tai / Active dataset", list(all_sheets.keys()))
        st.session_state["active_sheet"] = chosen

        df_active = all_sheets[chosen]
        st.markdown(f'<div class="card"><b style="color:#58a6ff">{chosen}</b><br>'
                    f'<span style="color:#8b949e">{df_active.shape[0]} rows × {df_active.shape[1]} cols</span></div>',
                    unsafe_allow_html=True)

        if len(all_sheets) > 1:
            st.markdown('<p class="section-header">🔗 Merge Datasets</p>', unsafe_allow_html=True)
            merge_on = st.text_input("Cot khoa chung (de gop) / Common key column", "")
            if st.button("Tu dong gop tat ca / Auto-merge all") and merge_on:
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
        'Nhap it nhat mot khoa de bat phan tich AI. '
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
        "Khoa OpenRouter API (mien phi) / OpenRouter API Key",
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
    '<div class="hero-sub">Tai du lieu len - Mo ta muc tieu - De AI huong dan phan tich</div>'
    '<div style="color:#6e7681;font-size:0.85rem;margin-bottom:1rem">'
    'Upload your data - Describe your goal - Let AI guide your analysis</div>',
    unsafe_allow_html=True,
)

if not st.session_state["sheets"]:
    st.markdown("""
    <div class="card card-accent">
    <b style="color:#58a6ff">👋 Welcome!</b><br><br>
    <ol style="color:#8b949e;line-height:2">
      <li>Tai len mot hoac nhieu tep du lieu o thanh ben (CSV, Excel, JSON, TXT).<br><small>Upload one or more data files in the sidebar.</small></li>
      <li>Mo ta muc tieu cua ban — AI se de xuat phuong phap phu hop.<br><small>Describe your goal — AI will suggest a method.</small></li>
      <li>Cau hinh tham so va chay ky thuat da chon.<br><small>Configure parameters and run the chosen technique.</small></li>
      <li>Xem ket qua, bieu do va giai thich tu AI.<br><small>View results, charts, and AI interpretation.</small></li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df_active = st.session_state["sheets"][st.session_state["active_sheet"]]

# ── Step 1 – Data Preview ─────────────────────────────────────────────────────
with st.expander("🔍 Xem truoc Du lieu / Data Preview & Profile", expanded=False):
    tab1, tab2, tab3 = st.tabs(["Table", "Statistics", "Column Types"])
    with tab1:
        st.dataframe(df_active.head(50), use_container_width=True)
    with tab2:
        st.dataframe(df_active.describe(include="all"), use_container_width=True)
    with tab3:
        dtypes = df_active.dtypes.reset_index()
        dtypes.columns = ["Column", "Type"]
        dtypes["Nulls"] = df_active.isnull().sum().values
        dtypes["Unique"] = df_active.nunique().values
        st.dataframe(dtypes, use_container_width=True)

st.divider()

# ── Step 2 – AI Goal Understanding ───────────────────────────────────────────
st.markdown('<div class="section-header">🤖 Buoc 1 / Step 1 - Mo ta Muc tieu / Describe Your Goal</div>', unsafe_allow_html=True)

user_goal = st.text_area(
    "Ban muon dat duoc dieu gi? / What do you want to achieve? (in any language)",
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

if st.button("🔎 Phan tich Muc tieu voi AI / Analyse Goal with AI"):
    with st.spinner("AI dang doc du lieu va muc tieu cua ban..."):
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

    with st.expander("🤖 AI Analysis (click to expand / Nhan de mo rong)", expanded=True):
        col_lang1, col_lang2 = st.columns([1, 1])
        with col_lang1:
            st.markdown("**English Analysis**")
            st.text(clean[:3000] + ("..." if len(clean) > 3000 else ""))
        with col_lang2:
            st.markdown("**Phan tich (Vietnamese / Tieng Viet)**")
            if st.button("Dich sang Tieng Viet", key="translate_btn"):
                with st.spinner("Dang dich..."):
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
                st.info("Nhan nut phia tren de dich sang Tieng Viet")

    # ── AI Quick Recommendation Summary ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Huong dan nhanh / Quick Recommendation")
    st.markdown(
        "Dua tren phan tich AI, hay chon mot trong cac phuong phap phu hop nhat duoi day "
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
                if st.button(f"Chon / Select", key=f"rec_{m_name}"):
                    st.session_state["chosen_method"] = m_name
                    st.rerun()
    else:
        st.info("Khong phat hien phuong phap cu the. Hay chon tu danh sach phia duoi.")

st.divider()

# ── Step 3 – Method Selection ─────────────────────────────────────────────────
st.markdown('<div class="section-header">🛠️ Buoc 2 / Step 2 - Chon Phuong phap / Choose a Method</div>', unsafe_allow_html=True)

for group_id, gmeta in GROUP_META.items():
    st.markdown(f"**{gmeta['icon']} {gmeta['label']}**")
    cols = st.columns(3)
    methods_in_group = [(n, m) for n, m in METHODS.items() if m["group"] == group_id]
    for i, (name, meta) in enumerate(methods_in_group):
        with cols[i % 3]:
            selected = st.session_state["chosen_method"] == name
            border = "2px solid #58a6ff" if selected else "1px solid #30363d"
            st.markdown(
                f'<div style="background:#161b22;border:{border};border-radius:10px;'
                f'padding:0.8rem;margin-bottom:0.6rem;">'
                f'<span class="badge {meta["badge"]}">{group_id.upper()}</span><br>'
                f'<b style="color:#e6edf3">{name}</b><br>'
                f'<small style="color:#8b949e">{meta["vn"]}</small><br>'
                f'<small style="color:#6e7681;font-size:0.75rem">{meta["desc"][:90]}…</small>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Select", key=f"sel_{name}"):
                st.session_state["chosen_method"] = name
                st.rerun()
    st.markdown("")

st.divider()

# ── Step 4 – Configure & Run ──────────────────────────────────────────────────
method = st.session_state["chosen_method"]
if not method:
    st.info("👆 Hay chon mot phuong phap o tren de cau hinh va chay / Select a method above.")
    st.stop()

st.markdown(f'<div class="section-header">⚡ Step 3 — Configure & Run: {method}</div>',
            unsafe_allow_html=True)

meta = METHODS[method]
st.markdown(f'<div class="card"><span class="badge {meta["badge"]}">{meta["group"].upper()}</span> '
            f'<b>{method}</b> — {meta["desc"]}</div>', unsafe_allow_html=True)

numeric_cols = df_active.select_dtypes(include=[np.number]).columns.tolist()
all_cols = df_active.columns.tolist()

group = meta["group"]

# ── Classification & Prediction shared config ────────────────────────────────
if group in ("classification", "prediction") or method in ("Random Oversampling", "SMOTE"):
    col_a, col_b = st.columns(2)
    with col_a:
        target_col = st.selectbox("🎯 Cot muc tieu / Target column", all_cols)
    with col_b:
        feature_cols = st.multiselect(
            "📐 Cac cot dac trung / Feature columns",
            [c for c in all_cols if c != target_col],
            default=[c for c in numeric_cols if c != target_col][:8],
        )

if group == "classification":
    test_size = st.slider("Ty le kiem tra % / Test split %", 10, 40, 20) / 100
    balance_opt = st.selectbox("Can bang lop (tuy chon) / Class balancing (optional)",
                               ["None", "Random Oversampling", "SMOTE"])
elif group == "prediction" and method != "Neural Networks Regression (MLP)":
    test_size = st.slider("Test split %", 10, 40, 20) / 100
elif group == "prediction":
    test_size = st.slider("Test split %", 10, 40, 20) / 100

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
    if group == "classification" and method not in ("Random Oversampling", "SMOTE"):
        if not feature_cols:
            st.error("Select at least one feature column.")
        else:
            run_classification(method, df_active, target_col, feature_cols, test_size,
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

    # ── AI interpretation ─────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">🤖 AI Result Interpretation</div>',
                unsafe_allow_html=True)
    with st.spinner("Gemini is interpreting the results…"):
        interp_prompt = f"""
You are a data mining expert.
The user just ran **{method}** on this dataset:
{df_summary(df_active)}

User's original goal: {user_goal or '(not specified)'}

Please:
1. Explain what the results likely mean in plain language.
2. Highlight what went well and any limitations.
3. Suggest the next steps the user should take.
4. Suggest 1-2 alternative methods they could try.

Keep it concise and practical. Respond in the same language the user used (default English).
"""
        interp = ask_gemini(interp_prompt)
    st.markdown('<div class="ai-bubble">🤖 <b style="color:#58a6ff">Gemini Interpretation</b><br><br>' +
                interp.replace("\n", "<br>") + "</div>", unsafe_allow_html=True)
