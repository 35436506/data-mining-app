import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Mining Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f6b8a 50%, #00a8b5 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0,168,181,0.25);
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; margin:0; letter-spacing:-0.5px; }
    .main-header p  { font-size: 1rem; opacity: 0.85; margin:0.4rem 0 0; }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #00a8b5;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 0.8rem;
    }
    .metric-card .label { font-size: 0.78rem; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #1e3a5f; }
    .metric-card .sub   { font-size: 0.8rem; color: #9ca3af; }
    
    .insight-box {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        border-left: 4px solid #0284c7;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
    }
    .warning-box {
        background: #fff7ed;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
    }
    .success-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
    }
    .step-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e3a5f;
        padding: 0.5rem 0;
        border-bottom: 2px solid #e5e7eb;
        margin-bottom: 0.8rem;
    }
    .tech-badge {
        display: inline-block;
        background: #dbeafe;
        color: #1d4ed8;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #1e3a5f, #0f6b8a);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.8rem 0;
    }
    .recommendation-card h4 { color: #7dd3fc; margin: 0 0 0.5rem; font-size: 0.9rem; text-transform: uppercase; }
    .recommendation-card p  { margin: 0; font-size: 1rem; line-height: 1.5; }

    div[data-testid="stTabs"] [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stSelectbox label, .stMultiSelect label { font-weight: 600; color: #374151; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔬 Data Mining Platform</h1>
    <p>Nền tảng phân tích dữ liệu thông minh dành cho lãnh đạo • Smart Analytics for Decision Makers</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình hệ thống")
    st.markdown("---")
    
    uploaded_files = st.file_uploader(
        "📁 Tải lên dữ liệu (CSV / XLSX)",
        type=["csv","xlsx"],
        accept_multiple_files=True,
        help="Tối đa 50MB mỗi file. Hỗ trợ .csv và .xlsx"
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Mục tiêu phân tích")
    task_type = st.radio(
        "Chọn loại bài toán:",
        ["🔮 Phân loại (Classification)", "📈 Dự báo (Prediction/Regression)", "🔗 Luật kết hợp (Association)"],
        help="Phân loại: dự đoán nhóm | Dự báo: dự đoán giá trị số | Luật kết hợp: tìm mẫu cùng xuất hiện"
    )
    
    st.markdown("---")
    with st.expander("📚 Giải thích các kỹ thuật"):
        technique_info = {
            "Logistic Regression": "Phân loại nhị phân dựa trên xác suất. Tốt nhất khi mối quan hệ tuyến tính.",
            "Linear Regression": "Dự báo giá trị liên tục (số). Phù hợp khi target là số thực.",
            "Neural Network (MLP)": "Mạng nơ-ron nhân tạo, học được pattern phức tạp phi tuyến. Cần nhiều dữ liệu.",
            "LDA": "Linear Discriminant Analysis: tìm chiều không gian phân biệt nhóm tốt nhất.",
            "KNN": "K-Nearest Neighbors: phân loại dựa trên K điểm dữ liệu gần nhất.",
            "Decision Tree": "Cây quyết định: tạo luật if-then dễ giải thích cho lãnh đạo.",
            "Random Forest": "Tập hợp nhiều cây quyết định, độ chính xác cao hơn.",
            "Naive Bayes": "Xác suất có điều kiện Bayes, nhanh, tốt cho text/category.",
            "SVM": "Support Vector Machine: tìm siêu phẳng phân cách tốt nhất.",
            "K-Means": "Phân cụm không giám sát: chia dữ liệu thành K nhóm tự nhiên.",
            "Hierarchical": "Phân cụm phân cấp: tạo cây phân cấp nhóm, không cần chọn K trước.",
            "Apriori (Rules)": "Tìm các quy luật 'Nếu A thì B' từ transaction data.",
            "SMOTE": "Synthetic Minority Over-sampling: tạo dữ liệu tổng hợp cho nhóm thiểu số."
        }
        for tech, desc in technique_info.items():
            st.markdown(f"**{tech}**")
            st.caption(desc)
            st.markdown("")
    
    st.markdown("---")
    st.caption("⚠️ Giới hạn: Mỗi file < 50MB | Định dạng: .csv, .xlsx")

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_merge_files(files_data):
    """Load multiple files and intelligently merge them."""
    dfs = {}
    for name, content in files_data:
        try:
            if name.endswith('.csv'):
                df = pd.read_csv(content)
            else:
                xl = pd.read_excel(content, sheet_name=None)
                # Pick the largest sheet
                df = max(xl.values(), key=len)
            dfs[name] = df
        except Exception as e:
            st.warning(f"Lỗi đọc file {name}: {e}")
    
    if len(dfs) == 0:
        return None, {}
    
    if len(dfs) == 1:
        name, df = list(dfs.items())[0]
        return df, {name: df.shape}
    
    # Find common join columns
    all_col_sets = [set(df.columns.str.lower()) for df in dfs.values()]
    common_cols = all_col_sets[0].intersection(*all_col_sets[1:])
    id_keywords = ['id','key','num','code','jobnum','reqid','equipid','no','number']
    join_keys = [c for c in common_cols if any(k in c for k in id_keywords)]
    
    df_list = list(dfs.values())
    names   = list(dfs.keys())
    shapes  = {n: df.shape for n, df in dfs.items()}
    
    if join_keys:
        # Merge on first detected key (map back original casing)
        key_lower = join_keys[0]
        key_maps  = []
        for df in df_list:
            match = next((c for c in df.columns if c.lower() == key_lower), None)
            key_maps.append(match)
        if all(key_maps):
            merged = df_list[0].rename(columns={key_maps[0]: key_lower})
            for i in range(1, len(df_list)):
                right = df_list[i].rename(columns={key_maps[i]: key_lower})
                suffix = f"_{names[i].split('.')[0]}"
                merged = merged.merge(right, on=key_lower, how='outer', suffixes=('', suffix))
            shapes['[Merged]'] = merged.shape
            return merged, shapes
    
    # Fallback: try column-based concat (same columns)
    try:
        merged = pd.concat(df_list, ignore_index=True)
        shapes['[Concatenated]'] = merged.shape
        return merged, shapes
    except:
        return df_list[0], shapes


def data_health_report(df):
    total_cells = df.shape[0] * df.shape[1]
    missing_total = df.isnull().sum().sum()
    missing_pct = missing_total / total_cells * 100
    
    col_report = pd.DataFrame({
        'Cột': df.columns,
        'Kiểu dữ liệu': df.dtypes.values.astype(str),
        'Ô trống': df.isnull().sum().values,
        'Tỷ lệ trống (%)': (df.isnull().sum().values / len(df) * 100).round(1),
        'Giá trị duy nhất': df.nunique().values
    })
    return missing_pct, col_report


def clean_and_explain(df, target_col=None):
    """Clean data with explanations."""
    steps = []
    df = df.copy()
    
    # Drop near-empty columns (>80% missing)
    high_missing = [c for c in df.columns if df[c].isnull().mean() > 0.8]
    if high_missing:
        df.drop(columns=high_missing, inplace=True)
        steps.append(f"🗑️ Loại bỏ {len(high_missing)} cột có >80% dữ liệu trống: `{', '.join(high_missing[:3])}{'...' if len(high_missing)>3 else ''}`")
    
    # Fill numeric missing
    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        miss = df[col].isnull().sum()
        if miss > 0:
            pct = miss / len(df) * 100
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            steps.append(f"🔢 Cột **{col}**: Phát hiện {miss} ô trống ({pct:.1f}%) → Lấp đầy bằng giá trị trung vị ({median_val:.2f}) để giữ lại thông tin của {len(df)-miss} dòng còn lại.")
    
    # Fill categorical missing
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        miss = df[col].isnull().sum()
        if miss > 0 and col != target_col:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_val, inplace=True)
            steps.append(f"🔤 Cột **{col}**: {miss} ô trống → Lấp đầy bằng giá trị phổ biến nhất ('{mode_val}').")
    
    # Drop datetime columns (not useful for ML)
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    # Also detect object cols that look like dates
    for col in df.select_dtypes(include='object').columns:
        try:
            sample = df[col].dropna().head(20)
            parsed = pd.to_datetime(sample, errors='coerce')
            if parsed.notna().mean() > 0.7:
                date_cols.append(col)
        except:
            pass
    
    if date_cols:
        df.drop(columns=[c for c in date_cols if c in df.columns and c != target_col], inplace=True, errors='ignore')
        steps.append(f"📅 Loại bỏ {len(date_cols)} cột ngày tháng (không phù hợp cho học máy): `{', '.join(date_cols[:3])}`")
    
    return df, steps


def encode_and_explain(df, target_col=None):
    """Label encode categorical columns."""
    from sklearn.preprocessing import LabelEncoder
    steps = []
    df = df.copy()
    encoders = {}
    
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if target_col and target_col in cat_cols:
        cat_cols.remove(target_col)
    
    # Drop high-cardinality columns (too many unique values = not useful)
    to_drop = []
    for col in cat_cols:
        if df[col].nunique() > min(50, len(df)*0.5):
            to_drop.append(col)
    
    if to_drop:
        df.drop(columns=to_drop, inplace=True, errors='ignore')
        steps.append(f"🚫 Loại bỏ {len(to_drop)} cột định danh có quá nhiều giá trị khác nhau (ID, mã số...) vì không giúp ích cho mô hình: `{', '.join(to_drop[:3])}`")
        cat_cols = [c for c in cat_cols if c not in to_drop]
    
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    if cat_cols:
        steps.append(f"🔡→🔢 Mã hóa {len(cat_cols)} cột văn bản sang số (Label Encoding) để thuật toán toán học có thể đo lường mức độ ảnh hưởng: `{', '.join(cat_cols[:5])}{'...' if len(cat_cols)>5 else ''}`")
    
    # Encode target if needed
    target_encoder = None
    if target_col and target_col in df.columns and df[target_col].dtype == 'object':
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col].astype(str))
        target_encoder = le
        steps.append(f"🎯 Mã hóa cột mục tiêu **{target_col}**: {list(le.classes_)} → {list(range(len(le.classes_)))}")
    
    return df, steps, encoders, target_encoder


def check_balance_and_smote(X, y, task):
    """Check class balance and optionally apply SMOTE."""
    if task != "🔮 Phân loại (Classification)":
        return X, y, [], False
    
    steps = []
    from collections import Counter
    counts = Counter(y)
    total = len(y)
    min_class_pct = min(counts.values()) / total * 100
    
    if min_class_pct < 20 and len(X) > 50:
        steps.append(f"⚠️ **Mất cân bằng dữ liệu phát hiện!** Nhóm thiểu số chỉ chiếm {min_class_pct:.1f}% — máy dễ bị 'học vẹt' theo nhóm đa số. Hệ thống kích hoạt **SMOTE** để tạo thêm dữ liệu tổng hợp, giúp mô hình bắt được dấu hiệu của nhóm hiếm này.")
        try:
            from imblearn.over_sampling import SMOTE
            k = min(5, min(counts.values()) - 1)
            if k >= 1:
                sm = SMOTE(k_neighbors=k, random_state=42)
                X_res, y_res = sm.fit_resample(X, y)
                new_counts = Counter(y_res)
                steps.append(f"✅ SMOTE hoàn tất: {total} → {len(X_res)} mẫu. Phân phối mới: {dict(new_counts)}")
                return X_res, y_res, steps, True
        except Exception as e:
            steps.append(f"ℹ️ SMOTE không áp dụng được ({e}), tiếp tục với dữ liệu gốc.")
    else:
        steps.append(f"✅ Dữ liệu cân bằng tốt: nhóm thiểu số chiếm {min_class_pct:.1f}%. Không cần SMOTE.")
    
    return X, y, steps, False


def run_models(X_train, X_test, y_train, y_test, task, selected_models):
    """Train selected models and return results."""
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC, SVR
    from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                                  mean_absolute_error, r2_score, mean_squared_error,
                                  confusion_matrix)
    
    results = {}
    
    clf_map = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=42),
        "LDA": LinearDiscriminantAnalysis(),
        "KNN": KNeighborsClassifier(n_neighbors=min(5, len(X_train)//10+1)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True, random_state=42)
    }
    reg_map = {
        "Linear Regression": LinearRegression(),
        "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300, random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=min(5, len(X_train)//10+1)),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        "SVM": SVR()
    }
    
    model_map = clf_map if task == "🔮 Phân loại (Classification)" else reg_map
    
    for name in selected_models:
        if name not in model_map:
            continue
        model = model_map[name]
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if task == "🔮 Phân loại (Classification)":
                results[name] = {
                    'model': model,
                    'y_pred': y_pred,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'cm': confusion_matrix(y_test, y_pred)
                }
            else:
                results[name] = {
                    'model': model,
                    'y_pred': y_pred,
                    'r2': r2_score(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                }
        except Exception as e:
            st.warning(f"Mô hình {name} gặp lỗi: {e}")
    
    return results


def get_feature_importance(model, feature_names, model_name):
    """Extract feature importances from model."""
    try:
        if hasattr(model, 'feature_importances_'):
            return pd.Series(model.feature_importances_, index=feature_names)
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            return pd.Series(np.abs(coef), index=feature_names)
    except:
        pass
    return None


def run_kmeans(df_num, k=3):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_num)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km


def run_hierarchical(df_num, k=3):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_num.values[:min(500, len(df_num))])
    hc = AgglomerativeClustering(n_clusters=k)
    labels = hc.fit_predict(X_scaled)
    return labels


def run_apriori(df, min_support=0.05, min_confidence=0.3):
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    
    # Convert to transaction format
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if not cat_cols:
        # Use binary columns
        binary_cols = [c for c in df.columns if df[c].nunique() == 2]
        if not binary_cols:
            return None, None
        df_bin = df[binary_cols].astype(bool)
        freq = apriori(df_bin, min_support=min_support, use_colnames=True)
    else:
        transactions = df[cat_cols[:10]].astype(str).values.tolist()
        te = TransactionEncoder()
        te_array = te.fit_transform(transactions)
        df_trans = pd.DataFrame(te_array, columns=te.columns_)
        freq = apriori(df_trans, min_support=min_support, use_colnames=True)
    
    if freq.empty:
        return freq, None
    rules = association_rules(freq, metric='confidence', min_threshold=min_confidence)
    return freq, rules


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

if not uploaded_files:
    # Landing page
    st.markdown("""
    <div style="text-align:center; padding: 3rem; background: #f8fafc; border-radius: 16px; border: 2px dashed #cbd5e1;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
        <h2 style="color: #1e3a5f; font-weight: 700;">Bắt đầu bằng cách tải lên dữ liệu</h2>
        <p style="color: #6b7280; max-width: 500px; margin: 0 auto;">
            Tải lên một hoặc nhiều file CSV/XLSX. Hệ thống sẽ tự động phát hiện mối quan hệ và kết nối dữ liệu.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="label">📁 Đầu vào</div>
            <div class="value" style="font-size:1.2rem">Multi-file</div>
            <div class="sub">Tự động kết nối bảng dữ liệu</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="label">🧠 Mô hình</div>
            <div class="value" style="font-size:1.2rem">10+ Kỹ thuật</div>
            <div class="sub">ML cổ điển & nâng cao</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="label">📊 Kết quả</div>
            <div class="value" style="font-size:1.2rem">Executive</div>
            <div class="sub">Dashboard tương tác cho lãnh đạo</div>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ── Load files ────────────────────────────────────────────────────────────────
files_data = [(f.name, f) for f in uploaded_files]

# Check file size
for f in uploaded_files:
    size_mb = f.size / (1024*1024)
    if size_mb > 50:
        st.error(f"⛔ File **{f.name}** vượt quá 50MB ({size_mb:.1f}MB). Vui lòng giảm kích thước file.")
        st.stop()

with st.spinner("🔍 Đang đọc và kết nối dữ liệu..."):
    df_raw, shape_report = load_and_merge_files(files_data)

if df_raw is None or df_raw.empty:
    st.error("Không thể đọc dữ liệu. Vui lòng kiểm tra định dạng file.")
    st.stop()

# ── Data Health Report ────────────────────────────────────────────────────────
st.markdown("## 📋 Báo cáo Sức khỏe Dữ liệu")

col1, col2, col3, col4 = st.columns(4)
missing_pct, col_report = data_health_report(df_raw)

with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Tổng số dòng</div>
        <div class="value">{df_raw.shape[0]:,}</div>
        <div class="sub">Records</div></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Tổng số cột</div>
        <div class="value">{df_raw.shape[1]}</div>
        <div class="sub">Features</div></div>""", unsafe_allow_html=True)
with col3:
    color = "#ef4444" if missing_pct > 20 else "#22c55e"
    st.markdown(f"""<div class="metric-card">
        <div class="label">Tỷ lệ ô trống</div>
        <div class="value" style="color:{color}">{missing_pct:.1f}%</div>
        <div class="sub">Missing data</div></div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Số file đã kết nối</div>
        <div class="value">{len(shape_report)}</div>
        <div class="sub">Files / Tables</div></div>""", unsafe_allow_html=True)

# File connection status
with st.expander("📁 Chi tiết file và kết nối dữ liệu", expanded=True):
    for fname, shape in shape_report.items():
        badge = "🔗" if "[Merged]" in fname or "[Concatenated]" in fname else "📄"
        st.markdown(f"{badge} **{fname}** — {shape[0]:,} dòng × {shape[1]} cột")
    if len(shape_report) > 1:
        st.markdown("""<div class="success-box">✅ Hệ thống đã tự động phát hiện và kết nối các bảng dữ liệu dựa trên cột định danh chung.</div>""", unsafe_allow_html=True)
    
    st.dataframe(col_report.style.background_gradient(subset=['Tỷ lệ trống (%)'], cmap='RdYlGn_r'), use_container_width=True)

st.markdown("---")

# ── Analysis Configuration ────────────────────────────────────────────────────
st.markdown("## ⚙️ Cấu hình Phân tích")

col_left, col_right = st.columns([1,1])

with col_left:
    all_cols = df_raw.columns.tolist()
    target_col = st.selectbox(
        "🎯 Chọn cột mục tiêu (Target)",
        ["(Không chọn — chỉ phân cụm)"] + all_cols,
        help="Cột bạn muốn dự báo hoặc phân loại"
    )
    if target_col == "(Không chọn — chỉ phân cụm)":
        target_col = None

with col_right:
    exclude_cols = st.multiselect(
        "🚫 Loại trừ cột (tuỳ chọn)",
        [c for c in all_cols if c != target_col],
        help="Các cột không muốn đưa vào mô hình"
    )

# Model selection
st.markdown("### 🤖 Chọn mô hình phân tích")

if task_type == "🔮 Phân loại (Classification)":
    available_models = ["Logistic Regression","Neural Network (MLP)","LDA","KNN","Decision Tree","Random Forest","Naive Bayes","SVM"]
elif task_type == "📈 Dự báo (Prediction/Regression)":
    available_models = ["Linear Regression","Neural Network (MLP)","KNN","Decision Tree","Random Forest","SVM"]
else:
    available_models = ["K-Means Clustering","Hierarchical Clustering","Apriori (Association Rules)"]

selected_models = st.multiselect(
    "Chọn các mô hình muốn chạy:",
    available_models,
    default=available_models[:3] if len(available_models) >= 3 else available_models
)

if task_type == "🔮 Phân loại (Classification)" or task_type == "📈 Dự báo (Prediction/Regression)":
    if target_col is None:
        st.warning("⚠️ Vui lòng chọn cột mục tiêu để chạy mô hình phân loại/dự báo.")

run_btn = st.button("🚀 Chạy phân tích", type="primary", use_container_width=True)

if not run_btn:
    st.info("👆 Cấu hình xong, nhấn **Chạy phân tích** để bắt đầu.")
    
    # Preview data
    with st.expander("👁️ Xem trước dữ liệu"):
        st.dataframe(df_raw.head(20), use_container_width=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 🔄 Quy trình Xử lý có Giải thích (Explainable AI Pipeline)")

df_work = df_raw.copy()
if exclude_cols:
    df_work.drop(columns=[c for c in exclude_cols if c in df_work.columns], inplace=True)

# STEP 1: Clean
with st.status("🧹 Bước 1: Làm sạch dữ liệu (Data Cleaning)...", expanded=True) as status:
    df_work, clean_steps = clean_and_explain(df_work, target_col)
    for step in clean_steps:
        st.markdown(step)
    if not clean_steps:
        st.markdown("✅ Dữ liệu đã sạch, không cần xử lý thêm.")
    status.update(label=f"✅ Bước 1 hoàn tất — {len(clean_steps)} thao tác làm sạch", state="complete")

# STEP 2: Encode
with st.status("🔡 Bước 2: Mã hóa dữ liệu (Feature Encoding)...", expanded=True) as status:
    df_encoded, encode_steps, encoders, target_encoder = encode_and_explain(df_work, target_col)
    for step in encode_steps:
        st.markdown(step)
    if not encode_steps:
        st.markdown("✅ Tất cả cột đã ở dạng số, không cần mã hóa.")
    status.update(label=f"✅ Bước 2 hoàn tất — {len(encode_steps)} thao tác mã hóa", state="complete")

# Remove remaining non-numeric
df_ml = df_encoded.select_dtypes(include='number').copy()
if target_col and target_col not in df_ml.columns:
    st.error(f"Cột mục tiêu '{target_col}' không hợp lệ sau khi xử lý.")
    st.stop()

# For clustering / association: skip step 3-4 supervised flow
if task_type == "🔗 Luật kết hợp (Association)":
    with st.status("🔗 Chạy Apriori Association Rules...", expanded=True) as status:
        try:
            freq, rules = run_apriori(df_work, min_support=0.03, min_confidence=0.3)
            status.update(label="✅ Apriori hoàn tất", state="complete")
        except Exception as e:
            st.error(f"Lỗi Apriori: {e}")
            rules = None
            status.update(label="❌ Lỗi", state="error")
    
    if rules is not None and not rules.empty:
        st.markdown("### 🔗 Luật kết hợp phát hiện được")
        rules_display = rules.sort_values('lift', ascending=False).head(20)[['antecedents','consequents','support','confidence','lift']]
        rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
        rules_display.columns = ['Điều kiện (Nếu...)','Kết quả (Thì...)','Support','Confidence','Lift']
        st.dataframe(rules_display.style.background_gradient(subset=['Lift'], cmap='Greens'), use_container_width=True)
        
        st.markdown("""<div class="recommendation-card">
            <h4>💡 Khuyến nghị hành động</h4>
            <p>Các luật có Lift > 1.5 cho thấy mối quan hệ có ý nghĩa thực tế. Tập trung vào nhóm điều kiện có Confidence cao để tối ưu hóa quy trình vận hành và bảo trì thiết bị.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("Không tìm thấy luật kết hợp đủ ý nghĩa với ngưỡng hiện tại. Thử giảm min_support.")
    st.stop()

# Clustering task
if task_type == "🔮 Phân loại (Classification)" and target_col is None and "K-Means Clustering" in selected_models:
    task_type_effective = "clustering"
else:
    task_type_effective = task_type

# STEP 3: Balancing
feature_cols = [c for c in df_ml.columns if c != target_col]
X_full = df_ml[feature_cols].values
y_full = df_ml[target_col].values if target_col else None

if y_full is not None:
    with st.status("⚖️ Bước 3: Kiểm tra cân bằng dữ liệu (Class Balancing)...", expanded=True) as status:
        X_full, y_full, balance_steps, smote_applied = check_balance_and_smote(X_full, y_full, task_type)
        for step in balance_steps:
            st.markdown(step)
        status.update(label="✅ Bước 3 hoàn tất", state="complete")
    
    # STEP 4: Train/Test Split
    with st.status("✂️ Bước 4: Phân chia Train/Test & Huấn luyện mô hình...", expanded=True) as status:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)
        
        st.markdown(f"📊 Tập huấn luyện: **{len(X_train):,}** mẫu | Tập kiểm tra: **{len(X_test):,}** mẫu (80/20 split)")
        
        results = run_models(X_train_sc, X_test_sc, y_train, y_test, task_type, selected_models)
        st.markdown(f"✅ Đã huấn luyện **{len(results)}** mô hình thành công.")
        status.update(label=f"✅ Bước 4 hoàn tất — {len(results)} mô hình", state="complete")
else:
    results = {}
    X_train_sc = X_test_sc = y_train = y_test = None

# Clustering models
cluster_results = {}
if "K-Means Clustering" in selected_models and len(df_ml) > 10:
    with st.status("🎯 Chạy K-Means Clustering...", expanded=False) as status:
        df_num = df_ml[feature_cols].select_dtypes(include='number').dropna()
        if df_num.shape[1] >= 2:
            labels_km, km_model = run_kmeans(df_num.head(2000), k=3)
            cluster_results['K-Means'] = {'labels': labels_km, 'data': df_num.head(2000)}
            status.update(label="✅ K-Means hoàn tất", state="complete")

if "Hierarchical Clustering" in selected_models and len(df_ml) > 10:
    with st.status("🌲 Chạy Hierarchical Clustering...", expanded=False) as status:
        df_num = df_ml[feature_cols].select_dtypes(include='number').dropna()
        if df_num.shape[1] >= 2:
            labels_hc = run_hierarchical(df_num, k=3)
            cluster_results['Hierarchical'] = {'labels': labels_hc, 'data': df_num.head(500)}
            status.update(label="✅ Hierarchical hoàn tất", state="complete")

# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD TABS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 📊 Executive Dashboard")

tab1, tab2, tab3 = st.tabs(["📈 Tổng quan dữ liệu", "🏆 Hiệu suất mô hình", "💡 Insight & Dự báo"])

# ── TAB 1: Overview ───────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🗺️ Bản đồ tương quan (Correlation Heatmap)")
    
    num_df = df_ml.select_dtypes(include='number')
    if num_df.shape[1] > 1:
        corr = num_df.corr()
        fig_hm = px.imshow(
            corr,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title="Ma trận tương quan giữa các biến",
            aspect='auto'
        )
        fig_hm.update_layout(height=500, font_family='Inter')
        st.plotly_chart(fig_hm, use_container_width=True)
    
    st.markdown("### 📊 Phân phối dữ liệu")
    num_cols_show = num_df.columns[:12].tolist()
    
    if num_cols_show:
        n_cols = min(3, len(num_cols_show))
        n_rows = (len(num_cols_show) + n_cols - 1) // n_cols
        fig_dist = make_subplots(rows=n_rows, cols=n_cols,
                                  subplot_titles=num_cols_show[:n_rows*n_cols])
        for i, col in enumerate(num_cols_show[:n_rows*n_cols]):
            r, c = divmod(i, n_cols)
            fig_dist.add_trace(
                go.Histogram(x=num_df[col].dropna(), name=col, showlegend=False,
                             marker_color='#00a8b5', opacity=0.8),
                row=r+1, col=c+1
            )
        fig_dist.update_layout(height=250*n_rows, title="Phân phối các biến số", font_family='Inter')
        st.plotly_chart(fig_dist, use_container_width=True)
    
    if target_col and target_col in df_ml.columns:
        st.markdown(f"### 🎯 Phân phối biến mục tiêu: `{target_col}`")
        val_counts = df_ml[target_col].value_counts().reset_index()
        val_counts.columns = ['Giá trị','Số lượng']
        fig_target = px.bar(val_counts, x='Giá trị', y='Số lượng',
                            color='Số lượng', color_continuous_scale='Blues',
                            title=f"Phân phối của {target_col}")
        fig_target.update_layout(font_family='Inter')
        st.plotly_chart(fig_target, use_container_width=True)
    
    # Clustering scatter
    for cname, cdata in cluster_results.items():
        st.markdown(f"### 🔵 Kết quả {cname}")
        data_plot = cdata['data'].copy()
        data_plot['Cụm'] = cdata['labels'].astype(str)
        cols_plot = data_plot.select_dtypes(include='number').columns.tolist()
        if len(cols_plot) >= 2:
            fig_cl = px.scatter(data_plot, x=cols_plot[0], y=cols_plot[1],
                                color='Cụm', title=f"{cname}: Phân cụm dữ liệu",
                                color_discrete_sequence=px.colors.qualitative.Set2)
            fig_cl.update_layout(font_family='Inter')
            st.plotly_chart(fig_cl, use_container_width=True)

# ── TAB 2: Performance ────────────────────────────────────────────────────────
with tab2:
    if not results:
        st.info("Không có kết quả mô hình phân loại/dự báo. Chọn cột mục tiêu và chạy lại.")
    else:
        st.markdown("### 🏆 So sánh hiệu suất các mô hình")
        
        if task_type == "🔮 Phân loại (Classification)":
            perf_data = []
            for name, r in results.items():
                perf_data.append({
                    'Mô hình': name,
                    'Accuracy (%)': round(r['accuracy']*100, 2),
                    'F1-Score': round(r['f1'], 4),
                    'Precision': round(r['precision'], 4),
                    'Recall (Sensitivity)': round(r['recall'], 4)
                })
            perf_df = pd.DataFrame(perf_data).sort_values('Accuracy (%)', ascending=False)
            
            fig_perf = px.bar(perf_df, x='Mô hình', y='Accuracy (%)',
                              color='Accuracy (%)', color_continuous_scale='Teal',
                              title='Độ chính xác (Accuracy) các mô hình',
                              text='Accuracy (%)')
            fig_perf.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_perf.update_layout(font_family='Inter', height=400, showlegend=False)
            st.plotly_chart(fig_perf, use_container_width=True)
            
            st.dataframe(perf_df.style.background_gradient(subset=['Accuracy (%)','F1-Score'], cmap='Greens'),
                         use_container_width=True)
            
            # Confusion Matrix for best model
            best_model_name = perf_df.iloc[0]['Mô hình']
            best_result = results[best_model_name]
            
            st.markdown(f"### 🔢 Confusion Matrix — Mô hình tốt nhất: **{best_model_name}**")
            cm = best_result['cm']
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               title=f"Confusion Matrix: {best_model_name}",
                               labels=dict(x="Dự báo", y="Thực tế"))
            fig_cm.update_layout(font_family='Inter', height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Best Accuracy</div>
                    <div class="value">{best_result['accuracy']*100:.1f}%</div>
                    <div class="sub">{best_model_name}</div></div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">F1-Score</div>
                    <div class="value">{best_result['f1']:.3f}</div>
                    <div class="sub">Weighted average</div></div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Recall</div>
                    <div class="value">{best_result['recall']:.3f}</div>
                    <div class="sub">Sensitivity</div></div>""", unsafe_allow_html=True)
        
        else:  # Regression
            perf_data = []
            for name, r in results.items():
                perf_data.append({
                    'Mô hình': name,
                    'R² Score': round(r['r2'], 4),
                    'MAE': round(r['mae'], 4),
                    'RMSE': round(r['rmse'], 4)
                })
            perf_df = pd.DataFrame(perf_data).sort_values('R² Score', ascending=False)
            
            fig_r2 = px.bar(perf_df, x='Mô hình', y='R² Score',
                            color='R² Score', color_continuous_scale='Teal',
                            title='R² Score (càng gần 1 càng tốt)', text='R² Score')
            fig_r2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_r2.update_layout(font_family='Inter', height=400, showlegend=False)
            st.plotly_chart(fig_r2, use_container_width=True)
            
            st.dataframe(perf_df.style.background_gradient(subset=['R² Score'], cmap='Greens'),
                         use_container_width=True)
            
            # Actual vs Predicted for best model
            best_name = perf_df.iloc[0]['Mô hình']
            best_r = results[best_name]
            fig_avp = px.scatter(x=y_test[:200], y=best_r['y_pred'][:200],
                                 labels={'x':'Giá trị thực tế','y':'Giá trị dự báo'},
                                 title=f"Thực tế vs Dự báo — {best_name}",
                                 trendline='ols')
            fig_avp.update_layout(font_family='Inter')
            st.plotly_chart(fig_avp, use_container_width=True)

# ── TAB 3: Insight ────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🔍 Top 3 Yếu tố Ảnh hưởng Mạnh nhất (Feature Importance)")
    
    importance_shown = False
    if results:
        # Find best interpretable model
        for preferred in ["Random Forest","Decision Tree","Logistic Regression","Linear Regression"]:
            if preferred in results:
                model_for_fi = results[preferred]['model']
                fi = get_feature_importance(model_for_fi, feature_cols, preferred)
                if fi is not None and len(fi) > 0:
                    fi_sorted = fi.sort_values(ascending=False)
                    top3 = fi_sorted.head(3)
                    
                    fig_fi = px.bar(
                        fi_sorted.head(15).reset_index(),
                        x=fi_sorted.head(15).values,
                        y=fi_sorted.head(15).index,
                        orientation='h',
                        color=fi_sorted.head(15).values,
                        color_continuous_scale='Teal',
                        title=f"Feature Importance — {preferred}",
                        labels={'x':'Mức độ ảnh hưởng','y':'Yếu tố'}
                    )
                    fig_fi.update_layout(font_family='Inter', height=400, showlegend=False, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_fi, use_container_width=True)
                    
                    st.markdown("#### 🏆 Top 3 yếu tố then chốt:")
                    for i, (feat, val) in enumerate(top3.items(), 1):
                        medals = ["🥇","🥈","🥉"]
                        pct = val / fi_sorted.sum() * 100
                        st.markdown(f"""<div class="insight-box">
                            {medals[i-1]} <strong>#{i}: {feat}</strong> — Mức độ ảnh hưởng: <strong>{pct:.1f}%</strong>
                        </div>""", unsafe_allow_html=True)
                    
                    importance_shown = True
                    break
    
    if not importance_shown:
        st.info("Chạy mô hình phân loại/dự báo để xem Feature Importance.")
    
    st.markdown("---")
    st.markdown("### 💡 Khuyến nghị Chiến lược cho Lãnh đạo")
    
    if results and importance_shown:
        best_model_name = list(results.keys())[0]
        if task_type == "🔮 Phân loại (Classification)":
            best_acc = results[best_model_name].get('accuracy', 0) * 100
        else:
            best_acc = results[best_model_name].get('r2', 0) * 100
        
        top_feat = top3.index[0] if 'top3' in dir() else "yếu tố hàng đầu"
        
        st.markdown(f"""
        <div class="recommendation-card">
            <h4>🎯 Hành động ưu tiên #1</h4>
            <p>Dựa trên phân tích dữ liệu, <strong>{top_feat}</strong> là yếu tố có ảnh hưởng lớn nhất đến kết quả. 
            Lãnh đạo nên tập trung nguồn lực vào việc kiểm soát và tối ưu yếu tố này để cải thiện hiệu quả tổng thể.</p>
        </div>
        <div class="recommendation-card" style="background: linear-gradient(135deg, #064e3b, #065f46); margin-top:0.8rem;">
            <h4>📊 Độ tin cậy mô hình</h4>
            <p>Mô hình <strong>{best_model_name}</strong> đạt hiệu suất tốt nhất với độ chính xác/R² = <strong>{best_acc:.1f}%</strong>. 
            Khuyến nghị sử dụng mô hình này cho các quyết định định kỳ và theo dõi chỉ số theo thời gian thực.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Trend analysis
    st.markdown("### 📈 Phân tích xu hướng theo thời gian")
    date_like_cols = []
    for col in df_raw.columns:
        if any(kw in col.lower() for kw in ['date','time','ngay','thang','nam','month','year','period']):
            date_like_cols.append(col)
    
    if date_like_cols and target_col:
        date_col_use = date_like_cols[0]
        try:
            trend_df = df_raw[[date_col_use, target_col]].copy()
            trend_df[date_col_use] = pd.to_datetime(trend_df[date_col_use], errors='coerce')
            trend_df = trend_df.dropna()
            if len(trend_df) > 10:
                trend_df['Month'] = trend_df[date_col_use].dt.to_period('M').astype(str)
                monthly = trend_df.groupby('Month').size().reset_index(name='Count')
                monthly = monthly.tail(24)
                fig_trend = px.line(monthly, x='Month', y='Count',
                                    title=f"Xu hướng theo tháng",
                                    markers=True, color_discrete_sequence=['#00a8b5'])
                fig_trend.update_layout(font_family='Inter')
                st.plotly_chart(fig_trend, use_container_width=True)
        except:
            pass
    else:
        # Show value distribution over index
        if target_col and target_col in df_ml.columns:
            sample_trend = df_ml[target_col].reset_index(drop=True)
            window = max(1, len(sample_trend)//50)
            rolling_mean = sample_trend.rolling(window=window).mean()
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(y=rolling_mean, name='Xu hướng (rolling mean)',
                                        line=dict(color='#00a8b5', width=2)))
            fig_t.update_layout(title=f"Xu hướng biến {target_col} theo thứ tự dữ liệu",
                                  font_family='Inter', height=300)
            st.plotly_chart(fig_t, use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#9ca3af; padding:1rem; font-size:0.8rem;">
        🔬 Data Mining Platform • Powered by Scikit-learn & Streamlit • Dành cho lãnh đạo
    </div>
    """, unsafe_allow_html=True)
