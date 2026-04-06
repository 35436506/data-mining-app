"""
Data Mining Platform - Inspired by Analytic Solver Data Mining (Chapter 10)
============================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Data Mining Platform",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Header ─────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(120deg, #0a1628 0%, #0d2347 50%, #1a3a5c 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(10,22,40,0.4);
}
.app-header::before {
    content: '⛏️';
    position: absolute; right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 5rem; opacity: 0.08;
}
.app-header h1 { font-size: 2rem; font-weight: 700; margin: 0; color: #e8f4fd; }
.app-header p  { font-size: 0.9rem; color: #89b4d4; margin: 0.4rem 0 0; }
.app-header .version { font-size: 0.72rem; color: #4a7fa8; font-family: 'DM Mono'; margin-top: 0.8rem; }

/* ── Step Cards ─────────────────────────────────────────── */
.step-card {
    background: white;
    border: 1px solid #e8edf2;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.step-badge {
    display: inline-block;
    background: #0d2347;
    color: white;
    border-radius: 20px;
    padding: 0.15rem 0.7rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 0.6rem;
}
.step-title {
    font-size: 1rem;
    font-weight: 700;
    color: #0d2347;
    margin-bottom: 0.4rem;
}

/* ── Metric Cards ───────────────────────────────────────── */
.metric-row { display: flex; gap: 0.8rem; margin-bottom: 1rem; }
.metric-card {
    flex: 1;
    background: white;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border-left: 3px solid #1a6db5;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-card .m-label { font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; }
.metric-card .m-value { font-size: 1.6rem; font-weight: 700; color: #0d2347; line-height: 1.2; }
.metric-card .m-sub   { font-size: 0.75rem; color: #94a3b8; }

/* ── Alerts ─────────────────────────────────────────────── */
.alert-info    { background:#eff6ff; border-left:4px solid #2563eb; border-radius:8px; padding:0.9rem 1.1rem; margin:0.5rem 0; font-size:0.88rem; }
.alert-success { background:#f0fdf4; border-left:4px solid #16a34a; border-radius:8px; padding:0.9rem 1.1rem; margin:0.5rem 0; font-size:0.88rem; }
.alert-warning { background:#fffbeb; border-left:4px solid #d97706; border-radius:8px; padding:0.9rem 1.1rem; margin:0.5rem 0; font-size:0.88rem; }
.alert-error   { background:#fef2f2; border-left:4px solid #dc2626; border-radius:8px; padding:0.9rem 1.1rem; margin:0.5rem 0; font-size:0.88rem; }

/* ── Guide Panel ─────────────────────────────────────────── */
.guide-panel {
    background: linear-gradient(135deg, #f8faff 0%, #eef4ff 100%);
    border: 1px solid #c7d9f7;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.8rem 0;
}
.guide-panel h4 { color: #1e40af; font-size: 0.9rem; margin: 0 0 0.5rem; }
.guide-panel p  { color: #334155; font-size: 0.84rem; margin: 0; line-height: 1.6; }

/* ── Recommendation ──────────────────────────────────────── */
.rec-card {
    background: linear-gradient(135deg, #0d2347, #1a4a7a);
    color: white;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.8rem 0;
}
.rec-card h4 { color: #7dd3fc; margin: 0 0 0.5rem; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.8px; }
.rec-card p  { margin: 0; font-size: 0.92rem; line-height: 1.6; }

/* ── Target Selector Guide ───────────────────────────────── */
.target-guide {
    background: #fafafa;
    border: 1.5px dashed #cbd5e1;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.82rem;
    color: #475569;
}

/* ── Accuracy Badge ──────────────────────────────────────── */
.acc-badge {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1rem;
}
.acc-high   { background: #dcfce7; color: #15803d; }
.acc-medium { background: #fef3c7; color: #92400e; }
.acc-low    { background: #fee2e2; color: #991b1b; }

div[data-testid="stTabs"] [data-baseweb="tab"] { font-weight: 600; font-size: 0.88rem; }
.stButton > button { border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <h1>⛏️ Data Mining Platform</h1>
    <p>Nền tảng Khai thác Dữ liệu thông minh · Inspired by Analytic Solver Data Mining (Chapter 10)</p>
    <div class="version">Classification · Prediction · Association · Clustering · v2.0</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def smart_merge(dfs_dict):
    """Intelligently merge multiple dataframes."""
    if len(dfs_dict) == 1:
        name, df = list(dfs_dict.items())[0]
        return df, {name: df.shape}, "single"

    # Find common columns
    all_cols = [set(df.columns.str.lower()) for df in dfs_dict.values()]
    common = all_cols[0].intersection(*all_cols[1:])
    id_kw = ['id','key','num','code','jobnum','reqid','equipid','no','number','ref']
    join_keys = [c for c in common if any(k in c for k in id_kw)]

    df_list = list(dfs_dict.values())
    names   = list(dfs_dict.keys())
    shapes  = {n: df.shape for n, df in dfs_dict.items()}

    if join_keys:
        key_lower = join_keys[0]
        key_maps  = []
        for df in df_list:
            match = next((c for c in df.columns if c.lower() == key_lower), None)
            key_maps.append(match)
        if all(key_maps):
            merged = df_list[0].rename(columns={key_maps[0]: key_lower})
            for i in range(1, len(df_list)):
                right = df_list[i].rename(columns={key_maps[i]: key_lower})
                suffix = f"_{names[i].split('.')[0][:8]}"
                merged = merged.merge(right, on=key_lower, how='outer', suffixes=('', suffix))
            shapes['[Merged]'] = merged.shape
            return merged, shapes, f"join:{key_lower}"
    
    # Try concat
    try:
        merged = pd.concat(df_list, ignore_index=True)
        shapes['[Stacked]'] = merged.shape
        return merged, shapes, "concat"
    except:
        return df_list[0], shapes, "first"


def get_smart_target_suggestions(df):
    """Suggest good target columns with explanations."""
    suggestions = []
    for col in df.columns:
        s = df[col]
        nuniq = s.nunique()
        miss  = s.isnull().mean()
        if miss > 0.5 or nuniq < 2:
            continue
        dtype = str(s.dtype)
        
        if nuniq == 2:
            pct = s.value_counts(normalize=True).min() * 100
            imbalance = "⚠️ Mất cân bằng" if pct < 10 else "✅ Cân bằng tốt"
            suggestions.append({
                'col': col, 'type': 'Phân loại nhị phân (Binary)',
                'nuniq': nuniq, 'task': 'Classification',
                'note': f"{imbalance} — Nhóm thiểu số: {pct:.1f}%",
                'score': 3 if pct >= 10 else 2
            })
        elif 3 <= nuniq <= 15 and ('object' in dtype or 'bool' in dtype or 'int' in dtype):
            suggestions.append({
                'col': col, 'type': f'Phân loại đa lớp ({nuniq} nhóm)',
                'nuniq': nuniq, 'task': 'Classification',
                'note': f"Phù hợp phân loại nhiều nhóm",
                'score': 2
            })
        elif nuniq > 15 and ('float' in dtype or 'int' in dtype):
            suggestions.append({
                'col': col, 'type': 'Dự báo liên tục (Regression)',
                'nuniq': nuniq, 'task': 'Regression',
                'note': f"Biến số, phù hợp dự báo giá trị",
                'score': 2
            })
    
    suggestions.sort(key=lambda x: -x['score'])
    return suggestions


def clean_data(df, target_col, log_steps):
    df = df.copy()
    original_rows = len(df)
    
    # 1. Drop near-empty cols (>85% missing)
    high_miss = [c for c in df.columns if df[c].isnull().mean() > 0.85 and c != target_col]
    if high_miss:
        df.drop(columns=high_miss, inplace=True)
        log_steps.append(f"🗑️ **Loại bỏ cột gần trống:** {len(high_miss)} cột có >85% dữ liệu trống bị xóa để giảm nhiễu.")
    
    # 2. Drop datetime-like columns
    date_cols = []
    for col in df.columns:
        if col == target_col: continue
        if 'datetime' in str(df[col].dtype) or 'date' in str(df[col].dtype):
            date_cols.append(col)
        elif df[col].dtype == 'object':
            try:
                sample = df[col].dropna().head(30)
                parsed = pd.to_datetime(sample, errors='coerce')
                if parsed.notna().mean() > 0.75:
                    date_cols.append(col)
            except: pass
    if date_cols:
        df.drop(columns=[c for c in date_cols if c in df.columns], inplace=True, errors='ignore')
        log_steps.append(f"📅 **Loại bỏ cột ngày giờ:** `{', '.join(date_cols[:4])}` không có ý nghĩa toán học trong ML.")
    
    # 3. Fill numeric
    num_cols = df.select_dtypes(include='number').columns
    filled_num = []
    for col in num_cols:
        if col == target_col: continue
        miss = df[col].isnull().sum()
        if miss > 0:
            med = df[col].median()
            df[col].fillna(med, inplace=True)
            filled_num.append(f"`{col}` ({miss} ô → {med:.2f})")
    if filled_num:
        log_steps.append(f"🔢 **Lấp đầy số trống bằng trung vị:** {'; '.join(filled_num[:4])}{'...' if len(filled_num)>4 else ''}")
    
    # 4. Fill categorical
    cat_cols = df.select_dtypes(include='object').columns
    filled_cat = []
    for col in cat_cols:
        if col == target_col: continue
        miss = df[col].isnull().sum()
        if miss > 0 and not df[col].mode().empty:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            filled_cat.append(f"`{col}` ({miss} ô → '{mode_val}')")
    if filled_cat:
        log_steps.append(f"🔤 **Lấp đầy văn bản trống bằng mode:** {'; '.join(filled_cat[:3])}{'...' if len(filled_cat)>3 else ''}")
    
    return df, log_steps


def encode_data(df, target_col, log_steps):
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    encoders = {}
    
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    
    # Drop high-cardinality ID-like cols
    to_drop = []
    for col in cat_cols:
        nuniq = df[col].nunique()
        threshold = min(50, len(df) * 0.4)
        if nuniq > threshold:
            to_drop.append(col)
    if to_drop:
        df.drop(columns=to_drop, inplace=True, errors='ignore')
        cat_cols = [c for c in cat_cols if c not in to_drop]
        log_steps.append(f"🚫 **Loại ID/mã định danh:** {len(to_drop)} cột có quá nhiều giá trị duy nhất bị loại (`{', '.join(to_drop[:3])}`). Chúng là số định danh, không phải đặc trưng.")
    
    # Encode
    encoded = []
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            encoded.append(col)
    
    if encoded:
        log_steps.append(f"🔡→🔢 **Label Encoding:** {len(encoded)} cột văn bản → số nguyên. Lý do: thuật toán ML chỉ hiểu số, không hiểu chữ. Các cột: `{', '.join(encoded[:5])}{'...' if len(encoded)>5 else ''}`")
    
    # Encode target
    target_encoder = None
    if target_col and target_col in df.columns and df[target_col].dtype == 'object':
        le = LabelEncoder()
        orig_classes = df[target_col].unique().tolist()
        df[target_col] = le.fit_transform(df[target_col].astype(str))
        target_encoder = le
        log_steps.append(f"🎯 **Mã hóa Target:** `{target_col}` → {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    return df, log_steps, encoders, target_encoder


def check_balance(X, y, task_type):
    from collections import Counter
    steps = []
    if 'Phân loại' not in task_type:
        return X, y, steps, False
    
    counts = Counter(y)
    total  = len(y)
    if len(counts) < 2:
        return X, y, steps, False
    
    min_pct = min(counts.values()) / total * 100
    
    if min_pct < 20 and len(X) > 100:
        steps.append(f"⚠️ **Mất cân bằng dữ liệu!** Nhóm thiểu số chiếm {min_pct:.1f}% — mô hình dễ 'học vẹt' theo nhóm đa số và bỏ qua nhóm quan trọng.")
        try:
            from imblearn.over_sampling import SMOTE
            k = max(1, min(5, min(counts.values()) - 1))
            if k >= 1 and min(counts.values()) > 1:
                sm = SMOTE(k_neighbors=k, random_state=42)
                X_res, y_res = sm.fit_resample(X, y)
                new_counts = Counter(y_res)
                steps.append(f"✅ **SMOTE kích hoạt:** {total} → {len(X_res)} mẫu. Tạo dữ liệu tổng hợp cho nhóm hiếm. Phân phối mới: {dict(new_counts)}")
                return X_res, y_res, steps, True
        except Exception as e:
            steps.append(f"ℹ️ SMOTE không áp dụng được: {e}. Tiếp tục với dữ liệu gốc.")
    else:
        steps.append(f"✅ **Dữ liệu cân bằng tốt:** Nhóm thiểu số chiếm {min_pct:.1f}%. Không cần SMOTE.")
    
    return X, y, steps, False


def run_models(X_tr, X_te, y_tr, y_te, task, selected):
    from sklearn.linear_model  import LogisticRegression, LinearRegression
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.tree      import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble  import RandomForestClassifier, RandomForestRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC, SVR
    from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                                  precision_score, mean_absolute_error,
                                  r2_score, mean_squared_error, confusion_matrix)
    
    clf_map = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42, class_weight='balanced'),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=400, random_state=42),
        "LDA": LinearDiscriminantAnalysis(),
        "KNN": KNeighborsClassifier(n_neighbors=min(7, max(1, len(X_tr)//20))),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_leaf=3, random_state=42, class_weight='balanced'),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True, random_state=42, class_weight='balanced'),
    }
    reg_map = {
        "Linear Regression": LinearRegression(),
        "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64,32), max_iter=400, random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=min(7, max(1, len(X_tr)//20))),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42),
        "SVM": SVR(),
    }
    
    is_clf = 'Phân loại' in task
    model_map = clf_map if is_clf else reg_map
    results = {}
    
    for name in selected:
        if name not in model_map: continue
        try:
            m = model_map[name]
            m.fit(X_tr, y_tr)
            yp = m.predict(X_te)
            if is_clf:
                avg = 'binary' if len(np.unique(y_te)) == 2 else 'weighted'
                results[name] = {
                    'model': m, 'y_pred': yp,
                    'accuracy':  accuracy_score(y_te, yp),
                    'f1':        f1_score(y_te, yp, average=avg, zero_division=0),
                    'recall':    recall_score(y_te, yp, average=avg, zero_division=0),
                    'precision': precision_score(y_te, yp, average=avg, zero_division=0),
                    'cm': confusion_matrix(y_te, yp)
                }
            else:
                results[name] = {
                    'model': m, 'y_pred': yp,
                    'r2':  r2_score(y_te, yp),
                    'mae': mean_absolute_error(y_te, yp),
                    'rmse': np.sqrt(mean_squared_error(y_te, yp))
                }
        except Exception as e:
            st.warning(f"⚠️ {name}: {e}")
    return results


def get_feature_importance(model, features):
    try:
        if hasattr(model, 'feature_importances_'):
            return pd.Series(model.feature_importances_, index=features)
        elif hasattr(model, 'coef_'):
            c = model.coef_
            if c.ndim > 1: c = np.abs(c).mean(axis=0)
            return pd.Series(np.abs(c), index=features)
    except: pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📁 Tải lên Dữ liệu")
    uploaded = st.file_uploader(
        "CSV / XLSX (tối đa 50MB mỗi file)",
        type=["csv","xlsx"], accept_multiple_files=True
    )
    
    st.markdown("---")
    st.markdown("### 📖 Hướng dẫn sử dụng")
    
    with st.expander("🔰 Bắt đầu — Chọn Target là gì?", expanded=False):
        st.markdown("""
**Target (Biến mục tiêu)** là cột bạn muốn mô hình *học* để dự đoán.

**Ví dụ thực tế:**
- Cột `Closed` (True/False) → Dự đoán job có đóng không  
- Cột `IssueTopic 1` (PM/BD/SAF) → Phân loại loại sự cố  
- Cột `DownTime` (số) → Dự báo thời gian dừng máy  

**Quy tắc chọn Target:**  
✅ Nên chọn cột có **2–15 giá trị** khác nhau  
✅ Cột có ý nghĩa kinh doanh rõ ràng  
❌ Không chọn cột ID, mã số duy nhất  
❌ Không chọn cột ngày giờ  

> 💡 App sẽ **tự động gợi ý** các cột phù hợp nhất!
        """)
    
    with st.expander("🎯 Chọn Task (Bài toán) như thế nào?"):
        st.markdown("""
**Phân loại (Classification):**  
Target là *nhóm/danh mục* (chữ hoặc số nguyên nhỏ)  
→ VD: Closed=True/False, Loại sự cố PM/BD...

**Dự báo (Regression):**  
Target là *số thực liên tục*  
→ VD: DownTime, Thời gian sửa chữa...

**Luật kết hợp (Association):**  
Không cần target, tìm pattern "nếu A thì B"  
→ VD: Nếu máy X hỏng thì thường hỏng gì tiếp?
        """)
    
    with st.expander("🤖 Giải thích các thuật toán"):
        algo_guide = {
            "Logistic Regression": ("Phân loại", "Tính xác suất thuộc nhóm. Đơn giản, nhanh, dễ giải thích. Tốt khi mối quan hệ tuyến tính."),
            "LDA": ("Phân loại", "Tìm đường biên phân cách tốt nhất giữa các nhóm. Tốt khi dữ liệu phân phối chuẩn."),
            "KNN": ("Phân loại/Dự báo", "Phân loại dựa trên K điểm gần nhất. Trực quan, không giả định phân phối."),
            "Decision Tree": ("Phân loại/Dự báo", "Tạo cây quyết định if-then dễ đọc. Lãnh đạo dễ hiểu nhất."),
            "Random Forest": ("Phân loại/Dự báo", "100+ cây quyết định, bỏ phiếu đa số. Độ chính xác cao, ổn định."),
            "Naive Bayes": ("Phân loại", "Xác suất có điều kiện Bayes. Rất nhanh, tốt cho dữ liệu phân loại."),
            "Neural Network": ("Phân loại/Dự báo", "Mạng nơ-ron nhân tạo, học pattern phức tạp. Cần nhiều dữ liệu."),
            "SVM": ("Phân loại/Dự báo", "Tìm siêu phẳng phân cách tối ưu. Hiệu quả với chiều cao."),
            "Linear Regression": ("Dự báo", "Mô hình tuyến tính cổ điển. Nhanh, dễ giải thích hệ số."),
            "K-Means": ("Phân cụm", "Chia dữ liệu thành K cụm. Cần chỉ định K trước."),
            "Hierarchical": ("Phân cụm", "Xây dendrogram phân cấp. Không cần chỉ định K, xem cây để quyết định."),
            "Apriori": ("Luật kết hợp", "Tìm {A}→{B}. Support, Confidence, Lift đánh giá sức mạnh luật."),
        }
        for algo, (task, desc) in algo_guide.items():
            st.markdown(f"**{algo}** `{task}`")
            st.caption(desc)
            st.markdown("")
    
    with st.expander("📊 Đọc kết quả như thế nào?"):
        st.markdown("""
**Accuracy:** % dự đoán đúng tổng thể  
**F1-Score:** Cân bằng Precision + Recall (tốt hơn Accuracy khi mất cân bằng)  
**Precision:** Trong số dự đoán "có", bao nhiêu đúng thật?  
**Recall (Sensitivity):** Trong số "có" thật, bao nhiêu được tìm ra?  
**R² (Regression):** Gần 1 = tốt, gần 0 = kém  
**MAE:** Sai số trung bình tuyệt đối  

**Confusion Matrix:**  
- Đường chéo = dự đoán đúng  
- Ngoài chéo = sai  

**Lift > 1.5** → Luật kết hợp có ý nghĩa thực tế
        """)
    
    st.markdown("---")
    st.caption("⚠️ Giới hạn: <50MB/file | CSV, XLSX")
    st.caption("📖 Dựa trên: Spreadsheet Modeling & Decision Analysis Ch.10")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN — NO FILES
# ══════════════════════════════════════════════════════════════════════════════
if not uploaded:
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="step-card">
            <div class="step-badge">BƯỚC 1</div>
            <div class="step-title">📁 Tải lên dữ liệu</div>
            Sử dụng nút tải lên ở cột bên trái. Hỗ trợ nhiều file — hệ thống tự động kết nối các bảng có cột ID chung.
        </div>
        <div class="step-card">
            <div class="step-badge">BƯỚC 2</div>
            <div class="step-title">🎯 Chọn Target & Bài toán</div>
            App sẽ <b>tự gợi ý</b> cột mục tiêu phù hợp nhất. Chọn loại bài toán: phân loại, dự báo, hay luật kết hợp.
        </div>
        <div class="step-card">
            <div class="step-badge">BƯỚC 3</div>
            <div class="step-title">🤖 Chọn mô hình & Chạy</div>
            Chọn các thuật toán muốn so sánh, nhấn "Chạy phân tích". App tự động làm sạch, mã hóa, và cân bằng dữ liệu.
        </div>
        <div class="step-card">
            <div class="step-badge">BƯỚC 4</div>
            <div class="step-title">📊 Xem Dashboard kết quả</div>
            So sánh hiệu suất, xem Feature Importance, và nhận khuyến nghị hành động cho lãnh đạo.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="guide-panel">
            <h4>💡 Ví dụ với file của bạn</h4>
            <p>
            <b>MaintenanceData.xlsx + Operation.xlsx</b><br><br>
            → Hệ thống join qua <code>JobNum</code> / <code>EquipID</code><br><br>
            <b>Target gợi ý:</b><br>
            • <code>IssueTopic 1</code> — Phân loại loại sự cố (PM, BD, SAF...)<br>
            • <code>Closed</code> — Dự đoán job có đóng không<br>
            • <code>Task Completed</code> — Dự đoán hoàn thành<br>
            • <code>DownTime</code> — Dự báo thời gian dừng máy<br>
            • <code>IsKeyMachine_c</code> — Phân loại máy trọng yếu
            </p>
        </div>
        <div class="guide-panel" style="margin-top:0.8rem;">
            <h4>⚠️ Lỗi thường gặp</h4>
            <p>
            <b>"Cột mục tiêu không hợp lệ"</b><br>
            → Chọn cột có 2–15 giá trị khác nhau<br>
            → Không chọn cột ID, mã số<br>
            → Hãy dùng <b>gợi ý tự động</b> của app!
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
for f in uploaded:
    if f.size > 50 * 1024 * 1024:
        st.error(f"⛔ {f.name} vượt 50MB")
        st.stop()

dfs_dict = {}
with st.spinner("📖 Đang đọc dữ liệu..."):
    for f in uploaded:
        try:
            if f.name.endswith('.csv'):
                dfs_dict[f.name] = pd.read_csv(f)
            else:
                xl = pd.read_excel(f, sheet_name=None)
                # Keep largest sheet per file, also expose all sheets
                for sname, sdf in xl.items():
                    key = f"{f.name} [{sname}]"
                    dfs_dict[key] = sdf
        except Exception as e:
            st.error(f"Lỗi đọc {f.name}: {e}")

if not dfs_dict:
    st.error("Không đọc được file nào.")
    st.stop()

df_raw, shape_report, merge_method = smart_merge(dfs_dict)

# ══════════════════════════════════════════════════════════════════════════════
# DATA HEALTH REPORT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📋 Báo cáo Sức khoẻ Dữ liệu")

total_cells = df_raw.shape[0] * df_raw.shape[1]
missing_pct = df_raw.isnull().sum().sum() / total_cells * 100

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card"><div class="m-label">Tổng dòng</div><div class="m-value">{df_raw.shape[0]:,}</div><div class="m-sub">Records</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card"><div class="m-label">Tổng cột</div><div class="m-value">{df_raw.shape[1]}</div><div class="m-sub">Features</div></div>""", unsafe_allow_html=True)
with c3:
    clr = "#dc2626" if missing_pct > 30 else ("#d97706" if missing_pct > 10 else "#16a34a")
    st.markdown(f"""<div class="metric-card" style="border-color:{clr}"><div class="m-label">Ô trống</div><div class="m-value" style="color:{clr}">{missing_pct:.1f}%</div><div class="m-sub">Missing data</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card"><div class="m-label">Bảng đã nạp</div><div class="m-value">{len(shape_report)}</div><div class="m-sub">Files / Sheets</div></div>""", unsafe_allow_html=True)

with st.expander("🔍 Chi tiết file & cột", expanded=False):
    for fname, shape in shape_report.items():
        icon = "🔗" if '[Merged]' in fname or '[Stacked]' in fname else "📄"
        st.markdown(f"{icon} **{fname}** — {shape[0]:,} dòng × {shape[1]} cột")
    
    if merge_method.startswith("join"):
        key = merge_method.split(":")[1]
        st.markdown(f"""<div class="alert-success">✅ Kết nối tự động qua cột khóa <code>{key}</code></div>""", unsafe_allow_html=True)
    
    col_df = pd.DataFrame({
        'Cột': df_raw.columns,
        'Kiểu': df_raw.dtypes.astype(str).values,
        'Trống (%)': (df_raw.isnull().mean() * 100).round(1).values,
        'Duy nhất': df_raw.nunique().values,
        'Ví dụ': [str(df_raw[c].dropna().iloc[0]) if df_raw[c].notna().any() else 'N/A' for c in df_raw.columns]
    })
    st.dataframe(col_df.style.background_gradient(subset=['Trống (%)'], cmap='RdYlGn_r'), use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TARGET SELECTION & TASK TYPE (Smart)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## ⚙️ Cấu hình Phân tích")

suggestions = get_smart_target_suggestions(df_raw)

col_left, col_right = st.columns([1.2, 1])

with col_left:
    # Show smart suggestions
    if suggestions:
        st.markdown("**🎯 Gợi ý cột mục tiêu (Target) phù hợp nhất:**")
        
        for s in suggestions[:5]:
            task_icon = "🔮" if s['task'] == 'Classification' else "📈"
            st.markdown(f"""
            <div class="target-guide">
                {task_icon} <b><code>{s['col']}</code></b> — {s['type']}<br>
                <span style="color:#64748b">{s['note']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        default_target = suggestions[0]['col']
    else:
        st.markdown("""<div class="alert-warning">⚠️ Không tìm thấy cột target phù hợp tự động. Vui lòng chọn thủ công.</div>""", unsafe_allow_html=True)
        default_target = df_raw.columns[0]
    
    all_cols = df_raw.columns.tolist()
    try:
        default_idx = all_cols.index(default_target)
    except:
        default_idx = 0
    
    target_col = st.selectbox(
        "Chọn cột mục tiêu:",
        ["(Không chọn — chỉ phân cụm/luật)"] + all_cols,
        index=default_idx + 1,
        help="Chọn cột bạn muốn dự đoán. App đã tự gợi ý cột tốt nhất ở trên."
    )
    if target_col == "(Không chọn — chỉ phân cụm/luật)":
        target_col = None

with col_right:
    task_type = st.radio(
        "Loại bài toán:",
        ["🔮 Phân loại (Classification)", "📈 Dự báo (Regression)", "🔗 Luật kết hợp (Association)"],
        help="Classification: target là nhóm | Regression: target là số liên tục | Association: không cần target"
    )
    
    exclude_cols = st.multiselect(
        "Loại trừ cột (tuỳ chọn):",
        [c for c in all_cols if c != target_col],
        placeholder="Chọn cột không cần dùng..."
    )

# Validate target
if target_col:
    nuniq = df_raw[target_col].nunique()
    if nuniq > 50 and 'Phân loại' in task_type:
        st.markdown(f"""<div class="alert-warning">⚠️ Cột <code>{target_col}</code> có <b>{nuniq}</b> giá trị duy nhất — quá nhiều cho Classification. Hãy chọn cột khác hoặc đổi sang Regression.</div>""", unsafe_allow_html=True)
    elif nuniq < 2:
        st.markdown(f"""<div class="alert-error">❌ Cột <code>{target_col}</code> chỉ có 1 giá trị duy nhất — không dùng được làm target.</div>""", unsafe_allow_html=True)
    else:
        vc = df_raw[target_col].value_counts()
        st.markdown(f"""<div class="alert-success">✅ Target: <code>{target_col}</code> | {nuniq} nhóm | Top: {', '.join([f'{k}({v})' for k,v in vc.head(4).items()])}</div>""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL SELECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🤖 Chọn Mô hình Phân tích")

if task_type == "🔮 Phân loại (Classification)":
    avail = ["Logistic Regression","LDA","KNN","Decision Tree","Random Forest","Naive Bayes","Neural Network (MLP)","SVM"]
    defaults = ["Logistic Regression","Decision Tree","Random Forest"]
elif task_type == "📈 Dự báo (Regression)":
    avail = ["Linear Regression","KNN","Decision Tree","Random Forest","Neural Network (MLP)","SVM"]
    defaults = ["Linear Regression","Decision Tree","Random Forest"]
else:
    avail = ["K-Means Clustering","Hierarchical Clustering","Apriori (Association Rules)"]
    defaults = avail

selected_models = st.multiselect("Chọn thuật toán muốn chạy:", avail, default=defaults)

run_btn = st.button("🚀 Chạy phân tích", type="primary", use_container_width=True)

if not run_btn:
    with st.expander("👁️ Xem trước dữ liệu"):
        st.dataframe(df_raw.head(30), use_container_width=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🔄 Quy trình Xử lý có Giải thích (Explainable AI Pipeline)")

df_work = df_raw.copy()
if exclude_cols:
    df_work.drop(columns=[c for c in exclude_cols if c in df_work.columns], inplace=True, errors='ignore')

# STEP 1: Clean
with st.status("🧹 Bước 1 — Làm sạch Dữ liệu (Data Cleaning)", expanded=True) as s1:
    clean_steps = []
    df_work, clean_steps = clean_data(df_work, target_col, clean_steps)
    for step in clean_steps:
        st.markdown(step)
    if not clean_steps:
        st.markdown("✅ Dữ liệu sạch, không cần xử lý thêm.")
    s1.update(label=f"✅ Bước 1 hoàn tất — {len(clean_steps)} thao tác", state="complete")

# STEP 2: Encode
with st.status("🔡 Bước 2 — Mã hoá Dữ liệu (Feature Encoding)", expanded=True) as s2:
    encode_steps = []
    df_encoded, encode_steps, encoders, target_enc = encode_data(df_work, target_col, encode_steps)
    for step in encode_steps:
        st.markdown(step)
    if not encode_steps:
        st.markdown("✅ Tất cả cột đã ở dạng số, không cần mã hóa.")
    s2.update(label=f"✅ Bước 2 hoàn tất — {len(encode_steps)} thao tác", state="complete")

# Handle Association Rules separately
if task_type == "🔗 Luật kết hợp (Association)":
    with st.status("🔗 Chạy Apriori Association Rules...", expanded=True) as s_ap:
        try:
            from mlxtend.frequent_patterns import apriori, association_rules
            from mlxtend.preprocessing import TransactionEncoder
            cat_cols_ap = df_work.select_dtypes(include='object').columns.tolist()[:8]
            if cat_cols_ap:
                transactions = df_work[cat_cols_ap].astype(str).values.tolist()
                te = TransactionEncoder()
                te_arr = te.fit_transform(transactions)
                df_te = pd.DataFrame(te_arr, columns=te.columns_)
                freq = apriori(df_te, min_support=0.03, use_colnames=True)
                if not freq.empty:
                    rules = association_rules(freq, metric='confidence', min_threshold=0.3)
                    st.markdown(f"✅ Tìm thấy **{len(rules)}** luật kết hợp.")
                else:
                    rules = pd.DataFrame()
                    st.warning("Không đủ pattern để tạo luật. Thử giảm min_support.")
            else:
                rules = pd.DataFrame()
                st.warning("Cần cột văn bản để tạo luật kết hợp.")
            s_ap.update(label="✅ Apriori hoàn tất", state="complete")
        except Exception as e:
            st.error(f"Lỗi Apriori: {e}")
            rules = pd.DataFrame()
            s_ap.update(label="❌ Lỗi", state="error")
    
    st.markdown("---")
    st.markdown("## 📊 Kết quả Luật Kết hợp (Association Rules)")
    if not rules.empty:
        rules_disp = rules.sort_values('lift', ascending=False).head(20).copy()
        rules_disp['antecedents'] = rules_disp['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_disp['consequents'] = rules_disp['consequents'].apply(lambda x: ', '.join(list(x)))
        rules_disp.columns = [c if c not in ['antecedents','consequents','support','confidence','lift'] 
                               else {'antecedents':'Nếu...','consequents':'Thì...','support':'Support','confidence':'Confidence','lift':'Lift'}[c]
                               for c in rules_disp.columns]
        st.dataframe(rules_disp[['Nếu...','Thì...','Support','Confidence','Lift']].style.background_gradient(subset=['Lift'], cmap='Greens'), use_container_width=True)
        st.markdown("""<div class="rec-card"><h4>💡 Đọc kết quả</h4><p>Lift > 1.5 = mối quan hệ có ý nghĩa thực tế. Confidence = xác suất "Thì..." xảy ra khi "Nếu..." đã xảy ra. Support = tần suất toàn bộ luật xuất hiện.</p></div>""", unsafe_allow_html=True)
    st.stop()

# Continue for Classification/Regression
df_ml = df_encoded.select_dtypes(include='number').copy()

if target_col and target_col not in df_ml.columns:
    # Try to put target back from original
    st.error(f"❌ Cột target `{target_col}` không còn sau xử lý. Thử chọn cột khác.")
    st.stop()

feature_cols = [c for c in df_ml.columns if c != target_col]
if len(feature_cols) < 1:
    st.error("❌ Không đủ cột đặc trưng sau xử lý.")
    st.stop()

results = {}
X_train_sc = X_test_sc = y_train = y_test = None
cluster_results = {}

if target_col:
    X_full = df_ml[feature_cols].fillna(0).values
    y_full = df_ml[target_col].fillna(df_ml[target_col].mode()[0] if not df_ml[target_col].mode().empty else 0).values
    
    # STEP 3: Balance
    with st.status("⚖️ Bước 3 — Kiểm tra Cân bằng Dữ liệu", expanded=True) as s3:
        X_full, y_full, bal_steps, smote_used = check_balance(X_full, y_full, task_type)
        for step in bal_steps:
            st.markdown(step)
        s3.update(label="✅ Bước 3 hoàn tất", state="complete")
    
    # STEP 4: Train & Evaluate
    with st.status("✂️ Bước 4 — Phân chia Train/Test (80/20) & Huấn luyện Mô hình", expanded=True) as s4:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing   import StandardScaler
        
        X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
        sc = StandardScaler()
        X_train_sc = sc.fit_transform(X_tr)
        X_test_sc  = sc.transform(X_te)
        
        st.markdown(f"📊 **Train:** {len(X_tr):,} mẫu | **Test:** {len(X_te):,} mẫu | **Features:** {len(feature_cols)}")
        st.markdown("🔄 Đang huấn luyện mô hình...")
        
        clf_models = [m for m in selected_models if m in ["Logistic Regression","LDA","KNN","Decision Tree","Random Forest","Naive Bayes","Neural Network (MLP)","SVM","Linear Regression"]]
        results = run_models(X_train_sc, X_test_sc, y_tr, y_te, task_type, clf_models)
        
        st.markdown(f"✅ Huấn luyện xong **{len(results)}** mô hình.")
        s4.update(label=f"✅ Bước 4 hoàn tất — {len(results)} mô hình", state="complete")

# Clustering
df_num = df_ml[feature_cols].fillna(0)
if "K-Means Clustering" in selected_models and len(df_num) > 20:
    with st.status("🎯 K-Means Clustering...", expanded=False) as sk:
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            sc_cl = StandardScaler()
            X_cl = sc_cl.fit_transform(df_num.head(3000).values)
            km = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels = km.fit_predict(X_cl)
            cluster_results['K-Means'] = {'labels': labels, 'data': df_num.head(3000)}
            sk.update(label="✅ K-Means hoàn tất", state="complete")
        except Exception as e:
            sk.update(label=f"❌ K-Means: {e}", state="error")

if "Hierarchical Clustering" in selected_models and len(df_num) > 20:
    with st.status("🌲 Hierarchical Clustering...", expanded=False) as sh:
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.preprocessing import StandardScaler
            n_sample = min(500, len(df_num))
            sc_hc = StandardScaler()
            X_hc = sc_hc.fit_transform(df_num.head(n_sample).values)
            hc = AgglomerativeClustering(n_clusters=3)
            labels_hc = hc.fit_predict(X_hc)
            cluster_results['Hierarchical'] = {'labels': labels_hc, 'data': df_num.head(n_sample)}
            sh.update(label="✅ Hierarchical hoàn tất", state="complete")
        except Exception as e:
            sh.update(label=f"❌ HC: {e}", state="error")

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📊 Executive Dashboard")

tab1, tab2, tab3 = st.tabs(["📈 Tab 1: Tổng quan Dữ liệu", "🏆 Tab 2: Hiệu suất Mô hình", "💡 Tab 3: Insight & Khuyến nghị"])

# ── TAB 1 ──────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🗺️ Correlation Heatmap (Ma trận tương quan)")
    num_df = df_ml.select_dtypes(include='number')
    if num_df.shape[1] > 1:
        corr = num_df.corr()
        fig_hm = px.imshow(corr, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                           title="Mức độ tương quan giữa các biến", aspect='auto')
        fig_hm.update_layout(height=min(600, 100 + 30*len(num_df.columns)), font_family='DM Sans')
        st.plotly_chart(fig_hm, use_container_width=True)
        
        # Find strong correlations
        strong = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.7:
                    strong.append(f"`{corr.columns[i]}` ↔ `{corr.columns[j]}`: **{val:.2f}**")
        if strong:
            st.markdown(f"""<div class="alert-info">🔗 <b>Tương quan mạnh (>0.7):</b> {' | '.join(strong[:5])}</div>""", unsafe_allow_html=True)
    
    # Distribution
    st.markdown("### 📊 Phân phối các biến số")
    show_cols = [c for c in num_df.columns if c != target_col][:9]
    if show_cols:
        n_c = min(3, len(show_cols))
        n_r = (len(show_cols) + n_c - 1) // n_c
        fig_d = make_subplots(rows=n_r, cols=n_c, subplot_titles=show_cols[:n_r*n_c])
        for i, col in enumerate(show_cols[:n_r*n_c]):
            r, c = divmod(i, n_c)
            fig_d.add_trace(go.Histogram(x=num_df[col].dropna(), name=col, showlegend=False,
                                          marker_color='#1a6db5', opacity=0.75), row=r+1, col=c+1)
        fig_d.update_layout(height=280*n_r, title="Phân phối dữ liệu", font_family='DM Sans')
        st.plotly_chart(fig_d, use_container_width=True)
    
    # Target distribution
    if target_col and target_col in df_raw.columns:
        st.markdown(f"### 🎯 Phân phối Target: `{target_col}`")
        vc = df_raw[target_col].value_counts().reset_index()
        vc.columns = ['Nhóm', 'Số lượng']
        vc['Tỷ lệ %'] = (vc['Số lượng'] / vc['Số lượng'].sum() * 100).round(1)
        fig_t = px.bar(vc, x='Nhóm', y='Số lượng', color='Số lượng',
                       color_continuous_scale='Blues', text='Tỷ lệ %',
                       title=f"Phân phối {target_col}")
        fig_t.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_t.update_layout(font_family='DM Sans', showlegend=False)
        st.plotly_chart(fig_t, use_container_width=True)
        
        min_pct = vc['Tỷ lệ %'].min()
        if min_pct < 10:
            st.markdown(f"""<div class="alert-warning">⚠️ <b>Mất cân bằng nghiêm trọng:</b> Nhóm thiểu số chỉ {min_pct:.1f}%. SMOTE đã được kích hoạt để bù đắp.</div>""", unsafe_allow_html=True)
    
    # Clustering viz
    for cname, cdata in cluster_results.items():
        st.markdown(f"### 🔵 Kết quả {cname}")
        data_cl = cdata['data'].copy()
        data_cl['Cụm'] = cdata['labels'].astype(str)
        num_show = data_cl.select_dtypes(include='number').columns.tolist()
        if len(num_show) >= 2:
            fig_cl = px.scatter(data_cl, x=num_show[0], y=num_show[1], color='Cụm',
                                title=f"{cname}: 3 cụm dữ liệu",
                                color_discrete_sequence=px.colors.qualitative.Set2)
            fig_cl.update_layout(font_family='DM Sans')
            st.plotly_chart(fig_cl, use_container_width=True)

# ── TAB 2 ──────────────────────────────────────────────────────────────────
with tab2:
    if not results:
        st.info("Chưa có kết quả mô hình. Chọn target và chạy phân tích.")
    else:
        st.markdown("### 🏆 So sánh Hiệu suất Mô hình")
        
        is_clf = 'Phân loại' in task_type
        
        if is_clf:
            perf = []
            for name, r in results.items():
                perf.append({'Mô hình': name,
                             'Accuracy (%)': round(r['accuracy']*100, 2),
                             'F1-Score': round(r['f1'], 4),
                             'Precision': round(r['precision'], 4),
                             'Recall': round(r['recall'], 4)})
            perf_df = pd.DataFrame(perf).sort_values('Accuracy (%)', ascending=False)
            
            fig_p = px.bar(perf_df, x='Mô hình', y='Accuracy (%)', color='Accuracy (%)',
                           color_continuous_scale='Blues', text='Accuracy (%)',
                           title='Độ chính xác (Accuracy) — trên tập Test')
            fig_p.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_p.update_layout(font_family='DM Sans', height=380, showlegend=False)
            st.plotly_chart(fig_p, use_container_width=True)
            
            # F1 comparison
            fig_f1 = px.bar(perf_df, x='Mô hình', y='F1-Score', color='F1-Score',
                            color_continuous_scale='Greens', text='F1-Score',
                            title='F1-Score — tốt hơn Accuracy khi dữ liệu mất cân bằng')
            fig_f1.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_f1.update_layout(font_family='DM Sans', height=340, showlegend=False)
            st.plotly_chart(fig_f1, use_container_width=True)
            
            st.dataframe(perf_df.style.background_gradient(subset=['Accuracy (%)','F1-Score'], cmap='Greens'), use_container_width=True)
            
            # Best model confusion matrix
            best = perf_df.iloc[0]['Mô hình']
            br   = results[best]
            acc  = br['accuracy'] * 100
            
            if acc >= 80:
                badge_cls, badge_txt = "acc-high", f"✅ Tốt ({acc:.1f}%)"
            elif acc >= 60:
                badge_cls, badge_txt = "acc-medium", f"⚠️ Trung bình ({acc:.1f}%)"
            else:
                badge_cls, badge_txt = "acc-low", f"❌ Cần cải thiện ({acc:.1f}%)"
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""<div class="metric-card"><div class="m-label">Best Model</div><div class="m-value" style="font-size:1rem">{best}</div><div class="m-sub"><span class="acc-badge {badge_cls}">{badge_txt}</span></div></div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card"><div class="m-label">Accuracy</div><div class="m-value">{acc:.1f}%</div><div class="m-sub">Trên tập Test 20%</div></div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card"><div class="m-label">F1-Score</div><div class="m-value">{br['f1']:.3f}</div><div class="m-sub">Weighted avg</div></div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""<div class="metric-card"><div class="m-label">Recall</div><div class="m-value">{br['recall']:.3f}</div><div class="m-sub">Sensitivity</div></div>""", unsafe_allow_html=True)
            
            st.markdown(f"### 🔢 Confusion Matrix — {best}")
            cm = br['cm']
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               title=f"Ma trận nhầm lẫn: {best}",
                               labels={'x':'Dự đoán','y':'Thực tế'})
            fig_cm.update_layout(font_family='DM Sans', height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            st.markdown("""<div class="alert-info">📌 <b>Đọc Confusion Matrix:</b> Đường chéo = dự đoán đúng. Ô ngoài chéo = sai. Cột = nhóm dự đoán, Hàng = nhóm thực tế.</div>""", unsafe_allow_html=True)
        
        else:  # Regression
            perf = []
            for name, r in results.items():
                perf.append({'Mô hình': name,
                             'R² Score': round(r['r2'], 4),
                             'MAE': round(r['mae'], 4),
                             'RMSE': round(r['rmse'], 4)})
            perf_df = pd.DataFrame(perf).sort_values('R² Score', ascending=False)
            
            fig_r2 = px.bar(perf_df, x='Mô hình', y='R² Score', color='R² Score',
                            color_continuous_scale='Blues', text='R² Score',
                            title='R² Score (càng gần 1 càng tốt)')
            fig_r2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_r2.update_layout(font_family='DM Sans', height=380, showlegend=False)
            st.plotly_chart(fig_r2, use_container_width=True)
            
            st.dataframe(perf_df.style.background_gradient(subset=['R² Score'], cmap='Greens'), use_container_width=True)
            
            best_r  = perf_df.iloc[0]['Mô hình']
            best_rs = results[best_r]
            fig_avp = px.scatter(x=y_test[:500], y=best_rs['y_pred'][:500],
                                 labels={'x':'Thực tế','y':'Dự báo'},
                                 title=f"Thực tế vs Dự báo — {best_r}",
                                 trendline='ols')
            fig_avp.update_layout(font_family='DM Sans')
            st.plotly_chart(fig_avp, use_container_width=True)

# ── TAB 3 ──────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🔍 Feature Importance — Yếu tố ảnh hưởng mạnh nhất")
    
    fi_shown = False
    fi_series = None
    best_fi_model = None
    
    if results:
        for preferred in ["Random Forest","Decision Tree","Logistic Regression","Linear Regression"]:
            if preferred in results:
                fi = get_feature_importance(results[preferred]['model'], feature_cols)
                if fi is not None and len(fi) > 0:
                    fi_sorted = fi.sort_values(ascending=False)
                    top15 = fi_sorted.head(15)
                    
                    fig_fi = px.bar(
                        top15.reset_index(),
                        x=top15.values, y=top15.index,
                        orientation='h',
                        color=top15.values,
                        color_continuous_scale='Blues',
                        title=f"Top 15 Yếu tố ảnh hưởng — {preferred}",
                        labels={'x':'Mức độ ảnh hưởng','y':'Yếu tố'}
                    )
                    fig_fi.update_layout(font_family='DM Sans', height=450, showlegend=False,
                                          yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_fi, use_container_width=True)
                    
                    top3 = fi_sorted.head(3)
                    medals = ["🥇","🥈","🥉"]
                    for i, (feat, val) in enumerate(top3.items()):
                        pct = val / fi_sorted.sum() * 100
                        st.markdown(f"""<div class="alert-info">{medals[i]} <b>#{i+1}: {feat}</b> — Mức ảnh hưởng: <b>{pct:.1f}%</b></div>""", unsafe_allow_html=True)
                    
                    fi_series     = fi_sorted
                    best_fi_model = preferred
                    fi_shown      = True
                    break
    
    if not fi_shown:
        st.info("Chạy mô hình phân loại/dự báo để xem Feature Importance.")
    
    st.markdown("---")
    st.markdown("### 💡 Khuyến nghị Chiến lược cho Lãnh đạo")
    
    if results and fi_shown and fi_series is not None:
        top1 = fi_series.index[0]
        top2 = fi_series.index[1] if len(fi_series) > 1 else top1
        
        is_clf = 'Phân loại' in task_type
        best_name = list(results.keys())[0]
        best_metric = results[best_name].get('accuracy', results[best_name].get('r2', 0)) * 100
        
        if is_clf:
            perf_sorted = sorted(results.items(), key=lambda x: -x[1].get('accuracy',0))
        else:
            perf_sorted = sorted(results.items(), key=lambda x: -x[1].get('r2',0))
        
        best_name = perf_sorted[0][0]
        best_val  = perf_sorted[0][1].get('accuracy', perf_sorted[0][1].get('r2', 0)) * 100
        
        metric_label = "Accuracy" if is_clf else "R²"
        
        st.markdown(f"""
        <div class="rec-card">
            <h4>🎯 Hành động ưu tiên #1</h4>
            <p>Yếu tố <b>{top1}</b> có ảnh hưởng lớn nhất đến kết quả. Lãnh đạo nên tập trung theo dõi và kiểm soát chỉ số này để cải thiện hiệu quả vận hành.</p>
        </div>
        <div class="rec-card" style="background:linear-gradient(135deg,#064e3b,#065f46);margin-top:0.8rem;">
            <h4>🤖 Lựa chọn Mô hình triển khai</h4>
            <p>Mô hình <b>{best_name}</b> cho hiệu suất tốt nhất ({metric_label} = <b>{best_val:.1f}%</b>). Khuyến nghị dùng mô hình này cho các quyết định định kỳ và tự động hóa quy trình dự báo.</p>
        </div>
        <div class="rec-card" style="background:linear-gradient(135deg,#312e81,#4c1d95);margin-top:0.8rem;">
            <h4>📊 Cải thiện dữ liệu</h4>
            <p>Thu thập thêm thông tin về <b>{top1}</b> và <b>{top2}</b> vì đây là hai yếu tố quan trọng nhất. Dữ liệu chất lượng cao về 2 biến này sẽ cải thiện độ chính xác đáng kể.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Trend
    st.markdown("---")
    st.markdown("### 📈 Phân tích xu hướng")
    
    if target_col and target_col in df_raw.columns:
        # Find date column
        date_col = None
        for col in df_raw.columns:
            if any(kw in col.lower() for kw in ['date','time','ngay','thang','nam','month','year']):
                try:
                    parsed = pd.to_datetime(df_raw[col], errors='coerce')
                    if parsed.notna().mean() > 0.5:
                        date_col = col
                        break
                except: pass
        
        if date_col:
            try:
                trend_df = df_raw[[date_col, target_col]].copy()
                trend_df[date_col] = pd.to_datetime(trend_df[date_col], errors='coerce')
                trend_df = trend_df.dropna()
                trend_df['Month'] = trend_df[date_col].dt.to_period('M').astype(str)
                monthly = trend_df.groupby('Month').size().reset_index(name='Số lượng')
                monthly = monthly.tail(24)
                fig_tr = px.line(monthly, x='Month', y='Số lượng', markers=True,
                                 title=f"Xu hướng theo tháng (24 tháng gần nhất)",
                                 color_discrete_sequence=['#1a6db5'])
                fig_tr.update_layout(font_family='DM Sans')
                st.plotly_chart(fig_tr, use_container_width=True)
            except Exception as e:
                pass
        else:
            # Rolling mean of target
            target_series = df_ml[target_col].reset_index(drop=True)
            w = max(1, len(target_series)//80)
            rolling = target_series.rolling(w).mean()
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(y=rolling, name='Xu hướng', line=dict(color='#1a6db5', width=2)))
            fig_roll.update_layout(title=f"Xu hướng biến `{target_col}` theo thứ tự dữ liệu",
                                    font_family='DM Sans', height=300)
            st.plotly_chart(fig_roll, use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#94a3b8; padding:1.2rem; font-size:0.8rem; font-family:'DM Mono'">
        ⛏️ Data Mining Platform v2.0 · Scikit-learn · Streamlit · Chapter 10 Analytic Solver
    </div>
    """, unsafe_allow_html=True)
