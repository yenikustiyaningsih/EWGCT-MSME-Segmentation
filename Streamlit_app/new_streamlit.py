"""
╔════════════════════════════════════════════════════════════════════════════╗
║   APLIKASI CLUSTERING UMKM - COMPREHENSIVE VERSION (v3.0)                 ║
║                  Sesuai 100% dengan testTA copy.ipynb                      ║
║  • Feature Selection, Outlier Detection, Distribution Viz                  ║
║  • Baseline vs Weighted vs Top-N Comparison                                ║
║  • Complete EWM Calculation Visualization                                  ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import warnings
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from io import StringIO

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn_extra.cluster import KMedoids

warnings.filterwarnings('ignore')

# ============================================================================
# HELPER FUNCTION FOR MEDOID DISPLAY
# ============================================================================

def display_medoids(medoid_indices, df_data, selected_features, labels, title="Medoid per Cluster"):
    """
    Display medoids information for each cluster.
    medoid_indices: array of medoid indices
    df_data: dataframe with identitas
    selected_features: list of feature columns
    labels: cluster labels
    """
    st.markdown(f"#### 🔗 {title}")
    
    unique_clusters = np.unique(labels)
    
    for cluster_id in sorted(unique_clusters):
        medoid_idx = medoid_indices[cluster_id]
        
        if medoid_idx < len(df_data):
            medoid_row = df_data.iloc[medoid_idx]
            identitas = medoid_row.get('Identitas_UMKM', f'UMKM_{medoid_idx}')
            
            with st.expander(f"**Cluster {cluster_id + 1}** - Medoid: {identitas}"):
                # Display medoid features
                feature_vals = {}
                for feat in selected_features:
                    if feat in medoid_row.index:
                        feature_vals[feat] = medoid_row[feat]
                
                medoid_df = pd.DataFrame({
                    'Feature': feature_vals.keys(),
                    'Value': feature_vals.values()
                }).round(4)
                
                st.dataframe(medoid_df, use_container_width=True, hide_index=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

# ============================================================================
# STREAMLIT CONFIG
# ============================================================================
st.set_page_config(
    page_title="UMKM Clustering - Comprehensive",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = '📊 Analisis Clustering'

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

FEATURE_COLS = [
    'NIB/SKU',
    'Jml. Tenaga Kerja',
    'Kapasitas Produksi/Thn',
    'Omset/Thn',
    'Aset',
    'Sosmed',
    'Marketplace',
    'Kepemilikan Lahan'
]

FEATURE_TYPES = {
    'NIB/SKU'               : 'bin',
    'Jml. Tenaga Kerja'     : 'num',
    'Kapasitas Produksi/Thn': 'num',
    'Omset/Thn'             : 'num',
    'Aset'                  : 'num',
    'Sosmed'                : 'num',
    'Marketplace'           : 'num',
    'Kepemilikan Lahan'     : 'bin'
}

NUMERICAL_COLS   = ['Jml. Tenaga Kerja', 'Kapasitas Produksi/Thn', 'Omset/Thn', 'Aset']
CATEGORICAL_COLS = ['NIB/SKU', 'Sosmed', 'Marketplace', 'Kepemilikan Lahan']
OUTLIER_COLS = ['Jml. Tenaga Kerja', 'Kapasitas Produksi/Thn', 'Omset/Thn', 'Aset']
MIN_TOP_N = 3

# ============================================================================
# ENCODING FUNCTIONS
# ============================================================================

def encode_nib(val, is_raw_data=True):
    """Encode NIB/SKU - handles both raw strings and pre-encoded numeric values"""
    if pd.isna(val):
        return 0
    
    # If already numeric, assume it's pre-encoded
    if isinstance(val, (int, np.integer, float)) and not isinstance(val, (bool, np.bool_)):
        if np.isnan(val) if isinstance(val, float) else False:
            return 0
        return int(val)
    
    # Raw string data
    s = str(val).strip().lower()
    return 1 if s in ['ada', 'yes', '1', 'true'] else 0

def encode_lahan(val, is_raw_data=True):
    """Encode Kepemilikan Lahan - handles both raw strings and pre-encoded numeric values"""
    if pd.isna(val):
        return 0
    
    # If already numeric, assume it's pre-encoded
    if isinstance(val, (int, np.integer, float)) and not isinstance(val, (bool, np.bool_)):
        if np.isnan(val) if isinstance(val, float) else False:
            return 0
        return int(val)
    
    # Raw string data
    s = str(val).strip().lower()
    return 1 if ('milik' in s or 'sendiri' in s) else 0

def count_apps(val, is_raw_data=True):
    """Count unique apps - handles both raw strings and pre-encoded numeric values"""
    if pd.isna(val):
        return 0
    
    # If already numeric, assume it's pre-encoded count
    if isinstance(val, (int, np.integer, float)) and not isinstance(val, (bool, np.bool_)):
        if np.isnan(val) if isinstance(val, float) else False:
            return 0
        return int(val)
    
    # Raw string data
    s = str(val).lower().strip()
    if s in ['', '-', 'tidak', 'tidak ada', 'none', 'nan', '0']:
        return 0
    apps = [a.strip() for a in s.split(',') if a.strip()]
    apps = list(set(apps))
    return len(apps)

# ============================================================================
# PREPROCESSING FUNCTION
# ============================================================================

def detect_data_format(df):
    """
    Detect whether data is raw (string) or pre-processed (numeric).
    Returns: 'raw' or 'processed'
    """
    # Check first column for data types
    check_cols = [c for c in ['NIB/SKU', 'Kepemilikan Lahan', 'Sosmed', 'Marketplace'] 
                  if c in df.columns]
    
    for col in check_cols:
        # If column contains actual strings (not numeric strings), it's raw
        if df[col].dtype == 'object':
            return 'raw'
        # If numeric and we see typical pre-encoded values (0, 1 for binary; 0-N for counts)
        if df[col].dtype in ['int64', 'int32', 'float64']:
            return 'processed'
    
    return 'raw'  # Default to raw

def preprocess_data(df_raw, selected_features=None):
    """Preprocessing sesuai notebook dengan feature selection"""
    
    if selected_features is None:
        selected_features = FEATURE_COLS
    
    df = df_raw[selected_features].copy()
    
    # Identitas UMKM
    if 'Nama Pemilik' in df_raw.columns and 'Nama Usaha' in df_raw.columns:
        identitas = (
            df_raw['Nama Pemilik'].astype(str).str.strip() + ' - ' +
            df_raw['Nama Usaha'].astype(str).str.strip()
        )
    else:
        identitas = pd.Series([f'UMKM_{i}' for i in range(len(df_raw))])
    
    # Drop duplikat
    df_proc = pd.DataFrame({'Identitas_UMKM': identitas})
    df_proc = pd.concat([df_proc, df], axis=1)
    n_dup = df_proc.duplicated().sum()
    df_proc = df_proc.drop_duplicates().reset_index(drop=True)
    
    # Handle missing
    num_cols = [c for c in selected_features if c in NUMERICAL_COLS]
    cat_cols = [c for c in selected_features if c in CATEGORICAL_COLS]
    
    for col in num_cols:
        if col in df_proc.columns and df_proc[col].isnull().sum() > 0:
            med = df_proc[col].median()
            df_proc[col] = df_proc[col].fillna(med)
    
    for col in cat_cols:
        if col in df_proc.columns and df_proc[col].isnull().sum() > 0:
            mode_val = df_proc[col].mode()
            if len(mode_val) > 0:
                df_proc[col] = df_proc[col].fillna(mode_val[0])
    
    # Detect data format
    data_format = detect_data_format(df_proc)
    is_raw_data = (data_format == 'raw')
    
    # Encoding (only if raw data)
    df_enc = df_proc.copy()
    if is_raw_data:
        if 'NIB/SKU' in selected_features:
            df_enc['NIB/SKU'] = df_enc['NIB/SKU'].apply(encode_nib)
        if 'Kepemilikan Lahan' in selected_features:
            df_enc['Kepemilikan Lahan'] = df_enc['Kepemilikan Lahan'].apply(encode_lahan)
        if 'Sosmed' in selected_features:
            df_enc['Sosmed'] = df_enc['Sosmed'].apply(count_apps)
        if 'Marketplace' in selected_features:
            df_enc['Marketplace'] = df_enc['Marketplace'].apply(count_apps)
    else:
        # Pre-processed data: ensure numeric conversion
        for col in ['NIB/SKU', 'Kepemilikan Lahan', 'Sosmed', 'Marketplace']:
            if col in df_enc.columns:
                df_enc[col] = pd.to_numeric(df_enc[col], errors='coerce').fillna(0).astype(int)
    
    # Outlier detection
    outlier_info = []
    df_clean = df_enc.copy()
    
    out_cols = [c for c in OUTLIER_COLS if c in selected_features]
    for col in out_cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_out = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
            df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
            
            outlier_info.append({
                'Kolom': col,
                'Q1': float(Q1),
                'Q3': float(Q3),
                'IQR': float(IQR),
                'Lower Bound': float(lower),
                'Upper Bound': float(upper),
                'Outliers': int(n_out)
            })
    
    # Data type validation & normalization
    for col in selected_features:
        if col in df_clean.columns:
            # Convert to numeric if needed
            if df_clean[col].dtype == 'object':
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            # Handle any remaining NaN from conversion errors
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median() if col in NUMERICAL_COLS else 0)
            # Ensure float64 for numerical stability
            df_clean[col] = df_clean[col].astype('float64')
    
    # Scaling
    X_raw = df_clean[selected_features].values.astype(float)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    return {
        'df_proc': df_proc,
        'df_enc': df_enc,
        'df_clean': df_clean,
        'X_raw': X_raw,
        'X_scaled': X_scaled,
        'scaler': scaler,
        'outlier_info': outlier_info,
        'n_dup': n_dup,
        'selected_features': selected_features
    }

# ============================================================================
# CLUSTERING FUNCTIONS
# ============================================================================

def compute_gower_matrix(X, feat_names, feat_types_dict, weights_dict=None):
    n, p = X.shape
    ftypes = [feat_types_dict.get(f, 'num') for f in feat_names]
    
    if weights_dict is None:
        weights_dict = {f: 1.0/p for f in feat_names}
    
    w_arr = np.array([weights_dict.get(f, 1.0/p) for f in feat_names])
    dist_matrix = np.zeros((n, n), dtype=float)
    
    for j, (ftype, w) in enumerate(zip(ftypes, w_arr)):
        col = X[:, j]
        if ftype == 'num':
            rng = col.max() - col.min()
            rng = rng if rng > 0 else 1.0
            diff = np.abs(col[:, None] - col[None, :]) / rng
        else:
            diff = (col[:, None] != col[None, :]).astype(float)
        dist_matrix += w * diff
    
    np.fill_diagonal(dist_matrix, 0.0)
    return dist_matrix

def clara_fit(dist_matrix, n_clusters, n_sampling_iter=120, random_state=42):
    rng = np.random.RandomState(random_state)
    n = dist_matrix.shape[0]
    sample_size = max(int(0.25 * n), 50 + 5 * n_clusters)
    sample_size = min(sample_size, n)
    
    best_cost = np.inf
    best_labels = None
    best_medoids = None
    
    for itr in range(n_sampling_iter):
        sample_idx = rng.choice(n, size=sample_size, replace=False)
        sample_dist = dist_matrix[np.ix_(sample_idx, sample_idx)]
        
        kmed = KMedoids(
            n_clusters=n_clusters,
            metric='precomputed',
            method='pam',
            init='build',
            random_state=random_state + itr
        )
        kmed.fit(sample_dist)
        
        medoid_indices = sample_idx[kmed.medoid_indices_]
        labels = np.argmin(dist_matrix[:, medoid_indices], axis=1)
        cost = np.sum([dist_matrix[i, medoid_indices[labels[i]]] for i in range(n)])
        
        if cost < best_cost:
            best_cost = cost
            best_labels = labels.copy()
            best_medoids = medoid_indices.copy()
    
    return best_labels, best_medoids

def davies_bouldin_gower(distance_matrix, labels, medoids):
    labels = np.array(labels)
    k = len(np.unique(labels))
    S = np.zeros(k)
    
    for i in range(k):
        cluster_idx = np.where(labels == i)[0]
        medoid_idx = medoids[i]
        if len(cluster_idx) > 0:
            S[i] = np.mean(distance_matrix[cluster_idx][:, medoid_idx])
        else:
            S[i] = 0
    
    dbi = 0
    for i in range(k):
        max_ratio = 0
        for j in range(k):
            if i != j:
                Mij = distance_matrix[medoids[i], medoids[j]]
                if Mij > 0:
                    ratio = (S[i] + S[j]) / Mij
                    max_ratio = max(max_ratio, ratio)
        dbi += max_ratio
    
    return dbi / k if k > 0 else np.nan

def safe_calinski_harabasz(X_features, labels):
    """Calculate CHI safely from a feature matrix (not a distance matrix)."""
    labels = np.asarray(labels)
    n_samples = len(labels)
    n_clusters = len(np.unique(labels))
    if n_clusters <= 1 or n_clusters >= n_samples:
        return np.nan
    try:
        return calinski_harabasz_score(X_features, labels)
    except Exception:
        return np.nan

def entropy_weight_method(X_norm, feat_names):
    n, p = X_norm.shape
    k_const = 1.0 / np.log(n)
    results = []
    
    for j in range(p):
        col = X_norm[:, j]
        col_sum = col.sum()
        pij = col / col_sum if col_sum > 0 else np.zeros(n)
        log_pij = np.where(pij > 0, np.log(pij), 0.0)
        ej = float(np.clip(-k_const * np.sum(pij * log_pij), 0.0, 1.0))
        dj = 1.0 - ej
        results.append({'feat': feat_names[j], 'entropy': ej, 'diversification': dj})
    
    sum_d = sum([r['diversification'] for r in results])
    for r in results:
        r['weight'] = r['diversification'] / sum_d if sum_d > 0 else 1.0 / p
    
    return results

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

# Cluster characteristics definition for Top-4 Features scenario (k=4)
CLUSTER_CHARACTERISTICS = {
    1: {
        'name': 'UMKM Skala Kecil dengan Kinerja Finansial Rendah',
        'avg_tenaga_kerja': 6.98,
        'avg_kapasitas': 1203,
        'avg_omset': 109621395,
        'avg_aset': 99599485,
        'icon': '📉',
        'color': '#FF6B6B'
    },
    2: {
        'name': 'UMKM Berkembang dengan Kapasitas Operasional Tinggi',
        'avg_tenaga_kerja': 29.51,
        'avg_kapasitas': 7486,
        'avg_omset': 427525000,
        'avg_aset': 337107931,
        'icon': '📈',
        'color': '#4ECDC4'
    },
    3: {
        'name': 'UMKM Finansial Kuat dengan Kapasitas Operasional Terbatas',
        'avg_tenaga_kerja': 7.02,
        'avg_kapasitas': 1041,
        'avg_omset': 791286004,
        'avg_aset': 765337591,
        'icon': '💰',
        'color': '#FFD93D'
    },
    4: {
        'name': 'UMKM Produktif dan Efisien',
        'avg_tenaga_kerja': 4.00,
        'avg_kapasitas': 10268,
        'avg_omset': 198036458,
        'avg_aset': 205406770,
        'icon': '⚙️',
        'color': '#95E1D3'
    }
}

def format_currency(value):
    """Format currency in Indonesian Rupiah"""
    if value >= 1_000_000_000:
        return f"Rp{value/1_000_000_000:.2f}M"
    elif value >= 1_000_000:
        return f"Rp{value/1_000_000:.2f}J"
    else:
        return f"Rp{value:.0f}"

def predict_cluster(input_data, selected_features, scaler, medoids, top_4_features, 
                   gower_dist_func, feat_types_dict, weights_dict=None):
    """
    Predict cluster for new UMKM data
    
    Args:
        input_data: dict dengan keys sesuai selected_features
        selected_features: list fitur yang digunakan
        scaler: fitted MinMaxScaler object
        medoids: array medoid indices dari CLARA Top-4
        top_4_features: list of top 4 feature names
        gower_dist_func: function to compute gower distance
        feat_types_dict: dict tipe fitur
        weights_dict: dict bobot fitur (dari EWM)
    
    Returns:
        dict with cluster prediction and details
    """
    try:
        # Prepare input as array matching selected_features order
        input_array = np.array([[input_data.get(feat, 0) for feat in selected_features]], dtype=float)
        
        # Scale using the same scaler
        input_scaled = scaler.transform(input_array)
        
        # Extract only top-4 features for distance calculation
        top4_indices = [selected_features.index(f) for f in top_4_features if f in selected_features]
        input_top4 = input_scaled[:, top4_indices]
        
        # Compute distance to each medoid using top-4 features only
        distances = []
        for medoid_idx in medoids:
            # We need to recalculate distance for top-4 features
            # For simplicity, use euclidean distance on scaled data
            medoid_top4 = input_scaled[0, top4_indices]  # Actually we need medoid data
            # This is a limitation - we compute on input data
            dist = np.linalg.norm(input_top4[0] - input_scaled[0, top4_indices])
            distances.append(dist)
        
        # Find nearest medoid cluster
        predicted_cluster = np.argmin(distances) + 1
        min_distance = np.min(distances)
        
        return {
            'cluster': predicted_cluster,
            'distance': min_distance,
            'distances_all': distances,
            'success': True
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'cluster': None
        }

def predict_cluster_improved(input_data, selected_features, X_scaled_training, scaler, 
                            medoids_indices, top_4_features, feat_types_dict, weights_dict=None):
    """
    Improved predict cluster for new UMKM data using medoid indices from training data
    
    Args:
        input_data: dict dengan keys sesuai selected_features
        selected_features: list fitur yang digunakan
        X_scaled_training: scaled training data matrix
        scaler: fitted MinMaxScaler object
        medoids_indices: array of actual medoid indices in X_scaled_training
        top_4_features: list of top 4 feature names (for feature selection)
        feat_types_dict: dict tipe fitur
        weights_dict: dict bobot fitur (dari EWM)
    
    Returns:
        dict with cluster prediction and details
    """
    try:
        # Prepare input as array matching selected_features order
        input_array = np.array([[input_data.get(feat, 0) for feat in selected_features]], dtype=float)
        
        # Scale using the same scaler
        input_scaled = scaler.transform(input_array)[0]
        
        # Get indices of top-4 features in selected_features
        top4_indices = [i for i, f in enumerate(selected_features) if f in top_4_features]
        
        # Extract top-4 features from both input and medoids
        input_top4 = input_scaled[top4_indices]
        
        # Compute weighted distance to each medoid
        distances = []
        for medoid_idx in medoids_indices:
            medoid_data = X_scaled_training[medoid_idx]
            medoid_top4 = medoid_data[top4_indices]
            
            # Compute weighted Euclidean distance using top-4 features
            if weights_dict:
                weighted_dist = 0
                for i, feat_idx in enumerate(top4_indices):
                    feat_name = selected_features[feat_idx]
                    weight = weights_dict.get(feat_name, 1.0 / len(selected_features))
                    dist_component = weight * (input_top4[i] - medoid_top4[i]) ** 2
                    weighted_dist += dist_component
                dist = np.sqrt(weighted_dist)
            else:
                # Unweighted Euclidean distance
                dist = np.linalg.norm(input_top4 - medoid_top4)
            
            distances.append(dist)
        
        # Find nearest medoid cluster (0-indexed medoid will map to cluster 0, 1, 2, 3)
        predicted_cluster = np.argmin(distances) + 1  # +1 to make it 1-indexed
        min_distance = np.min(distances)
        
        return {
            'cluster': predicted_cluster,
            'distance': min_distance,
            'distances_all': distances,
            'success': True
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'cluster': None
        }

# ============================================================================
# MAIN APP
# ============================================================================

def show_prediction_page():
    """Display prediction page for new UMKM"""
    st.markdown("""
        <h1 style='text-align: center; color: #667eea;'>
            🔮 PREDIKSI CLUSTER UMKM BARU
        </h1>
        <p style='text-align: center; color: #666;'>
            Tentukan cluster UMKM baru berdasarkan model terbaik CLARA Top-4 Features (k=4)
        </p>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Check if model is available
    if 'clustering_model' not in st.session_state or not st.session_state.clustering_model.get('data_loaded', False):
        st.warning("⚠️ Model clustering belum tersedia")
        st.info("💡 Ikuti langkah berikut:\n1. Pilih menu '📊 Analisis Clustering' di sidebar\n2. Upload dataset dan jalankan analisis\n3. Kembali ke menu ini untuk memprediksi cluster UMKM baru")
        return
    
    model = st.session_state.clustering_model
    
    # Input form
    st.markdown("### 📝 Form Input UMKM Baru")
    st.markdown("Masukkan data UMKM baru untuk diprediksi clusternya")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            nama_pemilik = st.text_input(
                "👤 Nama Pemilik",
                placeholder="Masukkan nama pemilik UMKM"
            )
        
        with col2:
            nama_usaha = st.text_input(
                "🏢 Nama Usaha",
                placeholder="Masukkan nama usaha/bisnis"
            )
        
        st.markdown("---")
        st.markdown("#### 📊 Data Fitur (Top-4 Features)")
        
        # Get feature values from top-4 features
        top_4_features = model['top_4_features']
        feature_values = {}
        
        col1, col2 = st.columns(2)
        
        for idx, feat in enumerate(top_4_features):
            if idx < 2:
                with col1:
                    if feat == 'Jml. Tenaga Kerja':
                        feature_values[feat] = st.number_input(
                            f"👥 {feat}",
                            min_value=0,
                            value=5,
                            step=1,
                            help="Jumlah pekerja di UMKM"
                        )
                    elif feat == 'Kapasitas Produksi/Thn':
                        feature_values[feat] = st.number_input(
                            f"🏭 {feat}",
                            min_value=0,
                            value=1000,
                            step=100,
                            help="Kapasitas produksi per tahun (unit)"
                        )
                    elif feat == 'Omset/Thn':
                        feature_values[feat] = st.number_input(
                            f"💵 {feat}",
                            min_value=0,
                            value=100000000,
                            step=10000000,
                            help="Omset per tahun (Rp)"
                        )
                    elif feat == 'Aset':
                        feature_values[feat] = st.number_input(
                            f"🏦 {feat}",
                            min_value=0,
                            value=100000000,
                            step=10000000,
                            help="Total aset (Rp)"
                        )
            else:
                with col2:
                    if feat == 'Jml. Tenaga Kerja':
                        feature_values[feat] = st.number_input(
                            f"👥 {feat}",
                            min_value=0,
                            value=5,
                            step=1,
                            help="Jumlah pekerja di UMKM"
                        )
                    elif feat == 'Kapasitas Produksi/Thn':
                        feature_values[feat] = st.number_input(
                            f"🏭 {feat}",
                            min_value=0,
                            value=1000,
                            step=100,
                            help="Kapasitas produksi per tahun (unit)"
                        )
                    elif feat == 'Omset/Thn':
                        feature_values[feat] = st.number_input(
                            f"💵 {feat}",
                            min_value=0,
                            value=100000000,
                            step=10000000,
                            help="Omset per tahun (Rp)"
                        )
                    elif feat == 'Aset':
                        feature_values[feat] = st.number_input(
                            f"🏦 {feat}",
                            min_value=0,
                            value=100000000,
                            step=10000000,
                            help="Total aset (Rp)"
                        )
        
        st.markdown("---")
        
        # Prediction button
        submit_btn = st.form_submit_button(
            "🔍 Prediksi Cluster",
            use_container_width=True
        )
    
    # Process prediction
    if submit_btn:
        if not nama_pemilik.strip() or not nama_usaha.strip():
            st.error("❌ Nama pemilik dan nama usaha harus diisi!")
            return
        
        if not all(feature_values):
            st.error("❌ Semua fitur harus diisi!")
            return
        
        # Prepare input data with all features (fill non-top-4 with 0 or mean)
        input_dict = feature_values.copy()
        for feat in model['selected_features']:
            if feat not in input_dict:
                input_dict[feat] = 0  # Default value for non-top-4 features
        
        # Perform prediction
        result = predict_cluster_improved(
            input_data=input_dict,
            selected_features=model['selected_features'],
            X_scaled_training=model['X_scaled'],
            scaler=model['scaler'],
            medoids_indices=model['medoids'],
            top_4_features=model['top_4_features'],
            feat_types_dict=model['feat_types'],
            weights_dict=model['weights']
        )
        
        if result['success']:
            predicted_cluster = result['cluster']
            
            # Get cluster characteristics
            cluster_info = CLUSTER_CHARACTERISTICS[predicted_cluster]
            
            st.markdown("---")
            st.markdown("### ✅ HASIL PREDIKSI")
            
            # Display prediction result
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**👤 Nama Pemilik:** {nama_pemilik}")
                st.markdown(f"**🏢 Nama Usaha:** {nama_usaha}")
            
            with col2:
                st.markdown(f"**🎯 Cluster Prediksi:** Cluster {predicted_cluster}")
                st.markdown(f"**Jarak ke Medoid:** {result['distance']:.6f}")
            
            st.markdown("---")
            
            # Display cluster characteristics with colored card
            st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; background-color: {cluster_info["color"]}; 
                        border: 2px solid #333; margin-bottom: 20px;'>
                <h3 style='margin-top: 0; color: white;'>{cluster_info["icon"]} {cluster_info["name"]}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Cluster characteristics table
            st.markdown("#### 📊 Karakteristik Cluster")
            
            char_data = {
                'Metrik': [
                    '👥 Rata-rata Tenaga Kerja',
                    '🏭 Kapasitas Produksi/Thn',
                    '💵 Omset/Thn',
                    '🏦 Aset'
                ],
                'Nilai': [
                    f"{cluster_info['avg_tenaga_kerja']:.2f} orang",
                    f"{cluster_info['avg_kapasitas']:,}",
                    format_currency(cluster_info['avg_omset']),
                    format_currency(cluster_info['avg_aset'])
                ]
            }
            
            df_char = pd.DataFrame(char_data)
            st.dataframe(df_char, use_container_width=True, hide_index=True)
            
            # Comparison with input
            st.markdown("---")
            st.markdown("#### 📈 Perbandingan Data Input vs Rata-rata Cluster")
            
            comparison_data = {
                'Fitur': ['👥 Jml. Tenaga Kerja', '🏭 Kapasitas Produksi/Thn', '💵 Omset/Thn', '🏦 Aset'],
                'Input UMKM Baru': [
                    f"{feature_values.get('Jml. Tenaga Kerja', 0):.0f}",
                    f"{feature_values.get('Kapasitas Produksi/Thn', 0):,}",
                    format_currency(feature_values.get('Omset/Thn', 0)),
                    format_currency(feature_values.get('Aset', 0))
                ],
                'Rata-rata Cluster': [
                    f"{cluster_info['avg_tenaga_kerja']:.2f}",
                    f"{cluster_info['avg_kapasitas']:,}",
                    format_currency(cluster_info['avg_omset']),
                    format_currency(cluster_info['avg_aset'])
                ]
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True, hide_index=True)
            
            # Summary
            st.markdown("---")
            st.markdown("#### 📝 Ringkasan")
            
            summary_text = f"""
            **UMKM yang Anda input masuk ke Cluster {predicted_cluster}** dengan karakteristik:
            
            **{cluster_info["icon"]} {cluster_info["name"]}**
            
            - **Rata-rata Tenaga Kerja:** {cluster_info['avg_tenaga_kerja']:.2f} orang
            - **Rata-rata Kapasitas Produksi:** {cluster_info['avg_kapasitas']:,} unit/tahun  
            - **Rata-rata Omset:** {format_currency(cluster_info['avg_omset'])}/tahun
            - **Rata-rata Aset:** {format_currency(cluster_info['avg_aset'])}
            
            **Interpretasi:** UMKM Anda memiliki profil yang sama dengan mayoritas UMKM di cluster ini dalam hal 
            skala operasional, kapasitas produksi, dan kondisi finansial.
            """
            
            st.info(summary_text)
            
            # Distance explanation
            st.markdown("---")
            st.markdown("#### 🔍 Penjelasan Teknis")
            
            distances_info = f"""
            Prediksi dilakukan menggunakan model CLARA Top-4 Features dengan k=4 cluster:
            
            - **Model:** CLARA (Clustering Large Applications)
            - **Fitur yang Digunakan:** {', '.join(model['top_4_features'])}
            - **Jumlah Cluster:** 4
            - **Bobot Fitur (EWM):** {dict((f, f"{w:.4f}") for f, w in model['weights'].items())}
            
            **Jarak ke Setiap Medoid:**
            """
            
            st.markdown(distances_info)
            
            dist_data = {
                'Cluster': [f'Cluster {i+1}' for i in range(len(result['distances_all']))],
                'Jarak ke Medoid': [f"{d:.6f}" for d in result['distances_all']]
            }
            
            df_distances = pd.DataFrame(dist_data)
            st.dataframe(df_distances, use_container_width=True, hide_index=True)
            
            st.success(f"✅ UMKM diprediksi masuk ke **Cluster {predicted_cluster}** karena memiliki jarak terdekat ke medoid cluster tersebut!")
        
        else:
            st.error(f"❌ Error dalam prediksi: {result.get('error', 'Unknown error')}")


# ============================================================================
# TOPSIS FUNCTIONS AND PAGE
# ============================================================================

def calculate_topsis_by_cluster(df_values, labels, feature_cols, weights):
    """Calculate TOPSIS independently inside every cluster.

    The implementation follows entropy_clara_topsis.ipynb: vector
    normalization, entropy-weighted normalized values, benefit-only ideal
    solutions, Euclidean separation, and within-cluster ranking.
    """
    if len(df_values) != len(labels):
        raise ValueError("Jumlah data dan label cluster tidak sama.")

    missing = [col for col in feature_cols if col not in df_values.columns]
    if missing:
        raise ValueError(f"Fitur TOPSIS tidak ditemukan: {', '.join(missing)}")

    total_weight = sum(float(weights.get(col, 0.0)) for col in feature_cols)
    if total_weight <= 0:
        normalized_weights = {col: 1.0 / len(feature_cols) for col in feature_cols}
    else:
        normalized_weights = {
            col: float(weights.get(col, 0.0)) / total_weight for col in feature_cols
        }

    work = df_values.copy().reset_index(drop=True)
    work["Cluster"] = np.asarray(labels, dtype=int) + 1
    results = {}
    ideals = {}

    for cluster_id in sorted(work["Cluster"].unique()):
        cluster_df = work.loc[work["Cluster"] == cluster_id].copy()
        matrix = cluster_df[feature_cols].to_numpy(dtype=float)

        denominator = np.sqrt(np.sum(np.square(matrix), axis=0))
        denominator = np.where(denominator == 0, 1.0, denominator)
        normalized = matrix / denominator

        weight_array = np.array(
            [normalized_weights[col] for col in feature_cols], dtype=float
        )
        weighted = normalized * weight_array
        ideal_positive = np.max(weighted, axis=0)
        ideal_negative = np.min(weighted, axis=0)

        d_positive = np.sqrt(np.sum((weighted - ideal_positive) ** 2, axis=1))
        d_negative = np.sqrt(np.sum((weighted - ideal_negative) ** 2, axis=1))
        distance_total = d_positive + d_negative
        score = np.divide(
            d_negative,
            distance_total,
            out=np.zeros_like(d_negative, dtype=float),
            where=distance_total != 0,
        )
        score = np.clip(score, 0.0, 1.0)

        cluster_df["D_positive"] = d_positive
        cluster_df["D_negative"] = d_negative
        cluster_df["skor_topsis"] = score
        cluster_df = cluster_df.sort_values(
            ["skor_topsis", "Identitas_UMKM"],
            ascending=[False, True],
        ).reset_index(drop=True)
        cluster_df["ranking_topsis"] = np.arange(1, len(cluster_df) + 1)

        results[int(cluster_id)] = cluster_df
        ideals[int(cluster_id)] = pd.DataFrame({
            "Feature": feature_cols,
            "Weight": [normalized_weights[col] for col in feature_cols],
            "Ideal Positive (A+)": ideal_positive,
            "Ideal Negative (A-)": ideal_negative,
        })

    return results, ideals, normalized_weights


def build_topsis_excel(topsis_results, ideals, weights):
    """Build an in-memory Excel workbook containing all TOPSIS outputs."""
    output = io.BytesIO()
    summary_rows = []

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame({
            "Feature": list(weights.keys()),
            "Weight": list(weights.values()),
        }).to_excel(writer, sheet_name="Bobot_TOPSIS", index=False)

        for cluster_id, result in topsis_results.items():
            result.to_excel(writer, sheet_name=f"TOPSIS_C{cluster_id}", index=False)
            ideals[cluster_id].to_excel(
                writer, sheet_name=f"Ideal_C{cluster_id}", index=False
            )
            summary_rows.append({
                "Cluster": cluster_id,
                "Jumlah_UMKM": len(result),
                "Skor_Tertinggi": result["skor_topsis"].max(),
                "Skor_Terendah": result["skor_topsis"].min(),
                "Skor_Rata_Rata": result["skor_topsis"].mean(),
            })

        pd.DataFrame(summary_rows).to_excel(
            writer, sheet_name="Ringkasan_TOPSIS", index=False
        )

    output.seek(0)
    return output.getvalue()


def show_topsis_page():
    """Display within-cluster TOPSIS results for the selected Top-4 model."""
    st.markdown("""
        <h1 style='text-align: center; color: #667eea;'>
            🏆 RANKING TOPSIS PER CLUSTER
        </h1>
        <p style='text-align: center; color: #666;'>
            Prioritas kesiapan ekonomi-operasional UMKM dalam kelompok yang sebanding
        </p>
        """, unsafe_allow_html=True)
    st.markdown("---")

    model = st.session_state.get("clustering_model")
    if not model or not model.get("data_loaded", False):
        st.warning("⚠️ Hasil clustering belum tersedia.")
        st.info(
            "Jalankan analisis pada menu **📊 Analisis Clustering** terlebih "
            "dahulu. Setelah selesai, kembali ke menu ini untuk melihat ranking TOPSIS."
        )
        return

    required_keys = {"X_scaled", "selected_features", "labels", "top_4_features", "weights", "df_clean"}
    missing_keys = sorted(required_keys.difference(model.keys()))
    if missing_keys:
        st.error(
            "Model tersimpan belum memiliki data TOPSIS lengkap: "
            + ", ".join(missing_keys)
        )
        return

    top_features = list(model["top_4_features"])
    if len(top_features) != 4:
        st.error("TOPSIS memerlukan hasil skenario Top-4. Jalankan analisis dengan minimal 4 fitur.")
        return

    selected_features = list(model["selected_features"])
    feature_indices = [selected_features.index(f) for f in top_features]
    df_topsis_input = pd.DataFrame(
        model["X_scaled"][:, feature_indices], columns=top_features
    )
    df_topsis_input.insert(
        0, "Identitas_UMKM", model["df_clean"]["Identitas_UMKM"].astype(str).values
    )

    try:
        topsis_results, ideals, topsis_weights = calculate_topsis_by_cluster(
            df_topsis_input,
            model["labels"],
            top_features,
            model["weights"],
        )
    except Exception as exc:
        st.error(f"TOPSIS tidak dapat dihitung: {exc}")
        return

    st.warning(
        "⚠️ **Batas interpretasi:** skor dan ranking TOPSIS hanya boleh "
        "dibandingkan di dalam cluster yang sama, bukan antar-cluster."
    )

    st.markdown("### ⚖️ Kriteria dan Bobot TOPSIS")
    weight_df = pd.DataFrame({
        "Feature": top_features,
        "Bobot TOPSIS": [topsis_weights[f] for f in top_features],
        "Jenis Kriteria": ["Benefit"] * len(top_features),
    })
    st.dataframe(
        weight_df.style.format({"Bobot TOPSIS": "{:.6f}"}),
        use_container_width=True,
        hide_index=True,
    )

    cluster_id = st.selectbox(
        "Pilih cluster yang ingin ditampilkan:",
        options=sorted(topsis_results.keys()),
        format_func=lambda value: f"Cluster {value}",
    )
    result = topsis_results[cluster_id]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cluster", cluster_id)
    col2.metric("Jumlah UMKM", len(result))
    col3.metric("Skor tertinggi", f"{result['skor_topsis'].max():.4f}")
    col4.metric("Skor rata-rata", f"{result['skor_topsis'].mean():.4f}")

    st.markdown("### 🎯 Solusi Ideal")
    st.dataframe(
        ideals[cluster_id].style.format({
            "Weight": "{:.6f}",
            "Ideal Positive (A+)": "{:.6f}",
            "Ideal Negative (A-)": "{:.6f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 📋 Ranking TOPSIS")
    display_option = st.selectbox(
        "Jumlah data yang ditampilkan:",
        [5, 10, 20, "Semua"],
        index=1,
    )
    shown = result if display_option == "Semua" else result.head(int(display_option))
    display_cols = [
        "ranking_topsis", "Identitas_UMKM", *top_features,
        "D_positive", "D_negative", "skor_topsis", "Cluster",
    ]
    st.dataframe(
        shown[display_cols].style.format({
            **{f: "{:.6f}" for f in top_features},
            "D_positive": "{:.6f}",
            "D_negative": "{:.6f}",
            "skor_topsis": "{:.6f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 📊 Skor UMKM Peringkat Teratas")
    chart_count = min(10, len(result))
    chart_data = result.head(chart_count).set_index("Identitas_UMKM")[["skor_topsis"]]
    st.bar_chart(chart_data, use_container_width=True)

    best = result.iloc[0]
    worst = result.iloc[-1]
    col_best, col_worst = st.columns(2)
    with col_best:
        st.success(
            f"🥇 **Ranking tertinggi Cluster {cluster_id}**\n\n"
            f"{best['Identitas_UMKM']} — skor {best['skor_topsis']:.6f}"
        )
    with col_worst:
        st.info(
            f"📍 **Ranking terendah Cluster {cluster_id}**\n\n"
            f"{worst['Identitas_UMKM']} — skor {worst['skor_topsis']:.6f}"
        )

    st.markdown("### 💾 Unduh Hasil TOPSIS")
    col_download_1, col_download_2 = st.columns(2)
    with col_download_1:
        csv_data = result.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            f"📥 Download TOPSIS Cluster {cluster_id} (CSV)",
            data=csv_data,
            file_name=f"topsis_cluster_{cluster_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_download_2:
        excel_data = build_topsis_excel(topsis_results, ideals, topsis_weights)
        st.download_button(
            "📥 Download Seluruh TOPSIS (Excel)",
            data=excel_data,
            file_name="hasil_topsis_semua_cluster.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

def main():
    st.markdown("""
        <h1 style='text-align: center; color: #667eea;'>
            📊 CLUSTERING UMKM Kec. Sampang
        </h1>
        <p style='text-align: center; color: #666;'>
            Segmentasi UMKM di Kecamatan Sampang Menggunakan Metode CLARA dengan Seleksi Fitur Berbasis Entropy
        </p>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ====================================================================
    # SIDEBAR - NAVIGATION MENU
    # ====================================================================
    
    with st.sidebar:
        st.markdown("### 🗂️ MENU UTAMA")
        st.markdown("---")
        
        menu_options = [
            "📊 Analisis Clustering",
            "🏆 Ranking TOPSIS",
            "🔮 Prediksi Cluster UMKM Baru"
        ]
        
        selected_menu = st.radio(
            "Pilih Menu:",
            menu_options,
            key='menu_navigation'
        )
        
        st.session_state.page = selected_menu
        st.markdown("---")
    
    # Render selected page
    if st.session_state.page == "🔮 Prediksi Cluster UMKM Baru":
        show_prediction_page()
        return
    if st.session_state.page == "🏆 Ranking TOPSIS":
        show_topsis_page()
        return
    
    # ====================================================================
    # SIDEBAR - DATA UPLOAD & FEATURE SELECTION (FOR ANALYSIS PAGE)
    # ====================================================================
    
    with st.sidebar:
        st.markdown("### ⚙️ KONTROL ANALISIS")
        st.markdown("---")
        
        st.markdown("**📤 Upload Dataset**")
        uploaded_file = st.file_uploader(
            "Pilih file Excel/CSV",
            type=['xlsx', 'csv'],
            help="File harus memiliki kolom: Nama Pemilik, Nama Usaha, 8 fitur"
        )
        
        if uploaded_file is None:
            st.warning("⚠️ Silakan upload file untuk memulai")
            return
        
        # Load file
        try:
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
            st.success(f"✅ File dimuat: {uploaded_file.name}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return
        
        st.markdown("---")
        
        # Feature selection
        st.markdown("**🏷️ Pilih Fitur**")
        selected_features = st.multiselect(
            "Fitur untuk clustering:",
            FEATURE_COLS,
            default=FEATURE_COLS,
            help="Minimal 3 fitur"
        )
        
        if len(selected_features) < 3:
            st.error("❌ Minimal pilih 3 fitur!")
            return
        
        # Update feature types dict
        feat_types = {f: FEATURE_TYPES[f] for f in selected_features}
        
        st.markdown("---")
        
        # Parameter k
        st.markdown("**🔢 Parameter Clustering**")
        k_min = st.slider("k minimum", 2, 8, 2, key='k_min')
        k_max = st.slider("k maksimum", k_min, 15, 8, key='k_max')
        
        if k_min >= k_max:
            st.error("❌ k minimum harus < k maksimum")
            return
        
        st.markdown("---")
        run_btn = st.button("🚀 Jalankan Analisis Lengkap", use_container_width=True)
    
    if not run_btn:
        st.info("👈 Sesuaikan parameter dan klik 🚀 untuk mulai")
        return
    
    # ====================================================================
    # PROCESSING
    # ====================================================================
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Preprocessing
        status_text.text("📋 1/10 Preprocessing data...")
        progress_bar.progress(10)
        
        prep_result = preprocess_data(df_raw, selected_features)
        df_clean = prep_result['df_clean']
        X_scaled = prep_result['X_scaled']
        X_raw = prep_result['X_raw']
        
        n_data = len(df_clean)
        
        # CLARA Baseline
        status_text.text("📊 2/10 CLARA Baseline (k=2-8)...")
        progress_bar.progress(20)
        
        results_baseline = {}
        for k in range(k_min, k_max + 1):
            gower_dist = compute_gower_matrix(X_scaled, selected_features, feat_types)
            labels, medoids = clara_fit(gower_dist, n_clusters=k)
            
            if len(np.unique(labels)) > 1:
                sil = silhouette_score(gower_dist, labels, metric='precomputed')
                dbi = davies_bouldin_gower(gower_dist, labels, medoids)
                chi = safe_calinski_harabasz(X_scaled, labels)
            else:
                sil, dbi, chi = -1.0, np.nan, np.nan
            
            results_baseline[k] = {
                'labels': labels, 'medoids': medoids,
                'silhouette': sil, 'dbi': dbi, 'chi': chi, 'gower': gower_dist
            }
        
        # Select k_best
        df_eval = pd.DataFrame([
            {'k': k, 'silhouette': results_baseline[k]['silhouette'],
             'dbi': results_baseline[k]['dbi'], 'chi': results_baseline[k]['chi']}
            for k in range(k_min, k_max + 1)
        ])
        
        sil_min, sil_max = df_eval['silhouette'].min(), df_eval['silhouette'].max()
        dbi_min, dbi_max = df_eval['dbi'].min(), df_eval['dbi'].max()
        
        df_eval['sil_norm'] = (df_eval['silhouette'] - sil_min) / (sil_max - sil_min + 1e-9)
        df_eval['dbi_norm'] = (dbi_max - df_eval['dbi']) / (dbi_max - dbi_min + 1e-9)
        df_eval['score'] = 0.5 * df_eval['sil_norm'] + 0.5 * df_eval['dbi_norm']
        
        # Get k_best from the row with highest score (not index!)
        best_idx = df_eval['score'].idxmax()
        k_best = int(df_eval.loc[best_idx, 'k'])
        
        # EWM
        status_text.text("⚖️ 3/10 Entropy Weight Method...")
        progress_bar.progress(30)
        
        ewm_results = entropy_weight_method(X_scaled, selected_features)
        df_ewm = pd.DataFrame(ewm_results).sort_values('weight', ascending=False).reset_index(drop=True)
        weights = {r['feat']: r['weight'] for r in ewm_results}
        
        # CLARA Weighted
        status_text.text("🎯 4/10 CLARA Weighted...")
        progress_bar.progress(40)
        
        results_weighted = {}
        for k in range(k_min, k_max + 1):
            gower_weighted = compute_gower_matrix(X_scaled, selected_features, feat_types, weights)
            labels, medoids = clara_fit(gower_weighted, n_clusters=k)
            
            if len(np.unique(labels)) > 1:
                sil = silhouette_score(gower_weighted, labels, metric='precomputed')
                dbi = davies_bouldin_gower(gower_weighted, labels, medoids)
                chi = safe_calinski_harabasz(X_scaled, labels)
            else:
                sil, dbi, chi = -1.0, np.nan, np.nan
            
            results_weighted[k] = {
                'labels': labels, 'medoids': medoids,
                'silhouette': sil, 'dbi': dbi, 'chi': chi, 'gower': gower_weighted
            }
        
        # Use k_best for final weighted clustering
        labels_weighted = results_weighted[k_best]['labels']
        medoids_weighted = results_weighted[k_best]['medoids']
        sil_weighted = results_weighted[k_best]['silhouette']
        dbi_weighted = results_weighted[k_best]['dbi']
        chi_weighted = results_weighted[k_best]['chi']
        gower_weighted = results_weighted[k_best]['gower']
        
        # Create evaluation dataframe for weighted results
        df_eval_weighted = pd.DataFrame([
            {'k': k, 'silhouette': results_weighted[k]['silhouette'],
             'dbi': results_weighted[k]['dbi'], 'chi': results_weighted[k]['chi']}
            for k in range(k_min, k_max + 1)
        ])
        
        # CLARA Top-N (Top-5, Top-6, Top-7, Top-8)
        status_text.text("🎯 5/10 CLARA Top-N...")
        progress_bar.progress(50)
        
        top_n_results = {}
        for top_n in range(MIN_TOP_N, len(selected_features) + 1):
            if top_n > len(selected_features):
                continue
            
            top_features = df_ewm.head(top_n)['feat'].tolist()
            top_weights = {f: weights[f] for f in top_features}
            
            # Renormalize weights
            sum_w = sum(top_weights.values())
            top_weights = {f: w/sum_w for f, w in top_weights.items()}
            
            X_top = X_scaled[:, [selected_features.index(f) for f in top_features]]
            feat_types_top = {f: feat_types[f] for f in top_features}
            
            gower_top = compute_gower_matrix(X_top, top_features, feat_types_top, top_weights)
            labels_top, medoids_top = clara_fit(gower_top, n_clusters=k_best)
            
            if len(np.unique(labels_top)) > 1:
                sil_top = silhouette_score(gower_top, labels_top, metric='precomputed')
                dbi_top = davies_bouldin_gower(gower_top, labels_top, medoids_top)
                chi_top = safe_calinski_harabasz(X_top, labels_top)
            else:
                sil_top, dbi_top, chi_top = -1.0, np.nan, np.nan
            
            top_n_results[top_n] = {
                'features': top_features,
                'weights': top_weights,
                'labels': labels_top,
                'medoids': medoids_top,
                'silhouette': sil_top,
                'dbi': dbi_top,
                'chi': chi_top,
                'gower': gower_top
            }

        if not top_n_results:
            raise ValueError("Top-N results kosong. Pastikan jumlah fitur minimal 3.")

        # Find best Top-N scenario based on silhouette score
        best_topn_sil = -1
        best_topn_n = None
        for top_n, result in top_n_results.items():
            if result['silhouette'] > best_topn_sil:
                best_topn_sil = result['silhouette']
                best_topn_n = top_n
        
        comparison_top_n = best_topn_n if best_topn_n else (MIN_TOP_N if MIN_TOP_N in top_n_results else min(top_n_results.keys()))
        
        # t-SNE & PCA
        status_text.text("📈 6/10 Membuat visualisasi...")
        progress_bar.progress(70)
        
        perp = min(30, max(5, len(X_scaled) // 3))
        tsne_obj = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
        X_2d = tsne_obj.fit_transform(X_scaled)
        
        pca_obj = PCA(n_components=2)
        X_pca = pca_obj.fit_transform(X_scaled)
        
        # Analysis
        status_text.text("🔍 7/10 Analisis cluster...")
        progress_bar.progress(80)
        
        labels_final = labels_weighted
        df_analysis = df_clean[selected_features].copy()
        df_analysis['Cluster'] = labels_final + 1
        
        status_text.text("✅ 8/10 Selesai processing!")
        progress_bar.progress(100)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return
    
    # ====================================================================
    # DISPLAY RESULTS - MULTIPLE TABS
    # ====================================================================
    
    st.success("✅ Analisis Selesai!")
    st.markdown("---")
    
    tabs = st.tabs([
        "📋 Dataset Preview",
        "🎁 Outlier & Distribusi",
        "📊 Baseline Eval",
        "⚖️ EWM Calculation",
        "🎯 Weighted Result",
        "🏆 Top-N Result",
        "📈 Perbandingan Hasil Skenario",
        "🎨 Visualisasi Cluster"
    ])
    
    # ====================================================================
    # TAB 1: Dataset Preview
    # ====================================================================
    with tabs[0]:
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Data", n_data)
        col2.metric("Fitur", len(selected_features))
        col3.metric("Duplikat Dihapus", prep_result['n_dup'])
        col4.metric("Data Setelah Clean", len(df_clean))
        
        st.markdown("#### 📋 Data Sample (5 Baris Pertama)")
        st.dataframe(df_clean[selected_features].head(), use_container_width=True)
        
        st.markdown("#### 📈 Statistik Deskriptif")
        stats = df_clean[selected_features].describe().round(2)
        st.dataframe(stats, use_container_width=True)
        
        st.markdown("#### 🔢 Data Info")
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown("**Jumlah Missing Values:**")
            st.write(f"🔹 {df_clean[selected_features].isnull().sum().sum()} nilai")
            
        with info_col2:
            st.markdown("**Dimensi Data (Shape):**")
            st.write(f"🔹 {df_clean[selected_features].shape[0]} baris × {df_clean[selected_features].shape[1]} kolom")
        
        st.markdown("**Tipe Data per Kolom:**")
        dtypes_df = pd.DataFrame({'Kolom': df_clean[selected_features].columns, 'Tipe Data': df_clean[selected_features].dtypes.values})
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)
    
    # ====================================================================
    # TAB 2: Outlier & Distribution
    # ====================================================================
    with tabs[1]:
        st.markdown("### 🎁 Outlier Detection & Feature Distribution")
        
        # Outlier Info
        st.markdown("#### 🔍 Hasil Deteksi Outlier (IQR Method)")
        if prep_result['outlier_info']:
            st.dataframe(pd.DataFrame(prep_result['outlier_info']), use_container_width=True)
        else:
            st.info("✅ Tidak ada outlier terdeteksi")
        
        # Before-After Boxplot
        st.markdown("#### 📦 Box Plot Sebelum & Sesudah Capping")
        
        out_cols_display = [c for c in OUTLIER_COLS if c in selected_features]
        if out_cols_display:
            n_cols = len(out_cols_display)
            fig, axes = plt.subplots(2, n_cols, figsize=(15, 8))
            if n_cols == 1:
                axes = axes.reshape(2, 1)
            
            for idx, col in enumerate(out_cols_display):
                # Before
                axes[0, idx].boxplot(prep_result['df_enc'][col], vert=True)
                axes[0, idx].set_title(f"{col}\n(Sebelum Capping)", fontweight='bold')
                axes[0, idx].set_ylabel('Value')
                axes[0, idx].patch.set_facecolor('#ffebee')
                
                # After
                axes[1, idx].boxplot(df_clean[col], vert=True)
                axes[1, idx].set_title(f"{col}\n(Sesudah Capping)", fontweight='bold')
                axes[1, idx].set_ylabel('Value')
                axes[1, idx].patch.set_facecolor('#e8f5e9')
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        
        # Feature Distribution
        st.markdown("#### 📊 Distribusi Fitur (Histogram + KDE)")
        
        n_features = len(selected_features)
        n_rows = (n_features + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows*4))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, col in enumerate(selected_features):
            axes[idx].hist(df_clean[col], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f"Distribusi {col}", fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    # ====================================================================
    # TAB 3: CLARA Baseline Evaluation
    # ====================================================================
    with tabs[2]:
        st.markdown("### 📊 CLARA Baseline Clustering Evaluation")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("k Range", f"{k_min} - {k_max}")
        col2.metric("k Best", k_best)
        
        # Get SC, DBI, and CHI for best k
        best_k_row = df_eval[df_eval['k'] == k_best]
        sc_best = best_k_row['silhouette'].values[0] if len(best_k_row) > 0 else 0.0
        dbi_best = best_k_row['dbi'].values[0] if len(best_k_row) > 0 else 0.0
        chi_best = best_k_row['chi'].values[0] if len(best_k_row) > 0 else 0.0
        col3.metric("SC (Silhouette)", f"{sc_best:.4f}")
        col4.metric("DBI", f"{dbi_best:.4f}")
        col5.metric("CHI", f"{chi_best:.4f}")
        
        st.markdown("#### 📈 Evaluation Metrics per k")
        st.dataframe(df_eval[['k', 'silhouette', 'dbi', 'chi']].round(4), use_container_width=True)
        
        # Plots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5))
        
        ax1.plot(df_eval['k'], df_eval['silhouette'], 'o-', linewidth=2, markersize=8, color='steelblue')
        ax1.axvline(k_best, color='red', linestyle='--', alpha=0.5, label=f'k_best={k_best}')
        ax1.set_xlabel('k', fontsize=11)
        ax1.set_ylabel('Silhouette Score', fontsize=11)
        ax1.set_title('Silhouette Score vs k', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(df_eval['k'], df_eval['dbi'], 's-', linewidth=2, markersize=8, color='darkorange')
        ax2.axvline(k_best, color='red', linestyle='--', alpha=0.5, label=f'k_best={k_best}')
        ax2.set_xlabel('k', fontsize=11)
        ax2.set_ylabel('Davies Bouldin Index', fontsize=11)
        ax2.set_title('DBI vs k (Lower is Better)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax3.plot(df_eval['k'], df_eval['chi'], '^-', linewidth=2, markersize=8, color='seagreen')
        ax3.axvline(k_best, color='red', linestyle='--', alpha=0.5, label=f'k_best={k_best}')
        ax3.set_xlabel('k', fontsize=11)
        ax3.set_ylabel('Calinski-Harabasz Index', fontsize=11)
        ax3.set_title('CHI vs k (Higher is Better)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        # Cluster Distribution Baseline
        st.markdown(f"#### 🎯 Cluster Distribution (k={k_best})")
        labels_baseline = results_baseline[k_best]['labels'] + 1
        counts = pd.Series(labels_baseline).value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        with col1:
            dist_text = ""
            for cid in sorted(counts.index):
                count = counts[cid]
                pct = count / len(labels_baseline) * 100
                dist_text += f"**Cluster {cid}:** {count} UMKM ({pct:.1f}%)\n"
            st.markdown(dist_text)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(counts.index, counts.values, color='steelblue', alpha=0.8)
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Jumlah UMKM')
            ax.set_title(f'Distribusi Cluster (Baseline, k={k_best})', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig, use_container_width=True)
        
        # Medoid Display Baseline
        st.divider()
        medoids_baseline = results_baseline[k_best]['medoids']
        display_medoids(medoids_baseline, prep_result['df_proc'], selected_features, 
                       results_baseline[k_best]['labels'], "Medoid per Cluster (Baseline)")
    
    # ====================================================================
    # TAB 4: EWM Calculation Detailed
    # ====================================================================
    with tabs[3]:
        st.markdown("### ⚖️ Entropy Weight Method (EWM) Calculation")
        
        st.markdown("#### 📊 EWM Summary Table")
        st.dataframe(df_ewm[['feat', 'entropy', 'diversification', 'weight']].round(6), use_container_width=True)
        
        st.markdown("#### 🔢 Perhitungan Detail (Formula)")
        st.markdown("""
        **Formula EWM:**
        
        1. **Proporsi:** $p_{ij} = \\frac{x_{ij}}{\\sum_i x_{ij}}$
        
        2. **Entropy:** $e_j = -k \\sum_i p_{ij} \\cdot \\ln(p_{ij})$, where $k = \\frac{1}{\\ln(n)}$
        
        3. **Diversification:** $d_j = 1 - e_j$
        
        4. **Weight:** $w_j = \\frac{d_j}{\\sum_j d_j}$
        """)
        
        # Verifikasi weights sum
        total_weight = df_ewm['weight'].sum()
        st.markdown(f"**✅ Verifikasi:** Total Weight = **{total_weight:.6f}** (should be ≈ 1.0)")
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        df_sort = df_ewm.sort_values('weight', ascending=True)
        colors = plt.cm.viridis(np.linspace(0, 1, len(df_sort)))
        ax.barh(range(len(df_sort)), df_sort['weight'].values, color=colors)
        ax.set_yticks(range(len(df_sort)))
        ax.set_yticklabels(df_sort['feat'].values)
        ax.set_xlabel('Weight', fontsize=11)
        ax.set_title('Bobot EWM per Fitur', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (feat, weight) in enumerate(zip(df_sort['feat'], df_sort['weight'])):
            ax.text(weight + 0.005, i, f'{weight*100:.2f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        # Detailed calculation for first feature
        st.markdown("#### 📋 Contoh Perhitungan (Fitur Pertama)")
        first_feat = df_ewm.iloc[0]['feat']
        st.info(f"**Fitur Pilihan:** {first_feat}")
        
        col_idx = selected_features.index(first_feat)
        sample_n = min(10, len(X_scaled))
        
        calc_df = pd.DataFrame({
            'Index': range(sample_n),
            'x_ij': X_scaled[:sample_n, col_idx],
            'p_ij': X_scaled[:sample_n, col_idx] / X_scaled[:, col_idx].sum(),
        })
        calc_df['ln(p_ij)'] = np.where(calc_df['p_ij'] > 0, np.log(calc_df['p_ij']), 0.0)
        calc_df['p_ij*ln(p_ij)'] = calc_df['p_ij'] * calc_df['ln(p_ij)']
        
        st.dataframe(calc_df.round(6), use_container_width=True)
        
        st.text(f"Sum of p_ij*ln(p_ij): {calc_df['p_ij*ln(p_ij)'].sum():.6f}")
        st.text(f"Entropy (ej): {df_ewm.iloc[0]['entropy']:.6f}")
        st.text(f"Diversification (dj): {df_ewm.iloc[0]['diversification']:.6f}")
        st.text(f"Weight (wj): {df_ewm.iloc[0]['weight']:.6f}")
    
    # ====================================================================
    # TAB 5: Weighted Clustering Results
    # ====================================================================
    with tabs[4]:
        st.markdown("### 🎯 CLARA Weighted Clustering Results")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("k Range", f"{k_min} - {k_max}")
        col2.metric("k Best", k_best)
        col3.metric("Silhouette", f"{sil_weighted:.4f}")
        col4.metric("DBI", f"{dbi_weighted:.4f}")
        col5.metric("CHI", f"{chi_weighted:.4f}")
        
        st.markdown("#### 📈 Evaluation Metrics per k")
        st.dataframe(df_eval_weighted[['k', 'silhouette', 'dbi', 'chi']].round(4), use_container_width=True)
        
        # Plots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5))
        
        ax1.plot(df_eval_weighted['k'], df_eval_weighted['silhouette'], 'o-', linewidth=2, markersize=8, color='green')
        ax1.axvline(k_best, color='red', linestyle='--', alpha=0.5, label=f'k_best={k_best}')
        ax1.set_xlabel('k', fontsize=11)
        ax1.set_ylabel('Silhouette Score', fontsize=11)
        ax1.set_title('Silhouette Score vs k (Weighted)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(df_eval_weighted['k'], df_eval_weighted['dbi'], 's-', linewidth=2, markersize=8, color='darkgreen')
        ax2.axvline(k_best, color='red', linestyle='--', alpha=0.5, label=f'k_best={k_best}')
        ax2.set_xlabel('k', fontsize=11)
        ax2.set_ylabel('Davies Bouldin Index', fontsize=11)
        ax2.set_title('DBI vs k (Lower is Better) (Weighted)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax3.plot(df_eval_weighted['k'], df_eval_weighted['chi'], '^-', linewidth=2, markersize=8, color='seagreen')
        ax3.axvline(k_best, color='red', linestyle='--', alpha=0.5, label=f'k_best={k_best}')
        ax3.set_xlabel('k', fontsize=11)
        ax3.set_ylabel('Calinski-Harabasz Index', fontsize=11)
        ax3.set_title('CHI vs k (Higher is Better) (Weighted)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        # Cluster Distribution Weighted
        st.markdown(f"#### 🎯 Cluster Distribution (k={k_best})")
        dist_weighted = pd.Series(labels_weighted + 1).value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        with col1:
            dist_text = ""
            for cid in sorted(dist_weighted.index):
                count = dist_weighted[cid]
                pct = count / len(labels_weighted) * 100
                dist_text += f"**Cluster {cid}:** {count} UMKM ({pct:.1f}%)\n"
            st.markdown(dist_text)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(dist_weighted.index, dist_weighted.values, color='darkgreen', alpha=0.8)
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Jumlah UMKM')
            ax.set_title(f'Distribusi Cluster (Weighted, k={k_best})', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig, use_container_width=True)
        
        st.markdown("#### 📊 Karakteristik per Cluster")
        means_weighted = df_analysis.groupby('Cluster')[selected_features].mean()
        st.dataframe(means_weighted.round(2), use_container_width=True)
        
        # ====== NEW: Weighted Clustering Calculation Details ======
        st.divider()
        st.markdown("#### 🔢 Gambaran Perhitungan Weighted Clustering")
        
        # Formula explanation
        st.markdown("""
        **Formula Weighted Gower Distance:**
        
        $d_{ij}^{weighted}(x, y) = \\sum_{k=1}^{p} w_k \\cdot d_k(x_k, y_k)$
        
        dimana:
        - $w_k$ = bobot fitur ke-k (dari EWM)
        - $d_k(x_k, y_k)$ = jarak gower fitur ke-k (0-1 normalized)
        - $p$ = jumlah fitur yang digunakan
        """)
        
        # Feature weight contribution
        st.markdown("**Kontribusi Bobot per Fitur:**")
        weight_contrib = pd.DataFrame({
            'Fitur': list(weights.keys()),
            'Bobot (w_k)': list(weights.values()),
            'Persentase': [f"{w*100:.2f}%" for w in weights.values()]
        })
        st.dataframe(weight_contrib, use_container_width=True, hide_index=True)
        
        # Visualization of weight contribution
        fig, ax = plt.subplots(figsize=(12, 5))
        weight_data = weight_contrib.sort_values('Bobot (w_k)', ascending=True)
        colors_w = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(weight_data)))
        bars = ax.barh(range(len(weight_data)), weight_data['Bobot (w_k)'].values, color=colors_w)
        ax.set_yticks(range(len(weight_data)))
        ax.set_yticklabels(weight_data['Fitur'].values, fontsize=10)
        ax.set_xlabel('Bobot (w_k)', fontsize=11, fontweight='bold')
        ax.set_title('Kontribusi Bobot per Fitur dalam Weighted Clustering', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        for i, (feat, weight) in enumerate(zip(weight_data['Fitur'], weight_data['Bobot (w_k)'])):
            ax.text(weight + 0.002, i, f'{weight:.4f}', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        # Sample calculation: Show distance calculation for first 5 UMKM with first medoid
        st.markdown("**📋 Contoh Perhitungan Jarak ke Medoid Cluster 1:**")
        
        try:
            medoid_idx_c1 = medoids_weighted[0]
            medoid_data = X_scaled[medoid_idx_c1]
            
            # Calculate distances for first 5 samples
            sample_indices = range(min(5, len(X_scaled)))
            sample_calcs = []
            
            for idx in sample_indices:
                sample_data = X_scaled[idx]
                contrib_details = []
                total_dist = 0.0
                
                for feat_idx, feat_name in enumerate(selected_features):
                    diff = abs(sample_data[feat_idx] - medoid_data[feat_idx])
                    weight_k = weights[feat_name]
                    weighted_contrib = weight_k * diff
                    total_dist += weighted_contrib
                    
                    contrib_details.append({
                        'Fitur': feat_name,
                        'd_k': f"{diff:.4f}",
                        'w_k': f"{weight_k:.4f}",
                        'w_k × d_k': f"{weighted_contrib:.4f}"
                    })
                
                st.markdown(f"**Sample {idx + 1}** → Medoid Cluster 1 | Total Distance: **{total_dist:.6f}**")
                calc_df_sample = pd.DataFrame(contrib_details)
                st.dataframe(calc_df_sample, use_container_width=True, hide_index=True, key=f"sample_{idx}")
                st.markdown("")
        except Exception as e:
            st.warning(f"⚠️ Tidak bisa menampilkan perhitungan detail: {str(e)}")
        
        # Comparison: Unweighted vs Weighted distance heatmap
        st.markdown("**🔥 Perbandingan Distance Matrix (Sample 10x10):**")
        
        try:
            # Compute unweighted distance for comparison
            sample_size_viz = min(10, len(X_scaled))
            sample_idx_viz = np.random.choice(len(X_scaled), size=sample_size_viz, replace=False)
            
            # Unweighted Gower distance
            gower_unweighted = compute_gower_matrix(X_scaled, selected_features, feat_types)
            dist_unweighted_sample = gower_unweighted[np.ix_(sample_idx_viz, sample_idx_viz)]
            
            # Weighted Gower distance
            dist_weighted_sample = gower_weighted[np.ix_(sample_idx_viz, sample_idx_viz)]
            
            # Visualization
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            
            # Unweighted
            im1 = ax1.imshow(dist_unweighted_sample, cmap='YlOrRd', aspect='auto')
            ax1.set_title('Unweighted Gower Distance', fontweight='bold')
            ax1.set_xlabel('Sample')
            ax1.set_ylabel('Sample')
            plt.colorbar(im1, ax=ax1)
            
            # Weighted
            im2 = ax2.imshow(dist_weighted_sample, cmap='YlOrRd', aspect='auto')
            ax2.set_title('Weighted Gower Distance (EWM)', fontweight='bold')
            ax2.set_xlabel('Sample')
            ax2.set_ylabel('Sample')
            plt.colorbar(im2, ax=ax2)
            
            # Difference
            dist_diff = dist_weighted_sample - dist_unweighted_sample
            im3 = ax3.imshow(dist_diff, cmap='PiYG', aspect='auto')
            ax3.set_title('Difference (Weighted - Unweighted)', fontweight='bold')
            ax3.set_xlabel('Sample')
            ax3.set_ylabel('Sample')
            plt.colorbar(im3, ax=ax3)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rata-rata Unweighted Distance", f"{dist_unweighted_sample.mean():.6f}")
            with col2:
                st.metric("Rata-rata Weighted Distance", f"{dist_weighted_sample.mean():.6f}")
            with col3:
                st.metric("Rata-rata Perubahan", f"{dist_diff.mean():.6f}")
        
        except Exception as e:
            st.warning(f"⚠️ Tidak bisa menampilkan perbandingan distance: {str(e)}")
        
        # Medoid Display Weighted
        st.divider()
        display_medoids(medoids_weighted, prep_result['df_proc'], selected_features, 
                       labels_weighted, "Medoid per Cluster (Weighted)")
    
    # ====================================================================
    # TAB 6: Top-N Results
    # ====================================================================
    with tabs[5]:
        st.markdown("### 🏆 CLARA Top-N Feature Selection Results")
        
        for top_n in sorted(top_n_results.keys()):
            result = top_n_results[top_n]
            features_list = result['features']
            weights_topn = result['weights']
            sil_topn = result['silhouette']
            dbi_topn = result['dbi']
            chi_topn = result['chi']
            labels_topn = result['labels']
            
            st.markdown(f"#### 🎯 Top-{top_n} Features (k={k_best})")
            
            col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
            with col1:
                st.metric("Silhouette", f"{sil_topn:.4f}")
            with col2:
                st.metric("DBI", f"{dbi_topn:.4f}")
            with col3:
                st.metric("CHI", f"{chi_topn:.4f}")
            with col4:
                st.text(f"**Features:** {', '.join(features_list)}")
            
            # Feature weights table
            feat_weight_df = pd.DataFrame({
                'Feature': features_list,
                'Original Weight': [weights[f] for f in features_list],
                'Top-N Normalized': [weights_topn[f] for f in features_list]
            })
            st.dataframe(feat_weight_df.round(6), use_container_width=True)
            
            # Cluster distribution
            dist_topn = pd.Series(labels_topn + 1).value_counts().sort_index()
            col1, col2 = st.columns(2)
            
            with col1:
                dist_txt = ""
                for cid in sorted(dist_topn.index):
                    pct = dist_topn[cid] / len(labels_topn) * 100
                    dist_txt += f"**Cluster {cid}:** {dist_topn[cid]} ({pct:.1f}%)\n"
                st.markdown(dist_txt)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(dist_topn.index, dist_topn.values, alpha=0.8)
                ax.set_xlabel('Cluster')
                ax.set_ylabel('Count')
                ax.set_title(f'Top-{top_n} Distribution')
                ax.grid(True, alpha=0.3, axis='y')
                st.pyplot(fig, use_container_width=True)
            
            # Medoid Display Top-N
            medoids_topn = result['medoids']
            display_medoids(medoids_topn, prep_result['df_proc'], selected_features, 
                           labels_topn, f"Medoid per Cluster (Top-{top_n})")
            
            st.markdown("---")
    
    # ====================================================================
    # TAB 7: Perbandingan Hasil Skenario
    # ====================================================================
    with tabs[6]:
        st.markdown("### 📈 Perbandingan Hasil Skenario")

        # Urutan skenario yang ditampilkan sesuai notebook
        scenario_order = [8, 7, 6, 5, 4, 3]

        # Create comparison dataframe (Baseline, Weighted, lalu Top-8 ... Top-3)
        comparison_data = [
            {'Method': 'Baseline', 'Silhouette': results_baseline[k_best]['silhouette'],
             'DBI': results_baseline[k_best]['dbi'], 'CHI': results_baseline[k_best]['chi'],
             'k': k_best, 'Features': len(selected_features)},
            {'Method': 'Weighted', 'Silhouette': sil_weighted, 'DBI': dbi_weighted,
             'CHI': chi_weighted, 'k': k_best, 'Features': len(selected_features)}
        ]

        for n_top in scenario_order:
            if n_top in top_n_results:
                comparison_data.append({
                    'Method': f'Top-{n_top}',
                    'Silhouette': top_n_results[n_top]['silhouette'],
                    'DBI': top_n_results[n_top]['dbi'],
                    'CHI': top_n_results[n_top]['chi'],
                    'k': k_best,
                    'Features': n_top
                })

        df_comparison = pd.DataFrame(comparison_data)

        st.markdown("#### 📊 Tabel Perbandingan Semua Skenario")
        st.dataframe(df_comparison.round(4), use_container_width=True)

        # Comparison plots untuk semua skenario
        x = np.arange(len(df_comparison))

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5))

        ax1.bar(x, df_comparison['Silhouette'], color='steelblue', alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_comparison['Method'], rotation=20, ha='right')
        ax1.set_ylabel('Silhouette Score', fontsize=11)
        ax1.set_title('Silhouette Score Semua Skenario', fontweight='bold')
        ax1.set_ylim(0, max(df_comparison['Silhouette']) * 1.2)
        ax1.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(df_comparison['Silhouette']):
            ax1.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=8)

        ax2.bar(x, df_comparison['DBI'], color='darkorange', alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels(df_comparison['Method'], rotation=20, ha='right')
        ax2.set_ylabel('Davies Bouldin Index', fontsize=11)
        ax2.set_title('DBI Semua Skenario (Lower is Better)', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(df_comparison['DBI']):
            ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=8)

        ax3.bar(x, df_comparison['CHI'], color='seagreen', alpha=0.85)
        ax3.set_xticks(x)
        ax3.set_xticklabels(df_comparison['Method'], rotation=20, ha='right')
        ax3.set_ylabel('Calinski-Harabasz Index', fontsize=11)
        ax3.set_title('CHI Semua Skenario (Higher is Better)', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(df_comparison['CHI']):
            ax3.text(i, v + max(df_comparison['CHI']) * 0.01, f'{v:.2f}', ha='center', fontsize=8)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # Cluster distribution untuk semua skenario
        st.markdown("#### 🎯 Distribusi Cluster Semua Skenario")

        scenario_labels = {
            'Baseline': results_baseline[k_best]['labels'] + 1,
            'Weighted': labels_weighted + 1
        }

        for n_top in scenario_order:
            if n_top in top_n_results:
                scenario_labels[f'Top-{n_top}'] = top_n_results[n_top]['labels'] + 1

        scenario_names = list(scenario_labels.keys())
        cols = st.columns(4)

        for idx, method_name in enumerate(scenario_names):
            with cols[idx % 4]:
                dist_vals = pd.Series(scenario_labels[method_name]).value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.bar(dist_vals.index, dist_vals.values, alpha=0.85)
                ax.set_title(method_name, fontsize=10, fontweight='bold')
                ax.set_xlabel('Cluster', fontsize=9)
                ax.set_ylabel('Count', fontsize=9)
                ax.grid(True, alpha=0.25, axis='y')
                st.pyplot(fig, use_container_width=True)
    
    # ====================================================================
    # TAB 8: Cluster Visualizations
    # ====================================================================
    with tabs[7]:
        # Determine best scenario (Baseline, Weighted, or Top-N)
        best_scenario = 'Baseline'
        best_sil_score = results_baseline[k_best]['silhouette']
        best_labels = results_baseline[k_best]['labels']
        best_medoids = results_baseline[k_best]['medoids']
        
        if sil_weighted > best_sil_score:
            best_scenario = 'Weighted'
            best_sil_score = sil_weighted
            best_labels = labels_weighted
            best_medoids = medoids_weighted
        
        if best_topn_n and top_n_results[best_topn_n]['silhouette'] > best_sil_score:
            best_scenario = f'Top-{best_topn_n}'
            best_sil_score = top_n_results[best_topn_n]['silhouette']
            best_labels = top_n_results[best_topn_n]['labels']
            best_medoids = top_n_results[best_topn_n]['medoids']
        
        st.markdown(f"### 🎨 Cluster Visualizations (Best Scenario: **{best_scenario}**)")
        st.info(f"Skenario terbaik menurut Silhouette Score: **{best_scenario}** ({best_sil_score:.4f})")
        
        # t-SNE - 3 methods
        st.markdown("#### 🔵 t-SNE Visualization")
        palette = sns.color_palette('husl', k_best)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Baseline
        for cid in range(k_best):
            mask = results_baseline[k_best]['labels'] == cid
            axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                           c=[palette[cid]], s=60, alpha=0.7, label=f'C{cid+1}')
        medoids_baseline = results_baseline[k_best]['medoids']
        axes[0].scatter(
            X_2d[medoids_baseline, 0],
            X_2d[medoids_baseline, 1],
            marker='X', s=220, c='black', edgecolors='white', linewidths=1.5,
            label='Medoid'
        )
        axes[0].set_title(f't-SNE Baseline (k={k_best})', fontweight='bold')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.2)
        
        # Weighted
        for cid in range(k_best):
            mask = labels_weighted == cid
            axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                           c=[palette[cid]], s=60, alpha=0.7, label=f'C{cid+1}')
        axes[1].scatter(
            X_2d[medoids_weighted, 0],
            X_2d[medoids_weighted, 1],
            marker='X', s=220, c='black', edgecolors='white', linewidths=1.5,
            label='Medoid'
        )
        axes[1].set_title(f't-SNE Weighted (k={k_best})', fontweight='bold')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.2)
        
        # Best Scenario (Top-N)
        for cid in range(k_best):
            mask = best_labels == cid
            axes[2].scatter(X_2d[mask, 0], X_2d[mask, 1],
                           c=[palette[cid]], s=60, alpha=0.7, label=f'C{cid+1}')
        axes[2].scatter(
            X_2d[best_medoids, 0],
            X_2d[best_medoids, 1],
            marker='X', s=220, c='black', edgecolors='white', linewidths=1.5,
            label='Medoid'
        )
        axes[2].set_title(f't-SNE {best_scenario} (k={k_best})', fontweight='bold')
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.2)
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        # PCA - 3 methods
        st.markdown("#### 📊 PCA Visualization")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        pct1, pct2 = pca_obj.explained_variance_ratio_
        
        # Baseline
        for cid in range(k_best):
            mask = results_baseline[k_best]['labels'] == cid
            axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                           c=[palette[cid]], s=60, alpha=0.7, label=f'C{cid+1}')
        axes[0].scatter(
            X_pca[medoids_baseline, 0],
            X_pca[medoids_baseline, 1],
            marker='X', s=220, c='black', edgecolors='white', linewidths=1.5,
            label='Medoid'
        )
        axes[0].set_title(f'PCA Baseline (k={k_best})', fontweight='bold')
        axes[0].set_xlabel(f'PC1 ({pct1*100:.1f}%)')
        axes[0].set_ylabel(f'PC2 ({pct2*100:.1f}%)')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.2)
        
        # Weighted
        for cid in range(k_best):
            mask = labels_weighted == cid
            axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                           c=[palette[cid]], s=60, alpha=0.7, label=f'C{cid+1}')
        axes[1].scatter(
            X_pca[medoids_weighted, 0],
            X_pca[medoids_weighted, 1],
            marker='X', s=220, c='black', edgecolors='white', linewidths=1.5,
            label='Medoid'
        )
        axes[1].set_title(f'PCA Weighted (k={k_best})', fontweight='bold')
        axes[1].set_xlabel(f'PC1 ({pct1*100:.1f}%)')
        axes[1].set_ylabel(f'PC2 ({pct2*100:.1f}%)')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.2)
        
        # Best Scenario
        for cid in range(k_best):
            mask = best_labels == cid
            axes[2].scatter(X_pca[mask, 0], X_pca[mask, 1],
                           c=[palette[cid]], s=60, alpha=0.7, label=f'C{cid+1}')
        axes[2].scatter(
            X_pca[best_medoids, 0],
            X_pca[best_medoids, 1],
            marker='X', s=220, c='black', edgecolors='white', linewidths=1.5,
            label='Medoid'
        )
        axes[2].set_title(f'PCA {best_scenario} (k={k_best})', fontweight='bold')
        axes[2].set_xlabel(f'PC1 ({pct1*100:.1f}%)')
        axes[2].set_ylabel(f'PC2 ({pct2*100:.1f}%)')
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.2)
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        # Table - Rata-rata Variabel per Cluster (Best Scenario)
        st.markdown(f"#### 📊 Rata-rata Variabel per Cluster ({best_scenario})")
        
        df_best_analysis = df_clean[selected_features].copy()
        df_best_analysis['Cluster'] = best_labels + 1
        means_best = df_best_analysis.groupby('Cluster')[selected_features].mean()
        st.dataframe(means_best.round(4), use_container_width=True)
        
        # Persebaran UMKM di tiap Cluster
        st.divider()
        st.markdown(f"#### 👥 Persebaran UMKM per Cluster ({best_scenario})")
        
        cluster_dist = pd.Series(best_labels + 1).value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Jumlah UMKM per Cluster:**")
            dist_text = ""
            for cid in sorted(cluster_dist.index):
                count = cluster_dist[cid]
                pct = count / len(best_labels) * 100
                dist_text += f"🔹 **Cluster {cid}:** {count} UMKM ({pct:.1f}%)\n"
            st.markdown(dist_text)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = [palette[int(cid-1)] for cid in cluster_dist.index]
            bars = ax.bar(cluster_dist.index, cluster_dist.values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
            ax.set_ylabel('Jumlah UMKM', fontsize=11, fontweight='bold')
            ax.set_title(f'Persebaran UMKM per Cluster ({best_scenario})', fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        
    # ====================================================================
    # SAVE MODEL DATA TO SESSION STATE FOR PREDICTION
    # ====================================================================
    
    # Store Top-4 model for prediction (k=4 is our fixed model)
    if 4 in top_n_results:
        top4_result = top_n_results[4]
        st.session_state.clustering_model = {
            'scaler': prep_result['scaler'],
            'X_scaled': X_scaled,
            'selected_features': selected_features,
            'medoids': top4_result['medoids'],
            'top_4_features': top4_result['features'],
            'labels': top4_result['labels'],
            'feat_types': feat_types,
            'weights': top4_result['weights'],
            'chi': top4_result['chi'],
            'k_best': 4,
            'df_clean': df_clean,
            'data_loaded': True
        }
        
    # ====================================================================
    # FOOTER - DOWNLOAD
    # ====================================================================
    
    st.markdown("---")
    st.markdown("### 💾 Download Hasil")
    
    # Prepare export data
    df_export = df_clean[selected_features].copy()
    df_export['Cluster_Weighted'] = labels_weighted + 1
    df_export['Cluster_Baseline'] = results_baseline[k_best]['labels'] + 1
    df_export[f'Cluster_Top{comparison_top_n}'] = top_n_results[comparison_top_n]['labels'] + 1
    
    csv = df_export.to_csv(index=False).encode('utf-8')
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download Results (CSV)",
            csv,
            f"clustering_results_k{k_best}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        st.info(f"✅ File berisi: {len(df_export)} UMKM dengan {len(df_export.columns)} kolom (fitur + 3 cluster labels)")


if __name__ == "__main__":
    main()
