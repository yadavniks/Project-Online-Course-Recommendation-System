# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix, hstack
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Online Course Recommender (Hybrid)", layout="wide")

# =========================
# UI: Sidebar
# =========================
st.sidebar.title("⚙️ Settings")

# ✅ Use a label here (not a path)
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ✅ Local fallback path (use raw string; avoid unicode-escape issues)
DEFAULT_PATH = r"C:\Users\Reliance\Desktop\project Deployment\online_course_recommendation_v2 (1).csv"


default_top_k = st.sidebar.number_input("Top-K recommendations", min_value=1, max_value=50, value=5, step=1)
alpha = st.sidebar.slider("Hybrid α (content weight)", min_value=0.0, max_value=1.0, value=0.40, step=0.05)
unique_instructor = st.sidebar.checkbox("Ensure unique instructor", value=True)

st.title("🎓 Online Course Recommendation System — Hybrid (Content + MF)")

st.markdown(
    "This app blends **content similarity** with **collaborative (item-item) similarity** from "
    "BM25-weighted item-user interactions reduced via **SVD**. Tune α to bias towards content (α→1) "
    "or collaborative signals (α→0)."
)

# =========================
# Helpers
# =========================
def _safe_numeric(df, cols):
    use = [c for c in cols if c in df.columns]
    if not use:
        return csr_matrix((df.shape[0], 0)), [], None
    df[use] = df[use].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaler = MinMaxScaler()
    arr = scaler.fit_transform(df[use])
    return csr_matrix(arr), use, scaler

def _safe_onehot(df, cols):
    ohe_cols = [c for c in cols if c in df.columns]
    if not ohe_cols:
        return csr_matrix((df.shape[0], 0)), [], None
    # Handle scikit-learn versions that changed 'sparse' -> 'sparse_output'
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    trans = ohe.fit_transform(df[ohe_cols].astype(str))
    cat_sparse = csr_matrix(trans) if isinstance(trans, np.ndarray) else trans.tocsr()
    return cat_sparse, ohe_cols, ohe

def bm25_weight(X, K1=1.2, B=0.75):
    """BM25 weighting for sparse item-user matrix."""
    X = X.tocsr().astype(np.float32)
    row_sums = np.array(X.sum(axis=1)).ravel()
    avgdl = row_sums.mean() if row_sums.size else 1.0
    rows, cols = X.nonzero()
    data = []
    for r, c in zip(rows, cols):
        tf = X[r, c]
        denom = tf + K1 * (1 - B + B * (row_sums[r] / (avgdl + 1e-9)))
        score = (tf * (K1 + 1)) / (denom + 1e-9)
        data.append(score)
    return csr_matrix((data, (rows, cols)), shape=X.shape)

# =========================
# Cached builders (recompute only when data changes)
# =========================
@st.cache_data(show_spinner=True)
def load_data_from_upload(file):
    return pd.read_csv(file)

@st.cache_data(show_spinner=True)
def load_data_from_path(path):
    return pd.read_csv(path)

@st.cache_resource(show_spinner=True)
def build_models(df,
                 tfidf_max_features=12000,
                 tfidf_ngram=(1, 3),
                 n_factors=150,
                 bm25_k1=1.2,
                 bm25_b=0.75):
    # ---------- Ensure columns exist & build combined text ----------
    for c in ["course_name", "course_description", "tags", "category", "instructor", "difficulty_level"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)

    df["combined_text"] = (
        df["course_name"] + " " +
        df["difficulty_level"] + " " +
        df["category"] + " " +
        df["tags"] + " " +
        df["instructor"] + " " +
        df["course_description"]
    ).str.lower()

    # ---------- TF-IDF ----------
    tfidf = TfidfVectorizer(stop_words="english",
                            max_features=tfidf_max_features,
                            ngram_range=tfidf_ngram,
                            min_df=2)
    content_feat = tfidf.fit_transform(df["combined_text"])

    # ---------- Numeric ----------
    num_sparse, num_cols, scaler = _safe_numeric(
        df, ["course_price", "course_duration_hours", "rating", "enrollment_numbers", "time_spent_hours"]
    )

    # ---------- One-hot (incl. instructor) ----------
    cat_sparse, cat_cols, ohe = _safe_onehot(
        df, ["difficulty_level", "certification_offered", "study_material_available", "instructor"]
    )

    # ---------- Content matrix + NN ----------
    X_content = hstack([content_feat, cat_sparse, num_sparse], format="csr")
    content_nn = NearestNeighbors(n_neighbors=200, metric="cosine", algorithm="brute", n_jobs=-1)
    content_nn.fit(X_content)

    # ---------- Interaction strength ----------
    if "rating" in df.columns and "time_spent_hours" in df.columns:
        df["interaction_strength"] = df["rating"].fillna(0.0) * np.log1p(df["time_spent_hours"].fillna(0.0))
        # if zero, fallback to rating
        df.loc[df["interaction_strength"] == 0, "interaction_strength"] = df["rating"].fillna(0.0)
    elif "rating" in df.columns:
        df["interaction_strength"] = df["rating"].fillna(0.0)
    elif "time_spent_hours" in df.columns:
        df["interaction_strength"] = df["time_spent_hours"].fillna(0.0)
    else:
        # fallback (no interactions available)
        df["interaction_strength"] = 1.0

    # ---------- Build item-user matrix ----------
    required = ["user_id", "course_id", "interaction_strength"]
    for c in required:
        if c not in df.columns:
            if c == "user_id":
                df["user_id"] = np.arange(len(df))
            if c == "course_id":
                df["course_id"] = np.arange(len(df))
    interactions = df[["user_id", "course_id", "interaction_strength"]].copy()
    interactions["user_id"] = interactions["user_id"].astype("category")
    interactions["course_id"] = interactions["course_id"].astype("category")

    user_ids = interactions["user_id"].cat.categories
    item_ids = interactions["course_id"].cat.categories

    user_index = pd.Series(range(len(user_ids)), index=user_ids)
    item_index = pd.Series(range(len(item_ids)), index=item_ids)

    rows = interactions["course_id"].map(item_index).values
    cols = interactions["user_id"].map(user_index).values
    vals = interactions["interaction_strength"].values.astype(np.float32)
    item_user_mat = csr_matrix((vals, (rows, cols)), shape=(len(item_ids), len(user_ids)))

    # ---------- BM25 + SVD item factors + NN ----------
    item_user_bm25 = bm25_weight(item_user_mat, K1=bm25_k1, B=bm25_b)
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    item_factors = svd.fit_transform(item_user_bm25)
    item_factors_norm = item_factors / (np.linalg.norm(item_factors, axis=1, keepdims=True) + 1e-9)
    item_nn = NearestNeighbors(n_neighbors=200, metric="cosine", algorithm="brute", n_jobs=-1)
    item_nn.fit(item_factors_norm)

    # ---------- Mappings ----------
    cid2dfidx = pd.Series(df.index.values, index=df["course_id"]).to_dict()
    cid2row = {int(cid): int(i) for i, cid in enumerate(item_ids)}
    row2cid = {int(i): int(cid) for i, cid in enumerate(item_ids)}

    # Pack everything
    bundle = dict(
        df=df,
        tfidf=tfidf,
        scaler=scaler,
        ohe=ohe,
        num_cols=num_cols,
        cat_cols=cat_cols,
        X_content=X_content,
        content_nn=content_nn,
        item_factors_norm=item_factors_norm,
        item_nn=item_nn,
        user_ids=list(user_ids),
        item_ids=list(item_ids),
        cid2dfidx=cid2dfidx,
        cid2row=cid2row,
        row2cid=row2cid,
    )
    return bundle

# =========================
# Recommend functions (aligned with your notebook)
# =========================
def recommend_content(bundle, course_id, top_n=5, buffer=60, ensure_unique_instructor=False):
    df = bundle["df"]; content_nn = bundle["content_nn"]; X_content = bundle["X_content"]
    cid2dfidx = bundle["cid2dfidx"]
    if course_id not in cid2dfidx:
        return pd.DataFrame([])
    idx = cid2dfidx[course_id]
    k = min(content_nn.n_neighbors, top_n + buffer)
    dists, neighs = content_nn.kneighbors(X_content[idx], n_neighbors=k, return_distance=True)
    rows = []; seen = {course_id}; seen_instr = set()
    for dist, n in zip(dists.flatten(), neighs.flatten()):
        cid = int(df.loc[n, "course_id"])
        if cid in seen: 
            continue
        rr = df.loc[n]
        instr = rr.get("instructor", None)
        if ensure_unique_instructor and instr in seen_instr:
            continue
        rows.append({
            "course_id": cid,
            "course_name": rr.get("course_name", ""),
            "instructor": instr,
            "difficulty": rr.get("difficulty_level", None),
            "rating": rr.get("rating", None),
            "content_score": round(1.0 - float(dist), 4)
        })
        seen.add(cid)
        if instr is not None:
            seen_instr.add(instr)
        if len(rows) >= top_n:
            break
    if not rows:
        return pd.DataFrame([])
    return pd.DataFrame(rows).drop_duplicates(subset=["course_name", "instructor"]).head(top_n)

def recommend_item_mf(bundle, course_id, top_n=5, buffer=250, beta=0.75):
    df = bundle["df"]; item_nn = bundle["item_nn"]; item_factors_norm = bundle["item_factors_norm"]
    cid2row = bundle["cid2row"]; row2cid = bundle["row2cid"]; cid2dfidx = bundle["cid2dfidx"]; X_content = bundle["X_content"]
    if course_id not in cid2row:
        return pd.DataFrame([])
    row = cid2row[course_id]
    k = min(item_nn.n_neighbors, top_n + buffer)
    dists, neighs = item_nn.kneighbors(item_factors_norm[row].reshape(1, -1), n_neighbors=k, return_distance=True)
    collab_sims = (1.0 - dists.flatten()).tolist()
    neighs = neighs.flatten()
    candidates = []
    for sim, r in zip(collab_sims, neighs):
        cid = int(row2cid[r])
        df_idx = cid2dfidx.get(cid, None)
        if df_idx is None:
            continue
        candidates.append((cid, sim, df_idx))
    q_idx = cid2dfidx.get(course_id)
    if q_idx is None:
        return pd.DataFrame([])
    # content sim to rerank
    if candidates:
        cand_dfidx = [t[2] for t in candidates]
        c_sims = cosine_similarity(X_content[q_idx], X_content[cand_dfidx]).flatten().tolist()
    else:
        c_sims = []
    max_enroll = df["enrollment_numbers"].max() if "enrollment_numbers" in df.columns else 0
    rows = []
    for i, (cid, collab_sim, df_idx) in enumerate(candidates):
        c_sim = c_sims[i] if i < len(c_sims) else 0.0
        pop = df.loc[df_idx].get("enrollment_numbers", 0) if "enrollment_numbers" in df.columns else 0
        pop_boost = (np.log1p(pop) / (np.log1p(max_enroll) + 1e-9)) if max_enroll > 0 else 0.0
        score = beta * collab_sim + (1 - beta) * c_sim + 0.05 * pop_boost
        rr = df.loc[df_idx]
        rows.append({
            "course_id": int(cid),
            "course_name": rr.get("course_name", ""),
            "instructor": rr.get("instructor", None),
            "difficulty": rr.get("difficulty_level", None),
            "rating": rr.get("rating", None),
            "collab_score": round(collab_sim, 4),
            "content_sim": round(c_sim, 4),
            "combined_score": round(score, 4)
        })
    if not rows:
        return pd.DataFrame([])
    return pd.DataFrame(rows).drop_duplicates(subset=["course_name", "instructor"]).sort_values(
        "combined_score", ascending=False
    ).head(top_n)

def recommend_hybrid_fixed(bundle, course_id, user_id=None, top_n=5, alpha=0.4,
                           content_pool=800, collab_pool=800, ensure_unique_instructor=False):
    # Gather candidate IDs from top content & collab lists
    cb = recommend_content(bundle, course_id, top_n=content_pool)
    cf = recommend_item_mf(bundle, course_id, top_n=collab_pool)
    df = bundle["df"]; cid2dfidx = bundle["cid2dfidx"]; X_content = bundle["X_content"]
    cid2row = bundle["cid2row"]; item_factors_norm = bundle["item_factors_norm"]

    cand_ids = set()
    if not cb.empty:
        cand_ids.update(cb["course_id"].astype(int).tolist())
    if not cf.empty:
        cand_ids.update(cf["course_id"].astype(int).tolist())
    cand_ids.discard(course_id)
    if not cand_ids:
        return pd.DataFrame([])

    # Content similarities (to query course)
    q_df_idx = cid2dfidx.get(course_id)
    if q_df_idx is None:
        return pd.DataFrame([])
    cand_list = [cid for cid in cand_ids if cid in cid2dfidx]
    cand_dfidx = [cid2dfidx[cid] for cid in cand_list]
    c_sims = cosine_similarity(X_content[q_df_idx], X_content[cand_dfidx]).flatten()
    content_map = {cid: float(c_sims[i]) for i, cid in enumerate(cand_list)}

    # Collab similarities (from cf where available; fill missing via factor cosine)
    collab_map = {}
    if not cf.empty:
        for _, r in cf.iterrows():
            collab_map[int(r["course_id"])] = float(r.get("combined_score", r.get("collab_score", 0.0)))
    if course_id in cid2row:
        q_row = cid2row[course_id]
        missing = [cid for cid in cand_list if cid not in collab_map and cid in cid2row]
        if missing:
            rows_idx = [cid2row[cid] for cid in missing]
            q_factor = item_factors_norm[q_row].reshape(1, -1)
            cand_factors = item_factors_norm[rows_idx]
            collab_sims_missing = cosine_similarity(q_factor, cand_factors).flatten()
            for i, cid in enumerate(missing):
                collab_map[cid] = float(collab_sims_missing[i])

    eps = 1e-6
    content_vals = np.array([content_map.get(cid, eps) for cid in cand_list], dtype=float)
    collab_vals = np.array([collab_map.get(cid, eps) for cid in cand_list], dtype=float)

    def _minmax(v):
        mn, mx = v.min(), v.max()
        if mx - mn <= 1e-9:
            return np.ones_like(v)
        return (v - mn) / (mx - mn)

    c_norm = _minmax(content_vals)
    m_norm = _minmax(collab_vals)

    max_enroll = df["enrollment_numbers"].max() if "enrollment_numbers" in df.columns else 0
    rows = []
    for i, cid in enumerate(cand_list):
        idx = cid2dfidx[cid]
        rr = df.loc[idx]
        pop = rr.get("enrollment_numbers", 0) if "enrollment_numbers" in rr else 0
        pop_boost = (np.log1p(pop) / (np.log1p(max_enroll) + 1e-9)) if max_enroll > 0 else 0.0
        rating_boost = (rr.get("rating", 0) / 5.0) if "rating" in rr else 0.0
        base = alpha * c_norm[i] + (1 - alpha) * m_norm[i]
        hybrid_score = base + 0.05 * pop_boost + 0.05 * rating_boost
        rows.append({
            "course_id": int(cid),
            "course_name": rr.get("course_name", ""),
            "instructor": rr.get("instructor", None),
            "difficulty": rr.get("difficulty_level", None),
            "rating": rr.get("rating", None),
            "content_score": round(float(c_norm[i]), 4),
            "collab_score": round(float(m_norm[i]), 4),
            "hybrid_score": round(float(hybrid_score), 4)
        })

    # Sort + unique instructors if requested
    out = pd.DataFrame(rows).sort_values("hybrid_score", ascending=False)
    if ensure_unique_instructor and "instructor" in out.columns:
        seen = set(); keep_idx = []
        for i, r in out.iterrows():
            instr = r.get("instructor")
            if instr in seen:
                continue
            seen.add(instr)
            keep_idx.append(i)
            if len(keep_idx) >= top_n:
                break
        return out.loc[keep_idx].reset_index(drop=True)
    else:
        return out.head(top_n).reset_index(drop=True)

# =========================
# App Flow  ✅ Upload → else fallback (no NoneType errors)
# =========================
try:
    if uploaded is not None:
        with st.spinner("Loading uploaded CSV..."):
            df = load_data_from_upload(uploaded)
        st.sidebar.success("✅ Using uploaded file")
    else:
        with st.spinner("No upload detected — loading default CSV..."):
            df = load_data_from_path(DEFAULT_PATH)
        st.sidebar.info(f"ℹ️ Using default CSV:\n{DEFAULT_PATH}")
except Exception as e:
    st.error(f"❌ Could not load a dataset.\n\n{e}")
    st.stop()

with st.spinner("Building models (TF-IDF, One-Hot, Numeric, BM25→SVD, NearestNeighbors)..."):
    bundle = build_models(df)

# Query selection
left, right = st.columns([2, 1])
with left:
    name_col = "course_name" if "course_name" in df.columns else None
    id_col = "course_id" if "course_id" in df.columns else None

    if name_col and id_col:
        names = df[[id_col, name_col]].drop_duplicates()
        # Some CSVs may have non-int IDs; cast safely
        try:
            names[id_col] = names[id_col].astype(int)
        except Exception:
            pass
        names["label"] = names[name_col].astype(str) + "  •  (id=" + names[id_col].astype(str) + ")"
        selected_label = st.selectbox("Pick a query course", names["label"].tolist())
        sel_id = int(selected_label.split("id=")[-1].rstrip(")"))
        st.caption(f"Selected course_id: **{sel_id}**")
        course_id = sel_id
    elif id_col:
        course_id = int(st.number_input("Enter course_id", min_value=0, value=int(df[id_col].iloc[0])))
    else:
        st.error("No `course_id` column found; cannot run recommendations.")
        st.stop()

with right:
    st.metric("Rows", f"{len(df):,}")
    st.metric("Unique courses", f"{df['course_id'].nunique() if 'course_id' in df.columns else len(df):,}")

# Run recommendations
run = st.button("🔎 Get Recommendations", type="primary")
if run:
    with st.spinner("Computing hybrid recommendations..."):
        recs = recommend_hybrid_fixed(bundle, course_id, top_n=default_top_k,
                                      alpha=alpha, ensure_unique_instructor=unique_instructor)
    if recs.empty:
        st.warning("No recommendations found for this query course.")
    else:
        st.subheader("⭐ Hybrid Recommendations")
        st.dataframe(recs.reset_index(drop=True))

    # Optional debug panels (content & MF individually)
    with st.expander("Advanced: inspect Content-only and MF-only results"):
        c_recs = recommend_content(bundle, course_id, top_n=default_top_k, ensure_unique_instructor=unique_instructor)
        m_recs = recommend_item_mf(bundle, course_id, top_n=default_top_k)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Content-based (top-K)**")
            st.dataframe(c_recs.reset_index(drop=True))
        with col2:
            st.markdown("**Item-item MF (top-K)**")
            st.dataframe(m_recs.reset_index(drop=True))

st.caption("Tip: Tune α — higher = more content-driven; lower = more collaborative. "
           "Enable *unique instructor* to diversify recommendations.")
