"""
Movie Recommendation System — CineMatch
========================================
Optimized for Hugging Face Spaces deployment.

movie_data.pkl is too large for GitHub/HF repo directly.
This app downloads it automatically from Google Drive on first run.

How to set your Google Drive File ID:
  1. Upload movie_data.pkl to Google Drive
  2. Right-click → Share → "Anyone with the link can view"
  3. Copy link: https://drive.google.com/file/d/YOUR_FILE_ID/view
  4. Paste YOUR_FILE_ID into GDRIVE_FILE_ID below (line 30)
"""

import random
import os
import pickle
import gdown
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# ★ PASTE YOUR GOOGLE DRIVE FILE ID HERE  (line 30)
# ─────────────────────────────────────────────────────────────────────────────
GDRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"
# Example: GDRIVE_FILE_ID = "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms"

PKL_PATH = Path(__file__).parent / "movie_data.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch – Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# API KEY
# Add TMDB_API_KEY in HF Space → Settings → Repository secrets
# ─────────────────────────────────────────────────────────────────────────────
def get_api_key() -> Optional[str]:
    try:
        return st.secrets["TMDB_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    return os.getenv("TMDB_API_KEY")

TMDB_API_KEY = get_api_key()
TMDB_BASE    = "https://api.themoviedb.org/3"
POSTER_BASE  = "https://image.tmdb.org/t/p/w500"
FALLBACK_IMG = "https://placehold.co/300x450/1a1a2e/facc15?text=No+Poster"

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Netflix-inspired dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
:root{--bg:#0a0a0f;--surface:#12121a;--card:#1a1a28;--accent:#facc15;--accent2:#f59e0b;--text:#e5e7eb;--muted:#9ca3af;--border:rgba(250,204,21,0.15);}
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background-color:var(--bg)!important;color:var(--text)!important;}
.main .block-container{background-color:var(--bg)!important;padding-top:1.5rem!important;max-width:1400px;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:var(--text)!important;}
[data-testid="stSidebar"] .stTextInput input,[data-testid="stSidebar"] select{background:var(--card)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:8px!important;}
h1{color:var(--accent)!important;letter-spacing:-1px;font-weight:900!important;}
h2,h3{color:var(--accent)!important;font-weight:700!important;}
.stButton>button{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;color:#000!important;font-weight:700!important;border:none!important;border-radius:10px!important;padding:0.5rem 1.4rem!important;transition:transform .15s,box-shadow .15s!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 6px 20px rgba(250,204,21,0.4)!important;}
.stButton>button:active{transform:translateY(0)!important;}
.stSlider>div>div>div>div{background:var(--accent)!important;}
hr{border-color:var(--border)!important;}
.movie-card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:12px;transition:transform .2s,box-shadow .2s,border-color .2s;height:100%;}
.movie-card:hover{transform:translateY(-6px);box-shadow:0 14px 36px rgba(250,204,21,0.2);border-color:var(--accent);}
.movie-card img{border-radius:10px;width:100%;object-fit:cover;}
.movie-title{font-size:0.9rem;font-weight:700;color:var(--text);margin-top:8px;line-height:1.3;}
.movie-meta{font-size:0.75rem;color:var(--muted);margin-top:4px;line-height:1.5;}
.similarity-badge{display:inline-block;background:linear-gradient(135deg,var(--accent),var(--accent2));color:#000;font-weight:700;font-size:0.72rem;padding:2px 8px;border-radius:20px;margin-top:6px;}
.genre-tag{display:inline-block;background:rgba(250,204,21,0.12);color:var(--accent);font-size:0.68rem;padding:2px 7px;border-radius:12px;margin:2px 2px 0 0;border:1px solid var(--border);}
.list-card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:16px;margin-bottom:14px;transition:border-color .2s,box-shadow .2s;}
.list-card:hover{border-color:var(--accent);box-shadow:0 8px 28px rgba(250,204,21,0.15);}
.stat-box{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px;text-align:center;}
.stat-value{font-size:2rem;font-weight:900;color:var(--accent);line-height:1;}
.stat-label{font-size:0.8rem;color:var(--muted);margin-top:4px;}
#MainMenu,footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE DRIVE DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
def download_from_gdrive(file_id: str, dest: Path) -> bool:
    """Download pkl from Google Drive using gdown. Returns True on success."""
    try:
        url = f"https://drive.google.com/uc?id={1M77RUKLCPAdIgE7M2JGiNhIAHBqShECz}"
        gdown.download(url, str(dest), quiet=False)
        return dest.exists() and dest.stat().st_size > 0
    except Exception as e:
        st.error(f"❌ Google Drive download failed: {e}")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_movie_data(path: str):
    p = Path(path)

    # Download from Google Drive if not already present
    if not p.exists():
        if GDRIVE_FILE_ID == "YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
            st.error("❌ Please set your GDRIVE_FILE_ID in app.py (line 30).")
            st.stop()
        with st.spinner("⬇️ Downloading model data from Google Drive (~250 MB) — only happens once…"):
            ok = download_from_gdrive(GDRIVE_FILE_ID, p)
        if not ok:
            st.error("❌ Download failed. Check your GDRIVE_FILE_ID and make sure the file is shared publicly.")
            st.stop()

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, (list, tuple)) and len(data) == 2:
        movies_df, cosine_sim = data
    elif isinstance(data, dict):
        movies_df  = data.get("movies")
        cosine_sim = data.get("cosine_sim")
    else:
        raise ValueError("Unexpected pickle format. Expected (movies_df, cosine_sim) tuple.")

    missing = {"title", "movie_id"} - set(movies_df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    movies_df = movies_df.reset_index(drop=True)
    movies_df["title_lower"] = movies_df["title"].str.lower().fillna("")
    return movies_df, cosine_sim

# ─────────────────────────────────────────────────────────────────────────────
# TMDB API
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_tmdb_details(movie_id: int) -> dict:
    default = {"poster_url": FALLBACK_IMG, "release_date": "N/A", "vote_average": "N/A", "genres": [], "overview": ""}
    if not TMDB_API_KEY or not movie_id or pd.isna(movie_id):
        return default
    try:
        r = requests.get(f"{TMDB_BASE}/movie/{int(movie_id)}?api_key={TMDB_API_KEY}", timeout=8)
        r.raise_for_status()
        d = r.json()
    except Exception:
        return default
    poster = d.get("poster_path")
    return {
        "poster_url":   f"{POSTER_BASE}{poster}" if poster else FALLBACK_IMG,
        "release_date": d.get("release_date", "N/A") or "N/A",
        "vote_average": round(float(d.get("vote_average", 0) or 0), 1),
        "genres":       [g["name"] for g in d.get("genres", []) if isinstance(g, dict)],
        "overview":     d.get("overview", "") or "",
    }

# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def get_recommendations(title: str, movies: pd.DataFrame, cosine_sim, top_n: int = 10) -> pd.DataFrame:
    matches = movies[movies["title"] == title]
    if matches.empty:
        raise ValueError(f"Movie '{title}' not found in dataset.")
    idx        = matches.index[0]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:top_n + 1]
    indices    = [i[0] for i in sim_scores]
    scores     = [round(float(s[1]) * 100, 1) for s in sim_scores]
    cols       = [c for c in ["title", "movie_id", "overview"] if c in movies.columns]
    result     = movies[cols].iloc[indices].copy()
    result["similarity"] = scores
    return result.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# CARD RENDERERS
# ─────────────────────────────────────────────────────────────────────────────
def render_grid_card(row: pd.Series) -> None:
    d       = fetch_tmdb_details(int(row["movie_id"]))
    sim     = row.get("similarity", None)
    ov_full = row.get("overview", "") or ""
    ov      = ov_full[:100] + ("…" if len(ov_full) > 100 else "")
    ghtml   = "".join(f'<span class="genre-tag">{g}</span>' for g in d["genres"][:3])
    meta    = " &nbsp;·&nbsp; ".join(p for p in [
        d["release_date"][:4] if d["release_date"] != "N/A" else "",
        f"⭐ {d['vote_average']}" if d["vote_average"] != "N/A" else ""
    ] if p)
    badge = f'<span class="similarity-badge">🎯 {sim}% match</span>' if sim is not None else ""
    st.markdown(f"""
    <div class="movie-card">
        <img src="{d['poster_url']}" alt="{row['title']}" loading="lazy"/>
        <div class="movie-title">{row['title']}</div>
        <div class="movie-meta">{meta}</div>
        <div style="margin-top:6px">{ghtml}</div>
        {badge}
        <div class="movie-meta" style="margin-top:6px">{ov}</div>
    </div>""", unsafe_allow_html=True)

def render_list_card(row: pd.Series) -> None:
    d     = fetch_tmdb_details(int(row["movie_id"]))
    sim   = row.get("similarity", None)
    ov    = row.get("overview", "") or "No overview available."
    ghtml = "".join(f'<span class="genre-tag">{g}</span>' for g in d["genres"])
    badge = f'<span class="similarity-badge">🎯 {sim}% match</span>' if sim is not None else ""
    c1, c2 = st.columns([1, 4])
    with c1:
        st.markdown(f'<img src="{d["poster_url"]}" style="width:100%;border-radius:10px;" loading="lazy"/>', unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="list-card">
            <div style="font-size:1.1rem;font-weight:700;color:#facc15">{row['title']}</div>
            <div class="movie-meta" style="margin:6px 0">📅 {d['release_date']} &nbsp;·&nbsp; ⭐ {d['vote_average']}</div>
            <div style="margin-bottom:8px">{ghtml}</div>
            <div style="font-size:0.85rem;color:#d1d5db;line-height:1.6">{ov}</div>
            <div style="margin-top:10px">{badge}</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
try:
    movies, cosine_sim = load_movie_data(str(PKL_PATH))
except Exception as exc:
    st.error(f"❌ Failed to load movie data: {exc}")
    st.stop()

all_titles: list = movies["title"].dropna().tolist()

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "active_movie"    not in st.session_state: st.session_state.active_movie    = all_titles[0] if all_titles else ""
if "recommendations" not in st.session_state: st.session_state.recommendations = None
if "rec_base"        not in st.session_state: st.session_state.rec_base        = ""

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="margin-bottom:0">🎬 CineMatch</h1>
<p style="color:#9ca3af;margin-top:4px;font-size:1rem">
    Discover movies you'll love · Powered by AI cosine similarity &amp; TMDB
</p>""", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='margin-bottom:0.5rem'>🎛️ Controls</h2>", unsafe_allow_html=True)

    q        = st.text_input("🔍 Search movie by name", placeholder="e.g. Inception, Avatar…").strip().lower()
    filtered = [t for t in all_titles if q in t.lower()] if q else all_titles
    if q and not filtered:
        st.warning("No match. Showing all.")
        filtered = all_titles

    active      = st.session_state.active_movie
    default_idx = filtered.index(active) if active in filtered else 0
    selected    = st.selectbox("🎥 Select a movie", options=filtered, index=default_idx)
    st.session_state.active_movie = selected

    st.markdown("---")
    if st.button("🎲 Surprise Me", use_container_width=True):
        pick = random.choice(all_titles)
        st.session_state.active_movie    = pick
        st.session_state.recommendations = None
        st.session_state.rec_base        = ""
        st.success(f"🎬 Random pick: **{pick}**")
        st.rerun()

    st.markdown("---")
    top_n        = st.slider("🔢 Recommendations", 5, 20, 10)
    layout       = st.radio("📐 Layout", ["🃏 Cards (grid)", "📋 List"])
    cols_per_row = st.slider("📊 Cards per row", 2, 6, 4, disabled=(layout != "🃏 Cards (grid)"))
    st.markdown("---")
    recommend_clicked = st.button("✨ Get Recommendations", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown(f'<div style="text-align:center"><div class="stat-value">{len(all_titles):,}</div><div class="stat-label">Movies in dataset</div></div>', unsafe_allow_html=True)
    if not TMDB_API_KEY:
        st.warning("⚠️ No TMDB API key.\nAdd TMDB_API_KEY in Space → Settings → Secrets.", icon="🔑")

# ─────────────────────────────────────────────────────────────────────────────
# MOVIE PREVIEW
# ─────────────────────────────────────────────────────────────────────────────
current = st.session_state.active_movie
match   = movies[movies["title"] == current]
if not match.empty:
    mid = int(match.iloc[0]["movie_id"])
    ov  = match.iloc[0].get("overview", "") or ""
    with st.spinner("Loading movie details…"):
        d = fetch_tmdb_details(mid)
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown(f'<img src="{d["poster_url"]}" style="width:100%;border-radius:14px;box-shadow:0 8px 32px rgba(250,204,21,0.2)"/>', unsafe_allow_html=True)
    with c2:
        gtags = "".join(f'<span class="genre-tag">{g}</span>' for g in d["genres"])
        st.markdown(f"""
        <h2 style="margin-bottom:4px">{current}</h2>
        <div class="movie-meta">📅 {d['release_date']} &nbsp;·&nbsp; ⭐ {d['vote_average']} / 10</div>
        <div style="margin:10px 0">{gtags}</div>
        <p style="font-size:0.95rem;line-height:1.7;color:#d1d5db">{ov or 'No overview available.'}</p>
        """, unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
if recommend_clicked:
    with st.spinner(f"🔍 Finding movies similar to **{current}**…"):
        try:
            recs = get_recommendations(current, movies, cosine_sim, top_n=top_n)
            st.session_state.recommendations = recs
            st.session_state.rec_base        = current
        except Exception as exc:
            st.error(f"❌ {exc}")
            st.session_state.recommendations = None

recs_df  = st.session_state.get("recommendations")
rec_base = st.session_state.get("rec_base", "")

if recs_df is not None and not recs_df.empty:
    st.markdown(f"<h2>🎯 Because you liked &nbsp;<em>{rec_base}</em></h2>", unsafe_allow_html=True)
    st.caption(f"Showing top {len(recs_df)} recommendations · similarity scored 0–100%")
    st.markdown("")
    if layout == "🃏 Cards (grid)":
        for i in range(0, len(recs_df), cols_per_row):
            for col, (_, r) in zip(st.columns(cols_per_row), recs_df.iloc[i:i+cols_per_row].iterrows()):
                with col: render_grid_card(r)
            st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)
    else:
        for _, r in recs_df.iterrows(): render_list_card(r)

elif not recommend_clicked:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px">
        <div style="font-size:4rem">🍿</div>
        <h3 style="color:#9ca3af;font-weight:400;margin-top:16px">
            Pick a movie, then click <strong style="color:#facc15">✨ Get Recommendations</strong>
        </h3>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────────────────────
if recs_df is not None and not recs_df.empty:
    st.markdown("---")
    st.markdown("<h2>📊 Recommendation Stats</h2>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    for col, val, lbl in [
        (s1, str(len(recs_df)),                      "Movies Recommended"),
        (s2, f"{recs_df['similarity'].mean():.1f}%",  "Avg. Similarity"),
        (s3, f"{recs_df['similarity'].max():.1f}%",   "Highest Match"),
        (s4, f"{recs_df['similarity'].min():.1f}%",   "Lowest Match"),
    ]:
        with col:
            st.markdown(f'<div class="stat-box"><div class="stat-value">{val}</div><div class="stat-label">{lbl}</div></div>', unsafe_allow_html=True)
