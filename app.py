"""
Movie Recommendation System
============================
A professional Streamlit app that uses cosine similarity to recommend
movies and fetches rich metadata (poster, rating, genres, overview)
from the TMDB API.

Run with:  streamlit run app.py
"""

import random
import os
import pickle
from typing import Optional

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch – Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT / API KEY
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv(override=True)  # load .env if present and override

def get_api_key() -> Optional[str]:
    """
    Retrieve the TMDB API key from st.secrets (preferred) or the .env file.
    Returns None if not found – graceful degradation is handled downstream.
    """
    # 1. Try Streamlit secrets (production / cloud deployments)
    try:
        return st.secrets["TMDB_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    # 2. Fall back to environment variable from .env
    key = os.getenv("TMDB_API_KEY")
    return key if key else None

TMDB_API_KEY = get_api_key()
TMDB_BASE    = "https://api.themoviedb.org/3"
POSTER_BASE  = "https://image.tmdb.org/t/p/w500"
FALLBACK_IMG = "https://placehold.co/300x450/1a1a2e/facc15?text=No+Poster"

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  – Netflix-inspired dark theme with yellow accents
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    /* ── Root palette ── */
    :root {
        --bg:        #0a0a0f;
        --surface:   #12121a;
        --card:      #1a1a28;
        --accent:    #facc15;
        --accent2:   #f59e0b;
        --text:      #e5e7eb;
        --muted:     #9ca3af;
        --border:    rgba(250,204,21,0.15);
    }

    /* ── Base reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    /* ── Main area ── */
    .main .block-container {
        background-color: var(--bg) !important;
        padding-top: 1.5rem !important;
        max-width: 1400px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stSelectbox select,
    [data-testid="stSidebar"] select {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 8px !important;
    }

    /* ── Headings ── */
    h1 { color: var(--accent) !important; letter-spacing: -1px; font-weight: 900 !important; }
    h2, h3 { color: var(--accent) !important; font-weight: 700 !important; }

    /* ── Accent buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
        color: #000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.4rem !important;
        transition: transform .15s, box-shadow .15s !important;
        letter-spacing: .3px;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(250,204,21,0.4) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* ── Slider ── */
    .stSlider > div > div > div > div { background: var(--accent) !important; }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; }

    /* ── Info / warning boxes ── */
    .stAlert { border-radius: 10px !important; }

    /* ── Movie card container ── */
    .movie-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 12px;
        transition: transform .2s, box-shadow .2s, border-color .2s;
        height: 100%;
    }
    .movie-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 14px 36px rgba(250,204,21,0.2);
        border-color: var(--accent);
    }
    .movie-card img {
        border-radius: 10px;
        width: 100%;
        object-fit: cover;
    }
    .movie-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: var(--text);
        margin-top: 8px;
        line-height: 1.3;
    }
    .movie-meta {
        font-size: 0.75rem;
        color: var(--muted);
        margin-top: 4px;
        line-height: 1.5;
    }
    .similarity-badge {
        display: inline-block;
        background: linear-gradient(135deg, var(--accent), var(--accent2));
        color: #000;
        font-weight: 700;
        font-size: 0.72rem;
        padding: 2px 8px;
        border-radius: 20px;
        margin-top: 6px;
    }
    .genre-tag {
        display: inline-block;
        background: rgba(250,204,21,0.12);
        color: var(--accent);
        font-size: 0.68rem;
        padding: 2px 7px;
        border-radius: 12px;
        margin: 2px 2px 0 0;
        border: 1px solid var(--border);
    }

    /* ── List card ── */
    .list-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
        transition: border-color .2s, box-shadow .2s;
    }
    .list-card:hover {
        border-color: var(--accent);
        box-shadow: 0 8px 28px rgba(250,204,21,0.15);
    }

    /* ── Stats box ── */
    .stat-box {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 900;
        color: var(--accent);
        line-height: 1;
    }
    .stat-label { font-size: 0.8rem; color: var(--muted); margin-top: 4px; }

    /* ── Hide Streamlit menu/footer ── */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_movie_data(path: str = "movie_data.pkl"):
    """
    Load the movies DataFrame and cosine-similarity matrix from a pickle file.
    The pickle is expected to contain a tuple: (movies_df, cosine_sim_matrix).
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Support both tuple/list and dict formats
    if isinstance(data, (list, tuple)) and len(data) == 2:
        movies_df, cosine_sim = data
    elif isinstance(data, dict):
        movies_df  = data.get("movies")
        cosine_sim = data.get("cosine_sim")
    else:
        raise ValueError("Unexpected pickle format. Expected (movies_df, cosine_sim) tuple.")

    # Ensure required columns exist
    required = {"title", "movie_id"}
    missing  = required - set(movies_df.columns)
    if missing:
        raise KeyError(f"Missing columns in movies DataFrame: {missing}")

    movies_df = movies_df.reset_index(drop=True)
    movies_df["title_lower"] = movies_df["title"].str.lower().fillna("")
    return movies_df, cosine_sim


# ─────────────────────────────────────────────────────────────────────────────
# TMDB API HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_tmdb_details(movie_id: int) -> dict:
    """
    Fetch poster, release date, vote average, genres, and overview from TMDB.
    Cached for 1 hour to avoid redundant API calls. 
    Cache explicitly invalidated.
    Returns a safe dict with fallback values on any error.
    """
    default = {
        "poster_url":   FALLBACK_IMG,
        "release_date": "N/A",
        "vote_average": "N/A",
        "genres":       [],
        "overview":     "",
    }

    if not TMDB_API_KEY:
        return default  # graceful degradation – no API key
    if not movie_id or pd.isna(movie_id):
        return default

    try:
        url      = f"{TMDB_BASE}/movie/{int(movie_id)}?api_key={TMDB_API_KEY}"
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        st.warning("⏱ TMDB request timed out. Some details may be missing.", icon="⚠️")
        return default
    except requests.exceptions.RequestException:
        return default
    except Exception:
        return default

    poster_path = data.get("poster_path")
    genres_raw  = data.get("genres", [])

    return {
        "poster_url":   f"{POSTER_BASE}{poster_path}" if poster_path else FALLBACK_IMG,
        "release_date": data.get("release_date", "N/A") or "N/A",
        "vote_average": round(float(data.get("vote_average", 0) or 0), 1),
        "genres":       [g["name"] for g in genres_raw if isinstance(g, dict)],
        "overview":     data.get("overview", "") or "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def get_recommendations(
    title: str,
    movies: pd.DataFrame,
    cosine_sim,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return the top-N most similar movies to `title` using cosine similarity.
    Raises ValueError if the title is not found in the dataset.
    """
    matches = movies[movies["title"] == title]
    if matches.empty:
        raise ValueError(f"Movie '{title}' not found in dataset.")

    idx        = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : top_n + 1]  # exclude the movie itself

    movie_indices = [i[0] for i in sim_scores]
    scores        = [round(float(s[1]) * 100, 1) for s in sim_scores]  # as %

    cols_to_keep = [c for c in ["title", "movie_id", "overview"] if c in movies.columns]
    result = movies[cols_to_keep].iloc[movie_indices].copy()
    result["similarity"] = scores
    return result.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# UI CARD RENDERERS
# ─────────────────────────────────────────────────────────────────────────────
def render_grid_card(row: pd.Series) -> None:
    """Render a compact movie card for the grid (cards) layout."""
    details   = fetch_tmdb_details(int(row["movie_id"]))
    title     = row["title"]
    sim       = row.get("similarity", None)
    overview  = row.get("overview", "") or ""
    genres    = details["genres"][:3]  # show max 3 genre tags

    genre_html = "".join(f'<span class="genre-tag">{g}</span>' for g in genres)

    rating_str = (
        f"⭐ {details['vote_average']}" if details["vote_average"] != "N/A" else ""
    )
    date_str = (
        details["release_date"][:4] if details["release_date"] != "N/A" else ""
    )
    meta_parts = [p for p in [date_str, rating_str] if p]
    meta_str   = " &nbsp;·&nbsp; ".join(meta_parts)
    sim_badge  = (
        f'<span class="similarity-badge">🎯 {sim}% match</span>' if sim is not None else ""
    )
    overview_short = overview[:100] + ("…" if len(overview) > 100 else "")

    st.markdown(
        f"""
        <div class="movie-card">
            <img src="{details['poster_url']}" alt="{title}" loading="lazy"/>
            <div class="movie-title">{title}</div>
            <div class="movie-meta">{meta_str}</div>
            <div style="margin-top:6px">{genre_html}</div>
            {sim_badge}
            <div class="movie-meta" style="margin-top:6px">{overview_short}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_list_card(row: pd.Series) -> None:
    """Render an expanded movie card for the list layout."""
    details  = fetch_tmdb_details(int(row["movie_id"]))
    title    = row["title"]
    sim      = row.get("similarity", None)
    overview = row.get("overview", "") or "No overview available."

    genres_html = "".join(
        f'<span class="genre-tag">{g}</span>' for g in details["genres"]
    )
    sim_html = (
        f'<span class="similarity-badge">🎯 {sim}% match</span>' if sim is not None else ""
    )

    col_img, col_info = st.columns([1, 4])
    with col_img:
        st.markdown(
            f'<img src="{details["poster_url"]}" style="width:100%;border-radius:10px;" loading="lazy"/>',
            unsafe_allow_html=True,
        )
    with col_info:
        st.markdown(
            f"""
            <div class="list-card">
                <div style="font-size:1.1rem;font-weight:700;color:#facc15">{title}</div>
                <div class="movie-meta" style="margin:6px 0">
                    📅 {details['release_date']} &nbsp;·&nbsp;
                    ⭐ {details['vote_average']}
                </div>
                <div style="margin-bottom:8px">{genres_html}</div>
                <div style="font-size:0.85rem;color:#d1d5db;line-height:1.6">{overview}</div>
                <div style="margin-top:10px">{sim_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA  (cached – runs once)
# ─────────────────────────────────────────────────────────────────────────────
try:
    movies, cosine_sim = load_movie_data("movie_data.pkl")
except FileNotFoundError:
    st.error("❌ `movie_data.pkl` not found. Please place it in the project directory.")
    st.stop()
except Exception as exc:
    st.error(f"❌ Failed to load movie data: {exc}")
    st.stop()

all_titles: list[str] = movies["title"].dropna().tolist()

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE  INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
if "active_movie" not in st.session_state:
    st.session_state.active_movie = all_titles[0] if all_titles else ""

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None  # DataFrame or None

if "rec_base" not in st.session_state:
    st.session_state.rec_base = ""  # which movie the recs were generated for


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style="margin-bottom:0">🎬 CineMatch</h1>
    <p style="color:#9ca3af;margin-top:4px;font-size:1rem">
        Discover movies you'll love · Powered by AI cosine similarity & TMDB
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='margin-bottom:0.5rem'>🎛️ Controls</h2>",
        unsafe_allow_html=True,
    )

    # ── Search ──────────────────────────────────────────────────────────────
    search_text = st.text_input(
        "🔍 Search movie by name",
        placeholder="e.g. Inception, Avatar…",
        key="search_input",
    )

    if search_text.strip():
        q = search_text.strip().lower()
        filtered_titles = [t for t in all_titles if q in t.lower()]
        if not filtered_titles:
            st.warning("No titles match your search. Showing all movies.")
            filtered_titles = all_titles
    else:
        filtered_titles = all_titles

    # ── Selectbox – ensure active_movie is always a valid option ────────────
    active = st.session_state.active_movie
    if active not in filtered_titles:
        default_idx = 0
    else:
        default_idx = filtered_titles.index(active)

    selected_movie = st.selectbox(
        "🎥 Select a movie",
        options=filtered_titles,
        index=default_idx,
        key="movie_select",
    )
    # Always keep session state in sync with selectbox
    st.session_state.active_movie = selected_movie

    st.markdown("---")

    # ── Surprise Me button ───────────────────────────────────────────────────
    if st.button("🎲 Surprise Me", use_container_width=True):
        rand_pick = random.choice(all_titles)
        st.session_state.active_movie  = rand_pick
        st.session_state.recommendations = None  # clear stale recs
        st.session_state.rec_base       = ""
        st.success(f"🎬 Random pick: **{rand_pick}**")
        st.rerun()

    st.markdown("---")

    # ── Controls ─────────────────────────────────────────────────────────────
    top_n = st.slider(
        "🔢 Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )
    layout_choice = st.radio(
        "📐 Layout",
        options=["🃏 Cards (grid)", "📋 List"],
        index=0,
    )
    cols_per_row = st.slider(
        "📊 Cards per row",
        min_value=2,
        max_value=6,
        value=4,
        step=1,
        disabled=(layout_choice != "🃏 Cards (grid)"),
    )

    st.markdown("---")

    # ── Recommend button ─────────────────────────────────────────────────────
    recommend_clicked = st.button(
        "✨ Get Recommendations",
        use_container_width=True,
        type="primary",
    )

    # ── Dataset stats ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align:center">
            <div class="stat-value">{len(all_titles):,}</div>
            <div class="stat-label">Movies in dataset</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not TMDB_API_KEY:
        st.warning(
            "⚠️ No TMDB API key found.\n\n"
            "Add `TMDB_API_KEY=your_key` to your `.env` file or `st.secrets`.",
            icon="🔑",
        )

# ─────────────────────────────────────────────────────────────────────────────
# SELECTED MOVIE PREVIEW  (always visible)
# ─────────────────────────────────────────────────────────────────────────────
current_title = st.session_state.active_movie

if current_title:
    match_row = movies[movies["title"] == current_title]

    if not match_row.empty:
        movie_id = int(match_row.iloc[0]["movie_id"])
        overview = match_row.iloc[0].get("overview", "") or ""

        with st.spinner("Loading movie details…"):
            details = fetch_tmdb_details(movie_id)

        col_poster, col_details = st.columns([1, 3])
        with col_poster:
            st.markdown(
                f'<img src="{details["poster_url"]}" style="width:100%;border-radius:14px;box-shadow:0 8px 32px rgba(250,204,21,0.2)"/>',
                unsafe_allow_html=True,
            )
        with col_details:
            genre_tags = "".join(
                f'<span class="genre-tag">{g}</span>' for g in details["genres"]
            )
            st.markdown(
                f"""
                <h2 style="margin-bottom:4px">{current_title}</h2>
                <div class="movie-meta">
                    📅 {details['release_date']} &nbsp;·&nbsp;
                    ⭐ {details['vote_average']} / 10
                </div>
                <div style="margin:10px 0">{genre_tags}</div>
                <p style="font-size:0.95rem;line-height:1.7;color:#d1d5db">
                    {overview if overview else "No overview available."}
                </p>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
if recommend_clicked:
    with st.spinner(f"🔍 Finding movies similar to **{current_title}**…"):
        try:
            recs = get_recommendations(current_title, movies, cosine_sim, top_n=top_n)
            st.session_state.recommendations = recs
            st.session_state.rec_base        = current_title
        except ValueError as exc:
            st.error(f"❌ {exc}")
            st.session_state.recommendations = None
        except Exception as exc:
            st.error(f"❌ Unexpected error during recommendation: {exc}")
            st.session_state.recommendations = None

# Display recommendations if available
recs_df: pd.DataFrame | None = st.session_state.get("recommendations")
rec_base: str                 = st.session_state.get("rec_base", "")

if recs_df is not None and not recs_df.empty:
    st.markdown(
        f"<h2>🎯 Because you liked &nbsp;<em>{rec_base}</em></h2>",
        unsafe_allow_html=True,
    )
    st.caption(f"Showing top {len(recs_df)} recommendations · similarity scored 0–100%")
    st.markdown("")

    if layout_choice == "🃏 Cards (grid)":
        # ── Grid layout ──────────────────────────────────────────────────────
        n = cols_per_row
        for row_start in range(0, len(recs_df), n):
            row_slice = recs_df.iloc[row_start : row_start + n]
            cols      = st.columns(n)
            for col, (_, row) in zip(cols, row_slice.iterrows()):
                with col:
                    render_grid_card(row)
            # Add spacing between rows
            st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)
    else:
        # ── List layout ───────────────────────────────────────────────────────
        for _, row in recs_df.iterrows():
            render_list_card(row)

elif not recommend_clicked:
    # Landing state
    st.markdown(
        """
        <div style="text-align:center;padding:60px 20px">
            <div style="font-size:4rem">🍿</div>
            <h3 style="color:#9ca3af;font-weight:400;margin-top:16px">
                Search or pick a movie, then click <strong style="color:#facc15">✨ Get Recommendations</strong>
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# MOVIE STATISTICS SECTION
# ─────────────────────────────────────────────────────────────────────────────
if recs_df is not None and not recs_df.empty:
    st.markdown("---")
    st.markdown("<h2>📊 Recommendation Stats</h2>", unsafe_allow_html=True)

    avg_sim  = round(recs_df["similarity"].mean(), 1)
    max_sim  = round(recs_df["similarity"].max(),  1)
    min_sim  = round(recs_df["similarity"].min(),  1)

    s1, s2, s3, s4 = st.columns(4)
    stats = [
        (s1, f"{len(recs_df)}",  "Movies Recommended"),
        (s2, f"{avg_sim}%",      "Avg. Similarity"),
        (s3, f"{max_sim}%",      "Highest Match"),
        (s4, f"{min_sim}%",      "Lowest Match"),
    ]
    for col, val, label in stats:
        with col:
            st.markdown(
                f"""
                <div class="stat-box">
                    <div class="stat-value">{val}</div>
                    <div class="stat-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
