"""
Movie Recommendation System — CineMatch
========================================
Optimized for Hugging Face Spaces deployment.

This version fixes:
✅ Google Drive download bug
✅ Proper file ID usage
✅ Better error handling
✅ Cleaner loading logic
"""

import random
import os
import pickle
from pathlib import Path
from typing import Optional

import gdown
import pandas as pd
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE DRIVE FILE ID
# Replace with your actual file ID
# Example:
# https://drive.google.com/file/d/1ABCxyz123/view
# File ID = 1ABCxyz123
# ─────────────────────────────────────────────────────────────────────────────
GDRIVE_FILE_ID = "1M77RUKLCPAdIgE7M2JGiNhIAHBqShECz"

PKL_PATH = Path(__file__).parent / "movie_data.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# TMDB API KEY
# Add in Hugging Face Secrets:
# TMDB_API_KEY = your_key
# ─────────────────────────────────────────────────────────────────────────────
def get_api_key() -> Optional[str]:
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return os.getenv("TMDB_API_KEY")

TMDB_API_KEY = get_api_key()

TMDB_BASE = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w500"
FALLBACK_IMG = "https://placehold.co/300x450?text=No+Poster"

# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE DRIVE DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
def download_from_gdrive(file_id: str, dest: Path) -> bool:
    """
    Download movie_data.pkl from Google Drive
    """
    try:
        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(
            url=url,
            output=str(dest),
            quiet=False
        )

        return dest.exists() and dest.stat().st_size > 0

    except Exception as e:
        st.error(f"Download failed: {e}")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MOVIE DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_movie_data(path: str):

    p = Path(path)

    # Download if missing
    if not p.exists():

        if not GDRIVE_FILE_ID:
            st.error("Google Drive File ID missing.")
            st.stop()

        with st.spinner("Downloading movie data..."):

            success = download_from_gdrive(
                GDRIVE_FILE_ID,
                p
            )

        if not success:
            st.error("Could not download movie_data.pkl")
            st.stop()

    # Load pickle
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)

    except Exception as e:
        st.error(f"Pickle load failed: {e}")
        st.stop()

    # Handle tuple format
    if isinstance(data, (list, tuple)) and len(data) == 2:
        movies_df, cosine_sim = data

    # Handle dictionary format
    elif isinstance(data, dict):
        movies_df = data.get("movies")
        cosine_sim = data.get("cosine_sim")

    else:
        raise ValueError(
            "Unexpected pickle format. "
            "Expected tuple or dictionary."
        )

    # Validate columns
    required_cols = {"title", "movie_id"}

    missing = required_cols - set(movies_df.columns)

    if missing:
        raise KeyError(f"Missing columns: {missing}")

    movies_df = movies_df.reset_index(drop=True)

    movies_df["title_lower"] = (
        movies_df["title"]
        .astype(str)
        .str.lower()
    )

    return movies_df, cosine_sim

# ─────────────────────────────────────────────────────────────────────────────
# TMDB DETAILS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_tmdb_details(movie_id: int):

    default = {
        "poster_url": FALLBACK_IMG,
        "release_date": "N/A",
        "vote_average": "N/A",
        "genres": [],
        "overview": ""
    }

    if not TMDB_API_KEY:
        return default

    try:

        url = (
            f"{TMDB_BASE}/movie/{movie_id}"
            f"?api_key={TMDB_API_KEY}"
        )

        response = requests.get(url, timeout=10)

        response.raise_for_status()

        data = response.json()

        poster = data.get("poster_path")

        return {
            "poster_url": (
                f"{POSTER_BASE}{poster}"
                if poster else FALLBACK_IMG
            ),
            "release_date": data.get("release_date", "N/A"),
            "vote_average": round(
                float(data.get("vote_average", 0)),
                1
            ),
            "genres": [
                g["name"]
                for g in data.get("genres", [])
            ],
            "overview": data.get("overview", "")
        }

    except Exception:
        return default

# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def get_recommendations(
    title,
    movies,
    cosine_sim,
    top_n=10
):

    matches = movies[movies["title"] == title]

    if matches.empty:
        raise ValueError(f"{title} not found.")

    idx = matches.index[0]

    sim_scores = list(
        enumerate(cosine_sim[idx])
    )

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:top_n + 1]

    movie_indices = [i[0] for i in sim_scores]

    similarities = [
        round(score * 100, 1)
        for _, score in sim_scores
    ]

    recs = movies.iloc[movie_indices][
        ["title", "movie_id"]
    ].copy()

    recs["similarity"] = similarities

    return recs.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
try:
    movies, cosine_sim = load_movie_data(
        str(PKL_PATH)
    )

except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🎬 CineMatch")

st.write(
    "AI-powered Movie Recommendation System"
)

movie_list = movies["title"].dropna().tolist()

selected_movie = st.selectbox(
    "Select Movie",
    movie_list
)

top_n = st.slider(
    "Number of Recommendations",
    5,
    20,
    10
)

if st.button("✨ Recommend"):

    with st.spinner("Finding recommendations..."):

        try:

            recommendations = get_recommendations(
                selected_movie,
                movies,
                cosine_sim,
                top_n
            )

            cols = st.columns(5)

            for idx, row in recommendations.iterrows():

                movie = row["title"]

                details = fetch_tmdb_details(
                    int(row["movie_id"])
                )

                with cols[idx % 5]:

                    st.image(
                        details["poster_url"],
                        use_container_width=True
                    )

                    st.markdown(
                        f"**{movie}**"
                    )

                    st.caption(
                        f"⭐ Match: {row['similarity']}%"
                    )

        except Exception as e:
            st.error(e)
