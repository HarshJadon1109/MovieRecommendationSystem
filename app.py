"""
Movie Recommendation System — CineMatch
Fully Fixed Version
"""

import os
import random
import pickle
from pathlib import Path
from typing import Optional

import gdown
import pandas as pd
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

GDRIVE_FILE_ID = "1M77RUKLCPAdIgE7M2JGiNhIAHBqShECz"

PKL_PATH = Path("movie_data.pkl")

TMDB_BASE = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w500"

# Better fallback image
FALLBACK_IMG = (
    "https://via.placeholder.com/300x450.png?text=No+Poster"
)

# ─────────────────────────────────────────────────────────────
# STREAMLIT PAGE
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# GET TMDB API KEY
# ─────────────────────────────────────────────────────────────

def get_api_key() -> Optional[str]:

    # Streamlit secrets
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        pass

    # Environment variable
    return os.getenv("TMDB_API_KEY")


TMDB_API_KEY = get_api_key()

# ─────────────────────────────────────────────────────────────
# DOWNLOAD PICKLE FILE
# ─────────────────────────────────────────────────────────────

def download_from_gdrive(file_id: str, dest: Path):

    try:

        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(
            url=url,
            output=str(dest),
            quiet=False
        )

        return dest.exists()

    except Exception as e:

        st.error(f"Download Error: {e}")
        return False

# ─────────────────────────────────────────────────────────────
# LOAD MOVIE DATA
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_movie_data():

    # Download if file missing
    if not PKL_PATH.exists():

        with st.spinner("Downloading movie dataset..."):

            success = download_from_gdrive(
                GDRIVE_FILE_ID,
                PKL_PATH
            )

        if not success:
            st.stop()

    # Load pickle
    try:

        with open(PKL_PATH, "rb") as f:
            data = pickle.load(f)

    except Exception as e:

        st.error(f"Pickle Load Error: {e}")
        st.stop()

    # Handle tuple
    if isinstance(data, (list, tuple)):

        movies_df, cosine_sim = data

    # Handle dictionary
    elif isinstance(data, dict):

        movies_df = data["movies"]
        cosine_sim = data["cosine_sim"]

    else:

        st.error("Invalid pickle format")
        st.stop()

    # Validation
    required = {"title", "movie_id"}

    if not required.issubset(movies_df.columns):

        st.error(
            f"Missing columns: "
            f"{required - set(movies_df.columns)}"
        )

        st.stop()

    movies_df = movies_df.reset_index(drop=True)

    return movies_df, cosine_sim

# ─────────────────────────────────────────────────────────────
# FETCH TMDB DETAILS
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_tmdb_details(movie_id):

    default = {
        "poster_url": FALLBACK_IMG,
        "release_date": "N/A",
        "vote_average": "N/A",
        "overview": "No overview available."
    }

    # Missing API key
    if not TMDB_API_KEY:
        return default

    try:

        url = f"{TMDB_BASE}/movie/{movie_id}"

        response = requests.get(
            url,
            params={"api_key": TMDB_API_KEY},
            timeout=10
        )

        # Invalid response
        if response.status_code != 200:
            return default

        data = response.json()

        poster_path = data.get("poster_path")

        poster_url = (
            f"{POSTER_BASE}{poster_path}"
            if poster_path
            else FALLBACK_IMG
        )

        return {
            "poster_url": poster_url,
            "release_date": (
                data.get("release_date")
                or "N/A"
            ),
            "vote_average": round(
                float(data.get("vote_average", 0)),
                1
            ),
            "overview": (
                data.get("overview")
                or "No overview available."
            )
        }

    except Exception:

        return default

# ─────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────

def get_recommendations(
    title,
    movies,
    cosine_sim,
    top_n=10
):

    matches = movies[movies["title"] == title]

    if matches.empty:

        raise ValueError(
            f"{title} not found in dataset."
        )

    idx = matches.index[0]

    similarity_scores = list(
        enumerate(cosine_sim[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[
        1:top_n + 1
    ]

    movie_indices = [
        i[0]
        for i in similarity_scores
    ]

    similarity_percentages = [
        round(score * 100, 1)
        for _, score in similarity_scores
    ]

    recommendations = movies.iloc[
        movie_indices
    ][["title", "movie_id"]].copy()

    recommendations[
        "similarity"
    ] = similarity_percentages

    return recommendations.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

movies, cosine_sim = load_movie_data()

# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────

st.title("🎬 CineMatch")

st.write(
    "AI-Powered Movie Recommendation System"
)

# TMDB status
if TMDB_API_KEY:
    st.success("✅ TMDB API Connected")
else:
    st.warning(
        "⚠️ TMDB API Key Missing "
        "(Posters won't load)"
    )

movie_titles = movies["title"].dropna().tolist()

selected_movie = st.selectbox(
    "Choose a Movie",
    movie_titles
)

top_n = st.slider(
    "Recommendations",
    5,
    20,
    10
)

if st.button("✨ Recommend"):

    with st.spinner("Generating recommendations..."):

        try:

            recommendations = get_recommendations(
                selected_movie,
                movies,
                cosine_sim,
                top_n
            )

            cols = st.columns(5)

            for idx, row in recommendations.iterrows():

                details = fetch_tmdb_details(
                    int(row["movie_id"])
                )

                with cols[idx % 5]:

                    st.image(
                        details["poster_url"],
                        use_container_width=True
                    )

                    st.markdown(
                        f"### {row['title']}"
                    )

                    st.caption(
                        f"⭐ Match: "
                        f"{row['similarity']}%"
                    )

                    st.caption(
                        f"📅 {details['release_date']}"
                    )

        except Exception as e:

            st.error(f"Error: {e}")
