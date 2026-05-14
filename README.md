# CineMatch - Movie Recommendation System 🎬

A professional, Netflix-inspired movie recommendation web application built with Python and Streamlit. It uses Machine Learning (Cosine Similarity) to recommend movies and fetches rich metadata (posters, ratings, genres, overviews) in real-time from the TMDB API.

## Features ✨

- **Smart Recommendations:** Uses content-based filtering (cosine similarity) to suggest movies you'll love.
- **Rich Metadata:** Displays high-quality movie posters, TMDB ratings, release dates, genres, and overviews.
- **Dynamic UI:** A stunning, responsive, dark-themed Netflix-like UI with hover effects and animations.
- **View Toggles:** Switch seamlessly between an interactive Grid View (Cards) and a detailed List View.
- **Surprise Me:** Can't decide? Let the app pick a random movie for you!
- **Fast & Reliable:** Utilizes `st.cache_data` for lightning-fast dataset loading and robust API request caching.
- **Error Handling:** Built-in fallbacks for missing images, timeout handling for the TMDB API, and graceful error messages.

## Tech Stack 🛠️

- **Python 3.8+**
- **Streamlit** (Frontend framework & state management)
- **Pandas** (Data manipulation)
- **Requests** (API communication)
- **Pickle** (Loading pre-computed similarity matrices)
- **python-dotenv** (Environment variable management)
- **TMDB API** (Movie metadata source)

## Setup & Installation 🚀

1. **Clone the repository:**
   Ensure you have the project files in your working directory.

2. **Install the dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Get a TMDB API Key:**
   - Create an account at [The Movie Database (TMDB)](https://www.themoviedb.org/).
   - Go to your account settings and generate an API key (v3 auth).

4. **Set up Environment Variables:**
   - Open the `.env` file in the root directory.
   - Replace `your_tmdb_api_key_here` with your actual API key:
     ```env
     TMDB_API_KEY=1234567890abcdef1234567890abcdef
     ```
   - *Alternatively, you can use Streamlit secrets by creating a `.streamlit/secrets.toml` file.*

5. **Ensure Data Files are Present:**
   Make sure you have the `movie_data.pkl` file in the same directory as `app.py`. This file must contain the pre-processed movie dataframe and the cosine similarity matrix.

6. **Run the App:**
   ```bash
   streamlit run app.py
   ```
   The application will automatically open in your default web browser.

## File Structure 📂

```text
project/
│
├── app.py              # Main Streamlit application logic
├── movie_data.pkl      # Serialized dataframe and similarity matrix
├── .env                # Environment variables (API keys)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Troubleshooting 💡

- **Missing Poster/Data:** If posters aren't loading, check if your TMDB API key is correct and ensure you have an active internet connection.
- **File Not Found Error:** If the app complains about `movie_data.pkl`, verify that the file is exactly named that and placed in the same folder as `app.py`.
- **Session State Warnings:** If recommendations disappear randomly, ensure you aren't manually reloading the browser page (use the app's controls to navigate).
