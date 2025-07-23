# Movie Recommendation System

A web-based movie recommender app built with Streamlit. This app provides both content-based and collaborative filtering recommendations using movie and ratings data.

## Features
- Content-based filtering (using precomputed similarity matrix)
- Collaborative filtering (using KNN on user ratings)
- Simple, clean UI
- No external API required for recommendations

## Setup Instructions
1. **Clone the repository and navigate to the project folder:**
   ```bash
   git clone https://github.com/Sunay4826/movie-recommendation-system.git
   cd movie-recommendation-system
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure the following data files are present in the project folder:**
   - `movies.csv`
   - `ratings.csv`
   - `movie_list.pkl`
   - `similarity.pkl`

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Usage
- Select a movie from the dropdown.
- Choose the recommendation type (Content-based or Collaborative filtering).
- Click "Show Recommendation" to see a list of recommended movies.

## Credits
Built by Sunay4826 