import os
import urllib.request
import pickle
import streamlit as st
import requests
import time
import pandas as pd
import numpy as np
import requests
import time
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Constants
MOVIE_LIST_FILE = 'movie_list.pkl'
SIMILARITY_FILE = 'similarity.pkl'
SIMILARITY_URL = "https://www.dropbox.com/scl/fi/8jdlz3c0t1bb20v7o40c3/similarity.pkl?rlkey=lt4q4br6yccvwl886bmy01fl9&st=n2d5cre4&dl=1"
TMDB_API_KEY = "c7385d9faab6ffabaf38b1f824a8b343"

# Download similarity.pkl if not present
def download_similarity_file():
    try:
        urllib.request.urlretrieve(SIMILARITY_URL, SIMILARITY_FILE)
        st.success("similarity.pkl downloaded from Dropbox")
    except Exception as e:
        st.error(f"Error downloading similarity.pkl: {e}")

if not os.path.exists(SIMILARITY_FILE):
    download_similarity_file()

# Load pickle files with error handling
def load_pickle_file(filename, error_message):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"{error_message}: {e}")
        st.stop()

# Load CSV files with error handling
def load_csv_file(filename, error_message):
    try:
        return pd.read_csv(filename)
    except Exception as e:
        st.error(f"{error_message}: {e}")
        st.stop()

movies = load_pickle_file(MOVIE_LIST_FILE, "Could not load movie_list.pkl. Please ensure the file is present.")
similarity = load_pickle_file(SIMILARITY_FILE, "Could not load similarity.pkl. Please ensure the file is present or downloadable.")

# For collaborative filtering
movies_csv = load_csv_file('movies.csv', 'Could not load movies.csv')
ratings_csv = load_csv_file('ratings.csv', 'Could not load ratings.csv')

# Build user-movie matrix for collaborative filtering
final_dataset = ratings_csv.pivot(index="movieId", columns="userId", values="rating")
no_user_voted = ratings_csv.groupby("movieId")['rating'].agg('count')
no_movies_voted = ratings_csv.groupby("userId")['rating'].agg('count')
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]
final_dataset.fillna(0, inplace=True)
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

# Fit KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=11, n_jobs=-1)
knn.fit(csr_data)

# Content-based recommendation (precomputed similarity)
def recommend_content_based(movie):
    if movie not in set(movies['title']):
        st.warning("Selected movie not found in database.")
        return []
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    for i in distances[1:6]:
        movie_title = movies.iloc[i[0]].title
        recommended_movie_names.append(movie_title)
        time.sleep(0.1)
    return recommended_movie_names

# Collaborative filtering recommendation (KNN)
def recommend_collaborative(movie):
    movie_list = movies_csv[movies_csv['title'].str.contains(movie, case=False, na=False)]
    if len(movie_list) == 0:
        st.warning("Movie not found in collaborative filtering database.")
        return []
    movie_id = movie_list.iloc[0]['movieId']
    try:
        movie_idx = final_dataset[final_dataset['movieId'] == movie_id].index[0]
    except IndexError:
        st.warning("Movie not found in collaborative filtering matrix.")
        return []
    distance, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=6)
    rec_movies_indices = sorted(list(zip(indices.squeeze().tolist(), distance.squeeze().tolist())), key=lambda x: x[1])[1:]
    recommended_movie_names = []
    for val in rec_movies_indices:
        rec_movie_id = final_dataset.iloc[val[0]]['movieId']
        idx = movies_csv[movies_csv['movieId'] == rec_movie_id].index
        if len(idx) > 0:
            title = movies_csv.iloc[idx[0]]['title']
            recommended_movie_names.append(title)
        else:
            recommended_movie_names.append("Unknown")
        time.sleep(0.1)
    return recommended_movie_names

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title('🎬 Movie Recommender System')

movie_list = movies['title'].values
selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

rec_type = st.radio("Choose recommendation type:", ("Content-based", "Collaborative filtering"))

if st.button('Show Recommendation'):
    with st.spinner('Finding recommendations...'):
        if rec_type == "Content-based":
            recommended_movie_names = recommend_content_based(selected_movie)
        else:
            recommended_movie_names = recommend_collaborative(selected_movie)
        if recommended_movie_names:
            cols = st.columns(5)
            for idx, col in enumerate(cols):
                col.text(recommended_movie_names[idx])
        else:
            st.info("No recommendations found.") 