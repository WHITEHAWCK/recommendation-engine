import random
import re
import streamlit as st
from surprise import Dataset, Reader, KNNBasic, SVD, KNNBaseline
import os
import pickle
import pandas as pd
from rapidfuzz import process

n_rows = 2500000
# skip_rows = sorted(random.sample(range(1, 1000000), 1000000 - n_rows))
df = pd.read_csv('./ml-25m/ratings.csv', nrows=n_rows)#, skiprows=skip_rows)

# for titles of movies
meta_df = pd.read_csv('ml-25m/movies.csv')
meta_df.drop('genres', axis=1, inplace=True)
meta_df['title'] = meta_df['title'].apply(lambda x: re.sub(r'\(\d+\)', '', x))
meta_df['title'] = meta_df['title'].apply(str.strip)

final_df = df.merge(meta_df, on='movieId')

reader = Reader(rating_scale = (1, 5))
data = Dataset.load_from_df(final_df[['userId', 'movieId', 'rating']], reader=reader)


def train_model_and_save():
    sim_options = {"name": "pearson_baseline", "user_based": False}
    algo = KNNBaseline(sim_options=sim_options)
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    with open("trained_model.pkl", "wb") as f:
        pickle.dump(algo, f)


def load_model():
    if os.path.exists("trained_model.pkl"):
        with open("trained_model.pkl", "rb") as f:
            return pickle.load(f)
    else:
        train_model_and_save()
        return load_model()


def get_movie_id(movie_title):
    matches = process.extractOne(movie_title.lower(), final_df['title'].str.lower())
    if matches[1] >= 90:  # Check if the similarity score is above a threshold (e.g., 90)
        similar_title = matches[0]
        filtered_df = final_df[final_df['title'].str.lower() == similar_title.lower()]
        if not filtered_df.empty:
            return filtered_df.iloc[0]['movieId']
        else:
            return None
    else:
        return None


def get_recommendations(movie_title, n_rec, algo):
    movie_id = get_movie_id(movie_title)
    if movie_id is not None:
        movie_neighbors = algo.get_neighbors(algo.trainset.to_inner_iid(movie_id), k=n_rec)
        movie_titles = []
        try:
            for movie_id in movie_neighbors:
                movie_titles.append(final_df[final_df['movieId'] == movie_id]['title'].iloc[0])
        except:
            movie_titles.append("REACHED END! (data used is small subset of total data)")
    else:
        movie_titles = ['Nothing found! Possibly due to small sample of data used in training']
    return movie_titles


st.title('Movie Recommendation App')

movie_input = st.text_input('Enter a movie you like:')
num_recommendations = st.slider('Number of Recommendations', min_value=1, max_value=20, value=10)

algo = load_model()

if st.button('Get Recommendations'):
    if movie_input:
        recommendations = get_recommendations(movie_input, num_recommendations, algo)
        st.write(f"Recommended movies for someone who likes **{movie_input}** :")
        for movie in recommendations:
            st.write(f"\u2022 {movie}")
    else:
        st.write("Please enter a movie.")
