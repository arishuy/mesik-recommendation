# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from utils import preprocess_lyrics, preprocess_input
import json
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
# Mongo connection
client = MongoClient(os.getenv("MONGODB_CONNECTION"))
db = client["mesik"]
collection = db["tokenizedlyrics"]
collection_songs = db["songs"]

# Load the data


def load_data_from_mongo():
    cursor = collection_songs.find()
    mongo_data = list(cursor)
    df = pd.DataFrame(mongo_data)
    df["_id"] = df["_id"].astype(str)
    df["genre"] = df["genre"].astype(str)
    df["artist"] = df["artist"].astype(str)
    df["region"] = df["region"].astype(str)
    df["release_date"] = pd.to_datetime(df["release_date"])
    return df


# Drop rows with missing values (optional, adjust as needed)
# data = data.dropna()


# Function to get content-based recommendations based on music features
def get_content_based_recommendations(song_id, top_n=5):
    data = load_data_from_mongo()
    # Get the index of the target song
    target_song_index = data[data["_id"] == song_id].index[0]
    # Create a CountVectorizer object for genre
    count_genre = CountVectorizer()
    count_matrix_genre = count_genre.fit_transform(data["genre"])

    # Compute the cosine similarity matrix based on genre
    genre_similarity = cosine_similarity(count_matrix_genre)

    # Create a CountVectorizer object for artist
    count_artist = CountVectorizer()
    count_matrix_artist = count_artist.fit_transform(data["artist"])

    # Compute the cosine similarity matrix based on artist
    artist_similarity = cosine_similarity(count_matrix_artist)

    # Create a CountVectorizer object for region
    count_region = CountVectorizer()
    count_matrix_region = count_region.fit_transform(data["region"])

    # Compute the cosine similarity matrix based on region
    region_similarity = cosine_similarity(count_matrix_region)

    # Get the year of the target song
    target_year = data.loc[target_song_index, "release_date"]

    # Get the indices of the top N most similar songs
    combined_similarity = (
        0.3 * genre_similarity
        + 0.3 * artist_similarity
        + 0.2 * region_similarity
        + 0.2 * (target_year == data["release_date"].values[:, None])
    )
    similar_indices = np.argsort(combined_similarity[target_song_index])[::-1][
        1 : top_n + 1
    ]

    # Get the recommended songs
    recommendations = data.iloc[similar_indices]

    return recommendations


def hybrid_recommendations(input_song_id, num_recommendations=5):

    # Get content-based recommendations
    recommendations = get_content_based_recommendations(
        input_song_id, top_n=num_recommendations
    )

    # Sort the recommendations by content-based similarity (cosine similarity)
    recommendations = recommendations.sort_values(
        by=["genre"], ascending=False
    )  # Adjust sorting criteria if needed

    return recommendations


@app.route("/recommend", methods=["GET"])
def recommend_songs():
    input_song_id = request.args.get("song_id")

    if not input_song_id:
        return jsonify({"error": "Please provide a song_id parameter"}), 400

    recommendations = hybrid_recommendations(input_song_id, num_recommendations=5)

    # Convert recommendations to JSON format
    recommendations_json = recommendations[["_id", "title", "genre"]].to_dict(
        orient="records"
    )

    return jsonify(recommendations_json)


@app.route("/refresh", methods=["GET"])
def refresh():
    return jsonify({"message": "Refreshed"})


@app.route("/add_lyrics", methods=["POST"])
def add_lyrics():
    data = request.get_json()
    song_id = data["id"]
    lyrics = data["lyric"]

    processed_lyrics = preprocess_lyrics(lyrics)
    return jsonify({"processed_lyrics": processed_lyrics})


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("query")
    processed_query = preprocess_input(query)
    processed_query_str = " ".join(processed_query)

    results_cursor = collection.find()
    results = list(results_cursor)
    corpus = [song["lyric"] for song in results]
    corpus.append(processed_query_str)

    vectorizer = CountVectorizer().fit_transform(corpus)
    vectors = vectorizer.toarray()

    query_vector = vectors[-1]
    song_vectors = vectors[:-1]

    similarities = cosine_similarity([query_vector], song_vectors).flatten()

    matches = []
    for idx, similarity in enumerate(similarities):
        if similarity > 0.2:
            matches.append(
                {"song_id": str(results[idx]["song"]), "similarity": similarity}
            )
    # Sort matches by similarity score in descending order
    matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)
    if len(matches) == 0:
        return jsonify({"message": "No matches found"}), 404
    return jsonify(matches[0]), 200


if __name__ == "__main__":
    app.run(debug=True)
