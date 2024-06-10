# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
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


def date_similarity(dates, target_date):
    # Convert dates to numerical values (e.g., days since the epoch)
    days_since_epoch = (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta("1D")
    target_days_since_epoch = (
        target_date - pd.Timestamp("1970-01-01")
    ) // pd.Timedelta("1D")

    # Calculate the absolute differences
    date_diff = np.abs(days_since_epoch - target_days_since_epoch)

    # Invert the differences to get similarity (higher difference means lower similarity)
    max_diff = date_diff.max()
    date_similarity = 1 - (date_diff / max_diff)

    return date_similarity


# Function to get content-based recommendations based on music features
def get_content_based_recommendations(song_id, top_n=10):
    data = load_data_from_mongo()
    # Get the index of the target song
    target_song_index = data[data["_id"] == song_id].index[0]

    # Create a TfidfVectorizer object for genre
    tfidf_genre = TfidfVectorizer()
    tfidf_matrix_genre = tfidf_genre.fit_transform(data["genre"])
    genre_similarity = cosine_similarity(tfidf_matrix_genre)

    # Create a TfidfVectorizer object for artist
    tfidf_artist = TfidfVectorizer()
    tfidf_matrix_artist = tfidf_artist.fit_transform(data["artist"])
    artist_similarity = cosine_similarity(tfidf_matrix_artist)

    # Create a TfidfVectorizer object for region
    tfidf_region = TfidfVectorizer()
    tfidf_matrix_region = tfidf_region.fit_transform(data["region"])
    region_similarity = cosine_similarity(tfidf_matrix_region)

    # Scale the similarities
    scaler = StandardScaler()
    genre_similarity = scaler.fit_transform(genre_similarity)
    artist_similarity = scaler.fit_transform(artist_similarity)
    region_similarity = scaler.fit_transform(region_similarity)
    # Calculate date similarity
    target_date = data.loc[target_song_index, "release_date"]
    date_sim = date_similarity(data["release_date"], target_date)

    # Calculate the combined similarity
    combined_similarity = (
        0.5 * genre_similarity[target_song_index]
        + 0.2 * artist_similarity[target_song_index]
        + 0.2 * region_similarity[target_song_index]
        + 0.1 * date_sim
    )

    # Get the indices of the top N most similar songs
    similar_indices = np.argsort(combined_similarity)[::-1][1 : top_n + 1]

    # Get the recommended songs
    recommendations = data.iloc[similar_indices]

    return recommendations


def hybrid_recommendations(input_song_id, num_recommendations=5):
    # Get content-based recommendations
    recommendations = get_content_based_recommendations(
        input_song_id, top_n=num_recommendations
    )

    # Sort the recommendations by a relevant feature
    recommendations = recommendations.sort_values(
        by=["genre"], ascending=False
    )  # Adjust sorting criteria if needed

    return recommendations


@app.route("/recommend", methods=["GET"])
def recommend_songs():
    input_song_id = request.args.get("song_id")

    if not input_song_id:
        return jsonify({"error": "Please provide a song_id parameter"}), 400

    recommendations = hybrid_recommendations(input_song_id, num_recommendations=10)

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
        return jsonify({"message": "No matches found"}), 200
    return jsonify(matches[0]), 200


if __name__ == "__main__":
    app.run(debug=True)
