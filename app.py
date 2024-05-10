from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the data
data = pd.read_csv("mesik.songs.csv")  # Replace with your filename

# Drop rows with missing values (optional, adjust as needed)
# data = data.dropna()


# Function to get content-based recommendations based on music features
def get_content_based_recommendations(song_id, top_n=5):
    # Get the features of the target song
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


def hybrid_recommendations(input_song_name, num_recommendations=5):
    # Get the song ID of the input song
    input_song_id = data[data["title"] == input_song_name]["_id"].values[0]

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
    input_song_name = request.args.get("song_name")

    if not input_song_name:
        return jsonify({"error": "Please provide a song_name parameter"}), 400

    recommendations = hybrid_recommendations(input_song_name, num_recommendations=5)

    # Convert recommendations to JSON format
    recommendations_json = recommendations[["_id", "title", "genre"]].to_dict(
        orient="records"
    )

    return jsonify(recommendations_json)


if __name__ == "__main__":
    app.run(debug=True)
