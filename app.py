import os
from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, hstack
import pickle

# Ensure the 'models' directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# Create the Flask app
app = Flask(__name__)

def load_pickled_objects(folder="models"):
    loaded_objects = {}
    for obj_name in ["tfidf_vectorizer", "scaler", "category_mapping", "features_matrix", "df"]:
        with open(os.path.join(folder, f"{obj_name}.pkl"), "rb") as f:
            loaded_objects[obj_name] = pickle.load(f)
    return loaded_objects

# Load pickled objects from models/ folder
pickled_objects = load_pickled_objects(folder="models")
tfidf_vectorizer = pickled_objects["tfidf_vectorizer"]
scaler = pickled_objects["scaler"]
category_mapping = pickled_objects["category_mapping"]
features_matrix = pickled_objects["features_matrix"]
df = pickled_objects["df"]

def recommend_perfume_by_note_gender_price(note, gender, price_range, model_type="cosine", num_recommendations=5):
    # Map gender to encoded value or handle "ALL"
    if gender.capitalize() == "All":
        encoded_gender = None  # Indicate that all genders are to be included
    else:
        encoded_gender = category_mapping.get(gender.capitalize())
        if encoded_gender is None:
            return "Invalid gender specified. Please enter 'Women', 'Men', 'Unisex', or 'All'."

    min_price, max_price = price_range

    # Filter by price and optionally by gender
    if encoded_gender is not None:
        filtered_df = df[(df['category_encoded'] == encoded_gender) &
                         (df['price'] >= min_price) & (df['price'] <= max_price)]
    else:
        filtered_df = df[(df['price'] >= min_price) & (df['price'] <= max_price)]

    if filtered_df.empty:
        return "No perfumes found for the given criteria."

    # Create a query vector for the specified note using TF-IDF vectorizer
    query_note_vector = tfidf_vectorizer.transform([note])

    # Add gender (set to 0 for "ALL") and normalized price to the query vector
    gender_vector = csr_matrix([[encoded_gender if encoded_gender is not None else 0]])
    price_vector = csr_matrix(scaler.transform([[min_price + (max_price - min_price) / 2]]))  # Midpoint of price range
    query_vector = hstack([query_note_vector, gender_vector, price_vector])

    # Ensure the query vector has the same number of features as the feature matrix
    if query_vector.shape[1] != features_matrix.shape[1]:
        # Fill missing features in the query vector with zeros if necessary
        missing_columns = features_matrix.shape[1] - query_vector.shape[1]
        query_vector = hstack([query_vector, csr_matrix([[0] * missing_columns])])

    # Calculate similarity scores based on the chosen model
    if model_type == "cosine":
        sim_scores = cosine_similarity(query_vector, features_matrix[filtered_df.index, :]).flatten()

    # Get indices of top recommendations
    sim_indices = sim_scores.argsort()[::-1][:num_recommendations]
    recommendations = filtered_df.iloc[sim_indices][['title', 'price', 'category', 'image', 'link']]

    return recommendations

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get data from form
        user_note = request.form["note"]
        user_category = request.form["category-filter"]  # Changed from 'gender' to 'category-filter'
        user_min_price = float(request.form["min_price"])
        user_max_price = float(request.form["max_price"])

        # Call the recommendation function
        recommendations = recommend_perfume_by_note_gender_price(
            user_note, user_category, (user_min_price, user_max_price)
        )

        if isinstance(recommendations, str):  # If it's a string, show error message
            return render_template("index.html", error=recommendations)

        # Return recommendations to the user
        return render_template("index.html", recommendations=recommendations.to_dict(orient="records"))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
