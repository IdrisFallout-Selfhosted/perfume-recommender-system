from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, hstack
import pickle

# Create the Flask app
app = Flask(__name__)

# Load the models and data
with open('models/category_mapping.pkl', 'rb') as f:
    category_mapping = pickle.load(f)

with open('models/perfume_data.pkl', 'rb') as f:
    df = pickle.load(f)

with open('models/mlb.pkl', 'rb') as f:
    mlb = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/features_matrix.pkl', 'rb') as f:
    features_matrix = pickle.load(f)

def recommend_perfume_by_note_category_price(note, category, price_range, num_recommendations=5):
    encoded_category = None
    # Check if category is 'All' and set a flag for filtering
    if category.lower() == 'all':
        filter_by_category = False
    else:
        filter_by_category = True
        # Map category to encoded value
        encoded_category = category_mapping.get(category.capitalize())
        if encoded_category is None:
            return "Invalid category specified. Please enter 'Women', 'Men', 'Unisex', or 'All'."

    # Filter perfumes by category if applicable and price range
    min_price, max_price = price_range
    if filter_by_category:
        filtered_df = df[(df['category_encoded'] == encoded_category) &
                         (df['price'] >= min_price) & (df['price'] <= max_price)]
    else:
        filtered_df = df[(df['price'] >= min_price) & (df['price'] <= max_price)]

    # If no perfumes match the criteria, return a message
    if filtered_df.empty:
        return "No perfumes found for the given criteria."

    # Create a query vector for the specified note using the one-hot encoding
    note_vector = pd.Series(0, index=mlb.classes_)
    for n in note.split(', '):
        if n in mlb.classes_:
            note_vector[n] = 1
    note_vector = note_vector.values.reshape(1, -1)

    # Add category (if filtered) and normalized price to the query vector for similarity calculation
    if filter_by_category:
        category_vector = csr_matrix([[encoded_category]])
    else:
        # Set a dummy category vector (0 since we are not filtering by category)
        category_vector = csr_matrix([[0]])

    price_vector = csr_matrix(scaler.transform([[min_price + (max_price - min_price) / 2]]))
    query_vector = hstack([csr_matrix(note_vector), category_vector, price_vector])

    # Calculate similarity scores between the query and filtered perfumes
    sim_scores = cosine_similarity(query_vector, features_matrix[filtered_df.index]).flatten()

    # Get the indices of the top recommendations based on similarity scores
    sim_indices = sim_scores.argsort()[::-1][:num_recommendations]
    recommendations = filtered_df.iloc[sim_indices][['title', 'price', 'category']]

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
        recommendations = recommend_perfume_by_note_category_price(
            user_note, user_category, (user_min_price, user_max_price)
        )

        if isinstance(recommendations, str):  # If it's a string, show error message
            return render_template("index.html", error=recommendations)

        # Return recommendations to the user
        return render_template("index.html", recommendations=recommendations.to_dict(orient="records"))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
