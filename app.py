# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

app = Flask(__name__)


# Load the model and data
def load_model():
    # Load preprocessed data and models
    with open('models/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)

    with open('models/similarity_matrix.pkl', 'rb') as f:
        similarity_matrix = pickle.load(f)

    with open('models/mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)

    with open('models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return data, similarity_matrix, mlb, label_encoder, scaler


# Initialize global variables
data, similarity_matrix, mlb, label_encoder, scaler = load_model()


def get_recommendations(perfume_name, n_recommendations=5):
    try:
        # Find the index of the perfume
        idx = data[data['title'].str.lower() == perfume_name.lower()].index[0]

        # Get similarity scores
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top N similar perfumes
        top_indices = [i[0] for i in sim_scores[1:n_recommendations + 1]]

        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'title': data.iloc[idx]['title'],
                'category': data.iloc[idx]['category'],
                'price': float(data.iloc[idx]['price']),
                'top_notes': data.iloc[idx]['top'],
                'middle_notes': data.iloc[idx]['middle'],
                'base_notes': data.iloc[idx]['base'],
                'similarity_score': float(sim_scores[idx][1])
            })

        return recommendations

    except IndexError:
        return None
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        return None


# Routes
@app.route('/')
def home():
    # Get unique perfume titles for dropdown
    perfume_titles = sorted(data['title'].unique())
    return render_template('index.html', perfume_titles=perfume_titles)


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        content = request.json
        perfume_name = content['perfume_name']
        n_recommendations = content.get('n_recommendations', 5)

        recommendations = get_recommendations(perfume_name, n_recommendations)

        if recommendations is None:
            return jsonify({'error': 'Perfume not found or error in processing'}), 404

        return jsonify({'recommendations': recommendations})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/search', methods=['GET'])
def search_perfumes():
    query = request.args.get('q', '').lower()
    matching_perfumes = [title for title in data['title'].unique()
                         if query in title.lower()]
    return jsonify(matching_perfumes)


@app.route('/filter_recommendations', methods=['POST'])
def filter_recommendations():
    try:
        content = request.json
        category = content.get('category')
        min_price = content.get('min_price')
        max_price = content.get('max_price')

        filtered_data = data.copy()

        if category and category != 'All':
            filtered_data = filtered_data[filtered_data['category'] == category]

        if min_price is not None:
            filtered_data = filtered_data[filtered_data['price'] >= min_price]

        if max_price is not None:
            filtered_data = filtered_data[filtered_data['price'] <= max_price]

        return jsonify({
            'perfumes': filtered_data[['title', 'category', 'price']].to_dict('records')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)