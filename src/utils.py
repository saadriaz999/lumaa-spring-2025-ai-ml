# Directory: data_loader.py
import pandas as pd

def load_movie_data(file_path):
    """load movie dataset from a CSV file."""
    df = pd.read_csv(file_path)
    required_columns = ['title', 'keywords', 'overview']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Dataset must contain 'title', 'keywords', and 'overview' columns.")
    return df[required_columns].fillna('')

# Directory: recommender.py
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_recommendations(user_input, vectorizer, tfidf_matrix, movie_titles, top_n=5):
    """compute cosine similarity and return top N movie recommendations."""
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    recommendations = [(movie_titles[i], similarities[i]) for i in top_indices]
    return recommendations

# Directory: vectorizer.py
from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_matrix(descriptions):
    """convert text descriptions to TF-IDF vectors."""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    return tfidf_matrix, vectorizer

# Directory: inference.py
def run_recommendation_system(query, data_path):
    data = load_movie_data(data_path)
    # combine keywords and overview for better recommendations
    data['content'] = data['keywords'] + ' ' + data['overview']
    tfidf_matrix, vectorizer = create_tfidf_matrix(data['content'])
    recommendations = get_recommendations(query, vectorizer, tfidf_matrix, data['title'])
    return recommendations