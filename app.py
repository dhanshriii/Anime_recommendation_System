import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset from the provided URL
@st.cache_data
def load_dataset():
    url = "https://raw.githubusercontent.com/dhanshriii/Anime_recommendation_System/master/anime.csv"
    data = pd.read_csv(url)
    return data

# Function to preprocess and compute similarity matrix
@st.cache_data
def compute_similarity(data, column):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(data[column].fillna(""))
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix

# Function to get recommendations
def recommend_anime(anime_name, data, similarity_matrix, top_n=5):
    if anime_name not in data["name"].values:
        st.warning("Anime not found in the dataset. Please try another name.")
        return pd.DataFrame()

    # Find the index of the anime
    anime_index = data[data["name"] == anime_name].index[0]
    similarity_scores = list(enumerate(similarity_matrix[anime_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1 : top_n + 1]  # Exclude the anime itself

    # Fetch recommendations
    recommended_indices = [i[0] for i in similarity_scores]
    recommendations = data.iloc[recommended_indices]
    recommendations = recommendations.assign(score=[i[1] for i in similarity_scores])
    return recommendations

# Streamlit app
st.title("Anime Recommendation System")

# Load dataset
anime_data = load_dataset()
if anime_data is not None:
    st.write("### Dataset Loaded Successfully!")
    st.write(anime_data.head())  # Display the first few rows of the dataset
    
    # Select feature column for recommendations
    feature_column = st.selectbox("Select a feature for recommendations:", anime_data.columns)
    
    if st.button("Compute Similarity"):
        similarity_matrix = compute_similarity(anime_data, feature_column)
        st.success("Similarity matrix computed successfully!")

        # User input
        anime_name = st.text_input("Enter an anime name for recommendations:")
        top_n = st.slider("Number of recommendations:", min_value=1, max_value=10, value=5)

        if st.button("Get Recommendations"):
            if anime_name:
                recommendations = recommend_anime(anime_name, anime_data, similarity_matrix, top_n)
                if not recommendations.empty:
                    st.write("### Recommended Anime:")
                    st.table(recommendations[["name", "genre", "score"]])
            else:
                st.warning("Please enter an anime name.")
