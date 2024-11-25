import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Function to preprocess and compute similarity matrix
@st.cache_data
def compute_similarity(data, feature_column):
    try:
        # Combine features into a single string for vectorization (adjust as needed)
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(data[feature_column].fillna(''))
        similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return similarity
    except Exception as e:
        st.error(f"Error in computing similarity matrix: {e}")
        return None

# Function to get recommendations
def get_recommendations(anime_name, data, similarity_matrix, top_n=5):
    try:
        if anime_name not in data['name'].values:
            return pd.DataFrame()  # Return empty if anime not found
        
        idx = data[data['name'] == anime_name].index[0]  # Get index of the anime
        similarity_scores = list(enumerate(similarity_matrix[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)  # Sort by score
        similarity_scores = similarity_scores[1:top_n + 1]  # Exclude the queried anime itself
        
        recommended_indices = [i[0] for i in similarity_scores]
        recommendations = data.iloc[recommended_indices].copy()
        recommendations['score'] = [i[1] for i in similarity_scores]
        return recommendations
    except Exception as e:
        st.error(f"Error in getting recommendations: {e}")
        return pd.DataFrame()

# Main Streamlit app
st.title("Anime Recommendation System")

# Load the dataset
dataset_path = st.text_input("Enter the path to your anime dataset CSV file:")
if dataset_path:
    anime_data = load_data(dataset_path)
    
    if anime_data is not None:
        st.write("### Dataset Loaded Successfully!")
        st.write(anime_data.head())  # Display first few rows of the dataset
        
        # Select feature column to compute similarity
        feature_column = st.selectbox("Select a feature column for recommendations:", anime_data.columns)
        
        if st.button("Compute Similarity"):
            similarity_matrix = compute_similarity(anime_data, feature_column)
            if similarity_matrix is not None:
                st.success("Similarity matrix computed successfully!")
                
                # Recommendation input
                anime_name = st.text_input("Enter the name of an anime for recommendations:")
                top_n = st.number_input("How many recommendations do you want?", min_value=1, max_value=10, value=5)
                
                if st.button("Get Recommendations"):
                    if anime_name:
                        recommendations = get_recommendations(anime_name, anime_data, similarity_matrix, top_n)
                        if not recommendations.empty:
                            st.write("### Recommended Anime:")
                            for i, row in recommendations.iterrows():
                                st.write(f"**{i + 1}. {row['name']}** (Score: {row['score']:.2f})")
                        else:
                            st.warning("No recommendations found. Please check the anime name or try another.")
                    else:
                        st.warning("Please enter an anime name.")
