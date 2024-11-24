import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/dhanshriii/Anime_recommendation_System/master/anime.csv'
    anime = pd.read_csv(url, encoding='utf8')
    anime['genre'] = anime['genre'].fillna('general')  # Fill missing genres with 'general'
    return anime

# Preprocess and create cosine similarity matrix
@st.cache_data
def create_similarity_matrix(anime):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(anime['genre'])
    cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    anime_index = pd.Series(anime.index, index=anime['name']).drop_duplicates()
    return cosine_sim_matrix, anime_index

# Recommendation function
def get_recommendations(Name, topN, cosine_sim_matrix, anime_index, anime):
    try:
        Name = Name.lower()  # Convert the user input to lowercase for case-insensitive matching
        anime_id = anime_index[Name]
    except KeyError:
        st.error("Anime not found in the dataset. Please try a different title.")
        return None
    
    cosine_scores = list(enumerate(cosine_sim_matrix[anime_id]))
    cosine_scores = sorted(cosine_scores, key=lambda x: x[1], reverse=True)
    cosine_scores_N = cosine_scores[0: topN + 1]
    anime_idx = [i[0] for i in cosine_scores_N]
    anime_scores = [i[1] for i in cosine_scores_N]
    anime_similar_show = pd.DataFrame({
        'name': anime.loc[anime_idx, 'name'],
        'score': anime_scores
    }).reset_index(drop=True)
    return anime_similar_show

# Streamlit interface
st.title("Anime Recommendation System")
st.write("Enter your favorite anime, and we'll recommend similar ones!")

anime = load_data()
cosine_sim_matrix, anime_index = create_similarity_matrix(anime)

anime_name = st.text_input("Enter the name of an anime:", "")
top_n = st.slider("Select the number of recommendations:", 1, 20, 10)

if st.button("Get Recommendations"):
    if anime_name:
        recommendations = get_recommendations(anime_name, top_n, cosine_sim_matrix, anime_index, anime)
        if not recommendations.empty:
            st.write("### Recommended Anime:")
            for i, row in recommendations.iterrows():
                st.write(f"**{i + 1}. {row['name']}** (Score: {row['score']:.2f})")
        else:
            st.warning("No recommendations found.")
    else:
        st.error("Please enter an anime name.")
