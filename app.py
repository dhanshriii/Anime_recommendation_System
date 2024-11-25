import streamlit as st
import pandas as pd

# Mock function to simulate recommendation logic
def get_recommendations(anime_name, top_n, similarity_matrix):
    try:
        # Example placeholder for real recommendation logic
        if anime_name not in similarity_matrix.index:
            return pd.DataFrame()  # Return an empty DataFrame if anime not found
        similarity_scores = similarity_matrix[anime_name].sort_values(ascending=False)
        top_anime = similarity_scores.iloc[1:top_n + 1]  # Exclude the queried anime itself
        recommendations = pd.DataFrame({
            'name': top_anime.index,
            'score': top_anime.values
        })
        return recommendations
    except Exception as e:
        st.error(f"Error in getting recommendations: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame on failure

# Example similarity matrix for testing
similarity_matrix = pd.DataFrame(
    {
        'Naruto': [1, 0.9, 0.7],
        'Bleach': [0.9, 1, 0.6],
        'One Piece': [0.7, 0.6, 1]
    },
    index=['Naruto', 'Bleach', 'One Piece']
)

# Streamlit app layout
st.title("Anime Recommendation System")

anime_name = st.text_input("Enter the name of an anime:")
top_n = st.number_input("How many recommendations do you want?", min_value=1, max_value=10, value=5)

if st.button("Get Recommendations"):
    if anime_name:
        recommendations = get_recommendations(anime_name, top_n, similarity_matrix)
        if not recommendations.empty:
            st.write("### Recommended Anime:")
            for i, row in recommendations.iterrows():
                st.write(f"**{i + 1}. {row['name']}** (Score: {row['score']:.2f})")
        else:
            st.warning("No recommendations found. Please check the anime name or try another.")
    else:
        st.warning("Please enter an anime name.")
