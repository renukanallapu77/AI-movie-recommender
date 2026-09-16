import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender Day 11", page_icon="🎬")
st.title("🎬 Day 11 - AI Movie Recommender")
st.write("Like Netflix & Amazon Prime!")

# Movie Data
data = {
    'title': ['Avengers', 'Iron Man', 'Batman', 'Spider-Man', 'Superman', 'Thor', 'Hulk', 'Captain America'],
    'genre': ['action superhero marvel', 'action superhero marvel', 'action superhero dc', 'action superhero marvel', 'action superhero dc', 'action superhero marvel', 'action superhero marvel', 'action superhero marvel']
}
df = pd.DataFrame(data)

# Train
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(df['genre'])
similarity = cosine_similarity(genre_matrix)

st.success("Trained on 8 Superhero Movies!")

movie = st.selectbox("Choose a movie you like:", df['title'].tolist())

if st.button("🎯 Recommend Similar"):
    idx = df[df['title'] == movie].index[0]
    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    st.write(f"### Because you liked **{movie}**, you will like:")

    for i in range(1, 4):
        rec_idx = sorted_scores[i][0]
        rec_movie = df.iloc[rec_idx]['title']
        score = sorted_scores[i][1] * 100
        st.success(f"🎬 {rec_movie} - {score:.0f}% Match")

    st.balloons()

st.caption("Day 11 - Recommendation System | 30 Days AI Challenge")
