import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ,Book,Author,Description,Genres,Avg_Rating,Num_Ratings,URL
books_data = pd.read_csv("./data/goodreads_data.csv")

books_data.dropna(inplace=True)

books_data['combined'] = books_data['Book'] + " " + \
    books_data['Author'] + " " + books_data['Genres']


vectorizer = TfidfVectorizer(stop_words='english')

tfidf = vectorizer.fit_transform(books_data['combined'])

cosine_sim = cosine_similarity(tfidf, tfidf)

joblib.dump(cosine_sim, 'model_rekomendasi_buku.joblib')
