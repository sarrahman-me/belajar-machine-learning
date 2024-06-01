import joblib
import pandas as pd

cosine_sim = joblib.load('./model_rekomendasi_buku.joblib')

books_data = pd.read_csv("./data/goodreads_data.csv")

# Membuat seri untuk memetakan judul buku ke indeks
indices = pd.Series(
    books_data.index, index=books_data['Book']).drop_duplicates()


def recommend_books(title, cosine_sim=cosine_sim):
    # Mendapatkan indeks buku yang diberikan dari judul
    idx = indices[title]

    # Mendapatkan skor kesamaan untuk semua buku dengan buku yang diberikan
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Mengurutkan buku berdasarkan skor kesamaan
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Mendapatkan skor dari 10 buku yang paling mirip
    sim_scores = sim_scores[1:11]

    # Mendapatkan indeks buku dari buku yang paling mirip
    book_indices = [i[0] for i in sim_scores]

    # Mengembalikan judul buku yang paling mirip
    return books_data['Book'].iloc[book_indices]


# Contoh penggunaan:
print(recommend_books('The Call of the Wild'))
