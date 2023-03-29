import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carrega o conjunto de dados movielens small
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Pré-processamento dos dados
movies['genres'] = movies['genres'].str.split('|')
movies['genres'] = movies['genres'].fillna("").astype('str')

# Extrai características dos filmes usando TF-IDF
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Calcula a similaridade entre os filmes
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Função de recomendação
def get_recommendations(movie_title, cosine_similarities=cosine_similarities):
    # Obtém o índice do filme que corresponde ao título
    idx = movies.loc[movies['title'] == movie_title].index[0]

    # Obtém as pontuações de similaridade do filme com todos os outros filmes
    sim_scores = list(enumerate(cosine_similarities[idx]))

    # Ordena os filmes com base nas pontuações de similaridade
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtém as 10 recomendações mais semelhantes, podendo variar
    sim_scores = sim_scores[1:11]

    # Obtém os índices dos filmes recomendados
    movie_indices = [i[0] for i in sim_scores]

    # Retorna os títulos dos filmes recomendados
    return movies['title'].iloc[movie_indices]


