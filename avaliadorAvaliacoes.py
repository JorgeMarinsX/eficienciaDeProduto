import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

# Baixar stopwords do NLTK (uma vez, se necessário)
nltk.download("stopwords")

# Stopwords em português
stop_words_pt = stopwords.words("portuguese")

def processar_avaliacoes(arquivo_avaliacoes, ngram_range=(1, 2), max_features=30):
    # Carregar o arquivo de avaliações
    df = pd.read_csv(arquivo_avaliacoes)
    
    # Separar avaliações positivas e negativas
    positivas = df[df["Estrelas"] >= 4]["Avaliação"]
    negativas = df[df["Estrelas"] <= 3]["Avaliação"]
    
    # Criar um vetorizador para n-grams com stopwords em português
    vectorizer = CountVectorizer(stop_words=stop_words_pt, ngram_range=ngram_range, max_features=max_features)
    
    # Processar avaliações positivas
    positivas_vectorizadas = vectorizer.fit_transform(positivas)
    positivas_palavras = dict(zip(vectorizer.get_feature_names_out(), positivas_vectorizadas.sum(axis=0).A1))
    
    # Processar avaliações negativas
    negativas_vectorizadas = vectorizer.fit_transform(negativas)
    negativas_palavras = dict(zip(vectorizer.get_feature_names_out(), negativas_vectorizadas.sum(axis=0).A1))
    
    # Retornar contagens de palavras
    return positivas_palavras, negativas_palavras

# Caminho do arquivo de avaliações
arquivo_avaliacoes = "df/avaliacoes_exemplo.csv"

# Processar avaliações e imprimir resultados
positivas, negativas = processar_avaliacoes(arquivo_avaliacoes)

print("\nPalavras mais frequentes em avaliações positivas:")
print(positivas)

print("\nPalavras mais frequentes em avaliações negativas:")
print(negativas)
