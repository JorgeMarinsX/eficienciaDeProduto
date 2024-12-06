import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

# Baixar stopwords do NLTK (uma vez, se necessário)
nltk.download("stopwords")

# Stopwords em português
stop_words_pt = stopwords.words("portuguese")

def processar_texto(texto, ngram_range=(1, 2), max_features=30):
    # Criar vetorizador para n-grams
    vectorizer = CountVectorizer(stop_words=stop_words_pt, ngram_range=ngram_range, max_features=max_features)
    texto_vectorizado = vectorizer.fit_transform([texto])
    palavras_frequentes = dict(zip(vectorizer.get_feature_names_out(), texto_vectorizado.sum(axis=0).A1))
    return palavras_frequentes

def analisar_texto_marketing(texto_marketing, arquivo_avaliacoes, ngram_range=(1, 2), max_features=30):
    # Carregar avaliações
    df_avaliacoes = pd.read_csv(arquivo_avaliacoes)
    todas_avaliacoes = " ".join(df_avaliacoes["Avaliação"])

    # Processar palavras e n-grams do texto de marketing e das avaliações
    palavras_marketing = processar_texto(texto_marketing, ngram_range, max_features)
    palavras_avaliacoes = processar_texto(todas_avaliacoes, ngram_range, max_features)

    # Comparar palavras
    alinhados = set(palavras_marketing) & set(palavras_avaliacoes)
    nao_alinhados = set(palavras_marketing) - set(palavras_avaliacoes)

    # Resultados
    print("\nPalavras/n-grams alinhados com os clientes:")
    for termo in alinhados:
        print(f"- {termo} (Marketing: {palavras_marketing[termo]}, Avaliações: {palavras_avaliacoes[termo]})")

    print("\nPalavras/n-grams NÃO encontrados nas avaliações dos clientes:")
    for termo in nao_alinhados:
        print(f"- {termo} (Marketing: {palavras_marketing[termo]})")

# Texto do setor de marketing (exemplo)
texto_marketing = """
Nosso produto é inovador, prático e confiável. Ele oferece uma experiência única, eficiência e qualidade superior.
"""

# Caminho do arquivo de avaliações
arquivo_avaliacoes = "avaliacoes_exemplo.csv"

# Analisar o texto de marketing em relação às avaliações
analisar_texto_marketing(texto_marketing, arquivo_avaliacoes)