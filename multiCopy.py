import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

# Este arquivo avalia as copies como "avaliadorCopy.py", 
# Porém gera um csv com os resultados, para que a campanha seja avaliada como um todo

nltk.download("stopwords")

# Stopwords em português
stop_words_pt = stopwords.words("portuguese")

def processar_texto(texto, ngram_range=(1, 2), max_features=30):
    # Criar vetorizador para n-grams
    vectorizer = CountVectorizer(stop_words=stop_words_pt, ngram_range=ngram_range, max_features=max_features)
    texto_vectorizado = vectorizer.fit_transform([texto])
    palavras_frequentes = dict(zip(vectorizer.get_feature_names_out(), texto_vectorizado.sum(axis=0).A1))
    return palavras_frequentes

def analisar_copies(caminho_copies, caminho_avaliacoes, ngram_range=(1, 2), max_features=30):

    # Carregar avaliações
    df_avaliacoes = pd.read_csv(caminho_avaliacoes)
    todas_avaliacoes = " ".join(df_avaliacoes["Avaliação"])
    palavras_avaliacoes = processar_texto(todas_avaliacoes, ngram_range, max_features)

    # Carregar copies para análise
    df_copies = pd.read_csv(caminho_copies)

    resultados = []
    for index, row in df_copies.iterrows():
        texto_marketing = row["Copy"]
        primeira_linha = texto_marketing.splitlines()[0]  # Captura apenas a primeira linha do texto

        palavras_marketing = processar_texto(texto_marketing, ngram_range, max_features)

        # Comparar palavras
        alinhados = set(palavras_marketing) & set(palavras_avaliacoes)
        nao_alinhados = set(palavras_marketing) - set(palavras_avaliacoes)

        # Registrar resultados para cada copy
        resultados.append({
            "Resumo Copy": primeira_linha,
            "Palavras Alinhadas": list(alinhados),
            "Palavras Não Alinhadas": list(nao_alinhados),
            "Total Alinhadas": len(alinhados),
            "Total Não Alinhadas": len(nao_alinhados)
        })

    # Retornar resultados em um DataFrame
    return pd.DataFrame(resultados)

# Caminhos dos arquivos
caminho_copies = "copies.csv"  # Arquivo com as copies (coluna: "Copy")
caminho_avaliacoes = "avaliacoes_exemplo.csv"  # Arquivo com avaliações

# Analisar copies e exibir resultados
df_resultados = analisar_copies(caminho_copies, caminho_avaliacoes)
df_resultados.to_csv("resultado_copies.csv", index=False)
print("Resultados salvos em 'resultado_copies.csv'")
