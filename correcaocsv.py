import pandas as pd
# Este documento corrige
# erros causados pela incidência de aspas duplas

# Exemplo de texto com aspas duplas
copies = [
    """Bom dia! Por aqui gostamos de começar o nosso dia com um caloroso "bom dia"!""",
    """Nosso produto é "inovador" e "prático". Ele traz eficiência ao seu cotidiano.""",
    """Teste de produto"""
]

# Criar DataFrame
df_copies = pd.DataFrame({"Copy": copies})

# Salvar o arquivo CSV (aspas duplas serão escapadas automaticamente)
df_copies.to_csv("copies.csv", index=False)
