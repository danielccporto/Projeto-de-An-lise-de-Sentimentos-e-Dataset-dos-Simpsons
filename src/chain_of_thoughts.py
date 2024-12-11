import google.generativeai as genai
import os
from dotenv import load_dotenv
import pandas as pd 

# Configuração LLM
load_dotenv('.env')

# Configurar a API do Gemini
genai.configure(api_key=os.environ["API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# Carregar o arquivo CSV
csv_path = "data/results/sentiment_analysis.csv"
try:
    sentiment_data = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Erro: Arquivo {csv_path} não encontrado.")
    exit()

# Obter uma visão geral dos dados
sample_data = sentiment_data.head().to_dict()


# Chain of Thoughts Prompting 
prompt_1 = f""" Escreva um código em Python que:
1. Carregue o arquivo CSV localizado no caminho `data/results/sentiment_analysis.csv`.
2. Conte o número de falas para cada categoria de sentimento (negative, neutral, positive).
3. Calcule a proporção de cada categoria com base no total de falas.
4. Retorne os resultados como um dicionário no formato {{"categoria": proporção}}.

Aqui estão os primeiros dados do arquivo CSV para referência:
{sample_data}
"""

prompt_2 = """
Baseando-se no dicionário de proporções de categorias gerado anteriormente, 
escreva um código em Python para criar um gráfico de pizza que mostre a proporção de falas 
por categoria. Use a biblioteca matplotlib para isso.
"""

prompt_3 = """
Baseando-se nos dois passos anteriores, escreva um código em Python que implemente uma aplicação Streamlit para:
1. Ler o arquivo CSV `data/results/sentiment_analysis.csv`.
2. Calcular a proporção de falas por categoria.
3. Exibir um gráfico de pizza interativo mostrando a proporção de falas por categoria.
"""

# Chamada modelo e print resultados 
print("Retorno Prompt_1:")
response_1 = model.generate_content(prompt_1)
summary_1 = response_1.text.strip()
print(summary_1)

print("Retorno Prompt_2:")
response_2 = model.generate_content(prompt_2)
summary_2 = response_2.text.strip()
print(summary_2)

print("Retorno Prompt_3:")
response_3 = model.generate_content(prompt_3)
summary_3 = response_3.text.strip()
print(summary_3)