import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json

# Configuração da API do Gemini
load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("A chave da API não foi encontrada. Verifique o arquivo .env.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

def coletar_manchetes(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    manchetes = [item.get_text(strip=True) for item in soup.select(
    'div.feed-post-body-title.gui-color-primary.gui-color-hover > div > h2 > a > p')]

    if not manchetes:
        raise ValueError("Nenhuma manchete encontrada. Verifique o seletor CSS.")
    return manchetes


def classificar_manchetes(manchetes):
    resultados = []
    for manchete in manchetes:
        try:
            response = model.generate_content(f"""
        Você é um assistente de análise de sentimentos. Sua tarefa é classificar manchetes 
        de notícias em três categorias: "Positiva", "Neutra" ou "Negativa". 
        Aqui estão alguns exemplos:

        Manchete: "A economia cresce 5% no último trimestre"
        Classificação: Positiva

        Manchete: "Preços da gasolina permanecem estáveis"
        Classificação: Neutra

        Manchete: "Tempestade deixa 10 mortos no sul do país"
        Classificação: Negativa

        Agora classifique a seguinte manchete. Responda apenas com a categoria (Positiva, Neutra ou Negativa):

        Manchete: "{manchete}"
        Resposta:

            """)
           
            categoria = response.text.split(":")[-1].strip()
            if categoria not in ["Positiva", "Neutra", "Negativa"]:
                categoria = "Indefinida"
            resultados.append({"manchete": manchete, "categoria": categoria})
        except Exception as e:
            print(f"Erro ao classificar a manchete '{manchete}': {e}")
            resultados.append({"manchete": manchete, "categoria": "Erro"})
    return resultados

def gerar_grafico(resultados):
    df = pd.DataFrame(resultados)
    if df.empty or "categoria" not in df.columns:
        raise ValueError("Dados insuficientes para gerar o gráfico.")
    contagem = df["categoria"].value_counts()
    contagem.plot(kind="bar")
    plt.title("Quantidade de Manchetes por Categoria")
    plt.xlabel("Categoria")
    plt.ylabel("Quantidade")
    plt.show()

# Pipeline
url = "https://g1.globo.com/"
try:
    manchetes = coletar_manchetes(url)
    resultados = classificar_manchetes(manchetes)
    with open("resultados.json", "w") as f:
        json.dump(resultados, f, indent=4)
    gerar_grafico(resultados)
except Exception as e:
    print(f"Erro: {e}")
