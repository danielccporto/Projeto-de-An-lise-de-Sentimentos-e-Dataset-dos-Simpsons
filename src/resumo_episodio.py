import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import tiktoken

# Configuração da API
load_dotenv('.env')
genai.configure(api_key=os.environ["API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# Diretório de logs
logs_path = "data/results"
os.makedirs(logs_path, exist_ok=True)

# Carregar Dados
data = pd.read_parquet('data/results/database_thesimpsons.parquet')

# Definir falas como linhas estruturadas
data['line'] = ("Espisode " + data['episode_id'].astype(str) + ' | ' + 
                data['location_normalized_name'].fillna('') + ', ' + 
                data['character_normalized_name'].fillna('') + ' said: ' + 
                data['normalized_text'].fillna(''))

# Filtrar o episódio
episode_season = 5
episode_id = 92
filtered_data = data[
    (data['episode_season'] == episode_season) &
    (data['episode_id'] == episode_id)
].sort_values('number')

# Garantir que apenas linhas com texto válido sejam processadas
filtered_lines = filtered_data.dropna(subset=['normalized_text'])['normalized_text'].tolist()

# Criar o prompt para o resumo baseado nas falas
prompt = f"""
You are an expert in summarizing TV show episodes.
The following are dialogues from a Simpsons episode. Based on the dialogues, create a concise summary (around 500 tokens) explaining the main plot, key events, and the resolution of the episode.

Dialogues:
{filtered_lines[:200]}  # Limitar a 200 falas para evitar excesso de tokens.

Please summarize the episode.
"""

# Chamar o modelo para gerar o resumo
response = model.generate_content(prompt)
summary = response.text.strip()

# Salvar o resumo em um arquivo
summary_path = os.path.join(logs_path, "episode_92_summary.txt")
with open(summary_path, "w") as summary_file:
    summary_file.write(summary)

# Analisar os tokens no resumo
encoding = tiktoken.get_encoding("cl100k_base")
token_count = len(encoding.encode(summary))

# Exibir resultados
print("Resumo do Episódio:")
print(summary)
print("\nQuantidade de tokens no resumo:", token_count)

