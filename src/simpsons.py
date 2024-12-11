import pandas as pd
import tiktoken
import json 
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Carregar arquivos csv e criar dataframe unificado
df_script = pd.read_csv('data/simpsons/simpsons_script_lines.csv', low_memory=False)
df_episodes = pd.read_csv('data/simpsons/simpsons_episodes.csv', low_memory=False)
df_characters = pd.read_csv('data/simpsons/simpsons_characters.csv', low_memory=False)
df_locations = pd.read_csv('data/simpsons/simpsons_locations.csv', low_memory=False)

df_script.set_index('id', inplace=True)
df_characters['id'] = df_characters['id'].astype(str)

df_characters = df_characters.add_prefix('character_')
df_locations = df_locations.add_prefix('location_')
df_episodes = df_episodes.add_prefix('episode_')

data = (
    df_script.merge(df_episodes, left_on='episode_id', right_on='episode_id')
             .merge(df_characters, left_on='character_id', right_on='character_id', how='left')
             .merge(df_locations, left_on='location_id', right_on='location_id', how='left')
)


assert data.shape[0] == df_script.shape[0]
    
# Função para estimar número de tokens
def estimar_tokens(texto):
    encoder = tiktoken.get_encoding("cl100k_base")  # Exemplo de codificação
    # encoder = tiktoken.get_encoding("gpt-4o")  # Exemplo de codificação
    tokens = encoder.encode(texto)
    return tokens

# Exemplo de uso
X = data.dropna(subset='normalized_text').copy()
X['n_tokens'] = X.normalized_text.fillna('').apply(lambda x: len(estimar_tokens(x)))
X.shape

# Soma de Tokens por episódio 
token_counts = X.groupby('episode_id').n_tokens.sum()
#print(f"token counts: {token_counts}")

# Opção gerar Histograma 
#X.groupby('episode_id').n_tokens.sum().plot.hist(bins=100)
#print(f"Histograma Episodes x Tokens: {X}")

# Média de tokens por episódio
avg_tokens_per_episode = token_counts.mean()
print(f"Média de tokens por episódio: {avg_tokens_per_episode}")

# Média de tokens por temporada
tokens_per_season = X.groupby('episode_season').n_tokens.sum()
avg_tokens_per_season = tokens_per_season.mean()
print(f"Média de tokens por temporada: {avg_tokens_per_season}")

# Episódio com mais tokens
episode_most_tokens = token_counts.idxmax()
episode_most_tokens_count = token_counts.max()
print(f"Episódio com mais tokens: {episode_most_tokens} ({episode_most_tokens_count} tokens)")

# Temporada com mais tokens
season_most_tokens = tokens_per_season.idxmax()
season_most_tokens_count = tokens_per_season.max()
print(f"Temporada com mais tokens: {season_most_tokens} ({season_most_tokens_count} tokens)")

# Análise descritiva dos tokens
tokens_description = X['n_tokens'].describe()
print("Análise descritiva dos tokens:")
print(tokens_description)

cols = ['episode_id', 'episode_season', 'episode_original_air_date', 
        'episode_imdb_rating', 'episode_imdb_votes', 
        'episode_us_viewers_in_millions', 'episode_views']
episode_stats = data[cols].drop_duplicates()
episode_stats.to_csv('series_data.csv', sep=';', index=False)

prompt_start = f"""
You are a data scientist specialized in analysing entertainment content. You are working on the show series
"The Simpsons", investigating patterns in the series series_data. 
How can we evaluate the relationship between episode ratings ('episode_imdb_rating', 'episode_imdb_votes')
and audiences ('episode_us_viewers_in_millions', 'episode_views') in series_data.csv, considering it a CSV file
splitted by ';' with columns:

- episode_id: episode unique identifier
- episode_season: episode season number
- episode_original_air_date: date that the episode was first exhibited
- episode_imdb_rating: episode with the IMDB rating 
- episode_imdb_votes: episode with the number of voters
- episode_us_viewers_in_millions: number of episode viewers (in millions)
- episode_views: total number of episode views.

Generate a list of 5 analyses that can be implemented given the available series_data, as a JSON file:
{[
    {'Name':'analysis name',
     'Objective': 'what we need to analyze',
     'Method': 'how we analyze it'
    }
]
}
"""

# Chamar Gemini e gerar saída para prompt
load_dotenv('.env')
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content(prompt_start)
print("Resposta 'response_text' completa da API:")
print(response.text)



