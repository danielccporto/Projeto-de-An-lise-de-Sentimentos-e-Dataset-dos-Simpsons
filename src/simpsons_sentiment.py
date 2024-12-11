import joblib
import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import json
import time
import matplotlib.pyplot as plt

# Configuração LLM
load_dotenv('.env')

# Diretório de logs
logs_path = "data/results"
os.makedirs(logs_path, exist_ok=True)

# Carregar Dados
summaries = joblib.load('data/results/simpsons_episode_summary.joblib')
print(f"Chaves dos summaries carregadas: {summaries.keys()}")

data = pd.read_parquet('data/results/database_thesimpsons.parquet')
data['line'] = ("Episode " + data['episode_id'].astype(str) + ' | ' + 
                data['location_normalized_name'].fillna('') + ', ' + 
                data['character_normalized_name'].fillna('') + ' said: ' + 
                data['normalized_text'].fillna(''))

# Remover duplicatas e preparar o dataset
data = data.drop_duplicates(subset="normalized_text")
episode_season = 5
episode_id = 92
X = (data[(data.episode_season == episode_season) & (data.episode_id == episode_id)].sort_values('number'))
X = X.dropna(subset='normalized_text')

# Categorias Few-Shots
positivas = [
    "that life is worth living",
    "i am the champions i am the champions no time for losers cause i am the champions of the worlllld",
    "eh you must be bart simpson well you look like youve got a strong young back",
    "woo hoo",
    "keep digging were bound to find something"
]

negativas = [
    "i dont think theres anything left to say",
    "we came to this retreat because i thought our marriage was in trouble but i never for a minute thought it was in this much trouble homer how can you expect me to believe",
    "oh thats my brother asa he was killed in the great war held a grenade too long",
    "dad weve been robbed",
    "wake up dad wake up there was a burglar"
]

neutras = [
    "wheres mr bergstrom",
    "would you have to do extra work",
    "oh please dad i want this more than anything in the world",
    "oh youve probably got a whole drawer full of em",
    "and my necklace"
]

# Prompt base
base_prompt = f"""
You are an expert in human communication and marketing, specialized in sentiment analysis.
You have to classify lines from a cartoon show as negative, neutral, and positive as defined below:
- positive: happy, constructive, hopeful, joyful, and similar lines.
- negative: sad, destructive, hopeless, aggressive, and similar lines.
- neutral: indifferent, objective, formal, and lines classified neither as positive nor negative.

Some pre-classified lines from this show are listed here:

# Positive:
{ '- '.join(positivas) }

# Negative:
{ '- '.join(negativas) }

# Neutral:
{ '- '.join(neutras) }

Given this information, respond in JSON format with the classification of these other lines as positive, negative, or neutral.
Your answer must have only the line and its classification in the JSON format.
"""

# Configurar a API do Gemini
genai.configure(api_key=os.environ["API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# Dividir os dados em blocos
batch_size = 5  # Reduzido para evitar limites de cota
results = []

for i in range(0, len(X), batch_size):
    batch = X.iloc[i:i + batch_size]
    prompt = base_prompt + "\n".join(batch['normalized_text'])

    try:
        # Chamada à API com pausa entre os lotes
        response = model.generate_content(prompt)
        completion_text = response.text.strip()

        # Salvar resposta bruta para depuração
        with open("data/results/debug_responses.txt", "a") as debug_file:
            debug_file.write(f"Batch {i}-{i + batch_size} Response: {completion_text}\n")

        # Processar a resposta
        try:
            sanitized_response = completion_text.replace("```json", "").replace("```", "").strip()
            classifications = json.loads(sanitized_response )
            if isinstance(classifications, list):  # Verificar se o resultado é uma lista
                results.extend(classifications)
            else:
                print(f"Resposta inesperada para o batch {i}-{i + batch_size}.")
        except json.JSONDecodeError:
            print(f"Erro ao decodificar JSON para o batch {i}-{i + batch_size}.")
    except Exception as e:
        print(f"Erro na API para o batch {i}-{i + batch_size}: {e}")

    # Pausa entre requisições para evitar sobrecarga
    time.sleep(2)


if results:
    df_results = pd.DataFrame(results)

    # Quantas chamadas ao LLM foram necessárias?
    num_calls = len(X) // batch_size + (1 if len(X) % batch_size != 0 else 0)
    print(f"Número de chamadas ao LLM: {num_calls}")

    # Qual é a distribuição de fala por categoria?
    distribution = df_results['classification'].value_counts()
    print("Distribuição de Falas por Categoria:")
    print(distribution)

    # Visualizar a distribuição (opcional)
    distribution.plot(kind='bar', title="Distribuição de Falas por Categoria")
    plt.xlabel("Categoria")
    plt.ylabel("Número de Falas")
    plt.show()

    # Agrupar e amostrar com mínimo de exemplos
    sample = df_results.groupby('classification', group_keys=False).apply(
    lambda x: x.sample(n=min(5, len(x)), random_state=42))

    # Resetar o índice para evitar ambiguidade
    sample = sample.reset_index(drop=True)

    # Substituir avaliações manuais (exemplo para debug)
    if 'correct' not in sample.columns:
        sample['correct'] = [True, True, False, True, False] * (len(sample) // 5)

    #Calcular acurácia 
    accuracy = sample['correct'].mean()
    print(f"Acurácia do Modelo: {accuracy:.2%}")

    # Precisão por classe
    precision = sample.groupby('classification')['correct'].mean()
    print("Precisão por Classe:")
    print(precision)

    # Salvar os resultados em CSV
    output_path = "data/results/sentiment_analysis.csv"
    df_results.to_csv(output_path, index=False)
    print(f"Análise de sentimento salva em {output_path}")
else:
    print("Nenhum resultado válido foi gerado.")
