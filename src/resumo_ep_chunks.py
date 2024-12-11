import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import tiktoken

# Configuração da API
load_dotenv('.env')
genai.configure(api_key=os.environ["API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# Carregar Dados
data = pd.read_parquet('data/results/database_thesimpsons.parquet')

# Diretório para salvar logs
logs_path = "data/results"
os.makedirs(logs_path, exist_ok=True)

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

# Função para criar chunks com sobreposição
def create_chunks(lines, chunk_size=100, overlap=25):
    chunks = []
    start = 0
    while start < len(lines):
        end = start + chunk_size
        chunk = lines[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Criar chunks de 100 falas com sobreposição de 25
chunk_size = 100
overlap = 25
chunks = create_chunks(filtered_lines, chunk_size, overlap)

# Gerar resumos para cada chunk
chunk_summaries = []
for i, chunk in enumerate(chunks):
    chunk_prompt = f"""
    You are an expert in summarizing TV show episodes.
    The following are dialogues from a chunk of a Simpsons episode. Based on these dialogues, create a concise summary explaining the main events in this chunk.

    Dialogues:
    {chunk}

    Please summarize the dialogues.
    """
    response = model.generate_content(chunk_prompt)
    chunk_summary = response.text.strip()
    
    # Salvar o resumo do chunk em logs
    chunk_log_path = os.path.join(logs_path, f"chunk_{i + 1}_summary.txt")
    with open(chunk_log_path, "w") as log_file:
        log_file.write(chunk_summary)
    
    chunk_summaries.append(chunk_summary)

# Criar o segundo prompt com os resumos dos chunks
combined_prompt = f"""
You are an expert in summarizing TV show episodes.
The following are summaries of chunks from a Simpsons episode. Based on these summaries, create a concise and coherent final summary explaining the main plot, key events, and the resolution of the episode.

Chunk Summaries:
{chunk_summaries}

Please generate the final summary.
"""
final_response = model.generate_content(combined_prompt)
final_summary = final_response.text.strip()

# Salvar o resumo final em logs
final_summary_path = os.path.join(logs_path, "final_summary.txt")
with open(final_summary_path, "w") as final_summary_file:
    final_summary_file.write(final_summary)

# Contar tokens no resumo final usando tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
final_token_count = len(encoding.encode(final_summary))

# Exibir resultados
print("Quantidade de chunks:", len(chunks))
print("\nResumo Final do Episódio:")
print(final_summary)
print("\nQuantidade de tokens no resumo final:", final_token_count)
