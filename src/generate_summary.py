import os
from dotenv import load_dotenv
import pandas as pd 
import json 
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import joblib 
import pyarrow

class ChunkSummary():
    def __init__(self, model_name, apikey, text, window_size, overlap_size):
        self.text = text
        if isinstance(self.text, str):
            self.text = [self.text]
        self.window_size = window_size
        self.overlap_size = overlap_size
        # Aplicacao dos chunks
        self.chunks = self.__text_to_chunks()
        self.model = self.__create_model(apikey, model_name)


    def __create_model(self, apikey, model_name):
        genai.configure(api_key=apikey)
        self.prompt_base = f"""
        You are an editor assistant from the "The Simpsons" show.
        You will receive the #subtitles# from real episodes in the format:
        <location>, <character> said: <character line>
        
        You must create a summary of the #subtitles#, pointing out the most
        relevant information, jokes and key players in the story. Bare in mind
        that the summary must describe how the episode started, which key
        points are relevant along the story and its gran finale.
        The summary output must be written as a plain JSON with field 'summary'.
        """
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        generation_config = {
            'temperature': 0.2,
            'top_p': 0.8,
            'top_k': 20,
            'max_output_tokens': 1000
        }
        return genai.GenerativeModel(
            model_name,
            system_instruction=self.prompt_base,
            generation_config = generation_config,
            safety_settings=safety_settings
        )


    
    def __text_to_chunks(self):       
        n = self.window_size  # Tamanho de cada chunk
        m = self.overlap_size  # overlap entre chunks
        return [self.text[i:i+n] for i in range(0, len(self.text), n-m)]


    def __create_chunk_prompt(self, chunk):
        episode_lines = '\n'.join(chunk)
        prompt = f"""
        #subtitles#
        {episode_lines}
        ######
        Summarize it.
        """
        return prompt
        
    
    def __summarize_chunks(self):
        # Loop over chunks
        chunk_summaries = []
        for i, chunk in enumerate(self.chunks):
            print(f'Summarizing chunk {i+1} from {len(self.chunks)}')
            # Create prompt
            prompt = self.__create_chunk_prompt(chunk)
            response = self.model.generate_content(prompt)
            # Apendar resposta do chunk
            chunk_summaries.append(response.text)
            
            # if i == 4: break

        return chunk_summaries


    def summarize(self):
        print('Summarizing text')
        # Chamar o sumario dos chunks
        self.chunk_summaries = self.__summarize_chunks()
        # Prompt final
        summaries = '- ' + '\n- '.join(self.chunk_summaries)
        prompt = f"""
        You are an editor working on The Simpsons show. You must summarize
        a show episode considering the other summaries from part of the episode.
        The partitioned summaries are listed below:
        {summaries}
        ######
        The summary must describe the details in the story, like jokes, and details
        on what happens in the end with the key characters.
        Write a final summary based on the partitioned summaries in JSON format with
        the field 'summary'
        """
        print('Final summarization')
        response = self.model.generate_content(prompt)
        
        return response.text


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

data.to_parquet('data/results/database_thesimpsons.parquet')

# Configuração Dataframe para episódio/temporada
episode_season = 5
episode_id = 92
X = (data[(data.episode_season == episode_season) &
          (data.episode_id == episode_id)].sort_values('number')
)

X['line'] = (X['location_normalized_name'].fillna('') + ', ' + 
             X['character_normalized_name'].fillna('') + ' said: ' + 
             X['normalized_text'].fillna('')
)

load_dotenv('../.env')

summarizer = ChunkSummary(
    model_name = "gemini-1.5-flash",
    apikey = os.environ["API_KEY"],
    text = X['line'].tolist(),
    window_size = 40,
    overlap_size = 5
)

episode_summary = summarizer.summarize()
final_summmary = json.loads(episode_summary.replace("```json\n",'').replace("\n```",''))
final_summmary

os.makedirs('data/results/')
# Exportar dicionario com chunks, resumos dos chunks e do resumo final
joblib.dump({
    'chunks': summarizer.chunks,
    'chunks': summarizer.chunk_summaries,
    'chunks': final_summmary,
    
}, 'data/results/simpsons_episode_summary.joblib')