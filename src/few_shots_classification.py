
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Configuração API
load_dotenv()
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

#Configuração Modelo Gemini
model = genai.GenerativeModel("gemini-1.5-flash")


# prompt
response = model.generate_content("""
Você é um assistente inteligente que classifica comentários em três categorias: "Positivos", "Neutros" ou "Negativos". 
Aqui estão alguns exemplos:

Exemplos de comentários POSITIVOS:
1. "Adorei este episódio, foi incrível!"
2. "Muito bem produzido, estou impressionado."
3. "Os personagens estão cada vez melhores, adorei!"

Exemplos de comentários NEUTROS:
1. "Este episódio foi ok, nada de especial."
2. "Os efeitos visuais são aceitáveis, mas não impressionantes."
3. "A história está andando, mas sem grandes reviravoltas."

Exemplos de comentários NEGATIVOS:
1. "Achei este episódio bem fraco e sem graça."
2. "Os diálogos estão confusos e mal escritos."
3. "Não gostei da direção que a história está tomando."

Agora, classifique o seguinte comentário:
"Este episódio é divertido, mas não tão bom quanto os antigos."

Responda com a categoria: Positivo, Neutro ou Negativo.
""")

print(response.text)