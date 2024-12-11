
import csv
import matplotlib.pyplot as plt

csv_file_path = "data/results/sentiment_analysis.csv"

def analyze_sentiment(csv_filepath):
    """
    Analisa um arquivo CSV de análise de sentimento, conta as ocorrências de cada categoria de sentimento   
    e calcula a proporção de cada categoria em relação ao total.

    Args:
        csv_filepath (str): O caminho para o arquivo CSV.

    Returns:
        dict: Um dicionário onde as chaves são as categorias de sentimento e os valores são suas proporções.
              Retorna um dicionário vazio se o arquivo não for encontrado ou estiver vazio.
    """

    try:
        with open(csv_filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)  # Assume que o arquivo CSV tem um cabeçalho
            sentiment_counts = {}
            total_utterances = 0

            for row in reader:
                sentiment = row['classification']
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
                total_utterances += 1

            if total_utterances == 0:
                return {}  # Retorna um dicionário vazio se o arquivo estiver vazio

            sentiment_proportions = {}
            for sentiment, count in sentiment_counts.items():
                sentiment_proportions[sentiment] = count / total_utterances

            return sentiment_proportions

    except FileNotFoundError:
        print(f"Erro: Arquivo CSV não encontrado em {csv_filepath}")
        return {}


proportions = analyze_sentiment(csv_file_path)
print("Dicionário de Proporções das Falas do arquivo CSV:")
print(proportions)

# Extrai rótulos e proporções
labels = list(proportions.keys())
sizes = list(proportions.values())

# Cria o gráfico de pizza
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Garante que a pizza seja um círculo

# Adiciona título
plt.title('Proporção de Falas por Categoria')
# Mostra o gráfico
plt.show()
