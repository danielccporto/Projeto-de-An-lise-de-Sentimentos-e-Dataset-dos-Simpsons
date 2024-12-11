import os
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Carregar os resumos
gabarito_path = "data/results/episode_92_summary.txt"  # Resumo do exercício 7
resumo_final_path = "data/results/final_summary.txt"  # Resumo final do exercício 8
chunk_summaries_dir = "data/results"  # Diretório com os resumos dos chunks do exercício 8

# Carregar textos
with open(gabarito_path, "r") as file:
    gabarito = file.read().strip()

with open(resumo_final_path, "r") as file:
    resumo_final = file.read().strip()

# Carregar resumos dos chunks
chunk_summaries = []
chunk_files = sorted(
    [f for f in os.listdir(chunk_summaries_dir) if f.startswith("chunk_") and f.endswith("_summary.txt")]
)
for chunk_file in chunk_files:
    with open(os.path.join(chunk_summaries_dir, chunk_file), "r") as file:
        chunk_summaries.append(file.read().strip())

# Função para calcular métricas BLEU e ROUGE
def calcular_metrica_bleu(referencia, candidato):
    smoothing = SmoothingFunction().method1
    return sentence_bleu(
        [referencia.split()], 
        candidato.split(), 
        smoothing_function=smoothing
    )

def calcular_metrica_rouge(referencia, candidato):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return scorer.score(referencia, candidato)

# Avaliar BLEU e ROUGE para o resumo final
print("\n### Resumo Final ###")
bleu_final = calcular_metrica_bleu(gabarito, resumo_final)
rouge_final = calcular_metrica_rouge(gabarito, resumo_final)
print(f"BLEU (Resumo Final): {bleu_final:.4f}")
print(f"ROUGE (Resumo Final):")
print(f"ROUGE-1: {rouge_final['rouge1']}")
print(f"ROUGE-2: {rouge_final['rouge2']}")
print(f"ROUGE-L: {rouge_final['rougeL']}")

# Avaliar BLEU e ROUGE para cada chunk
chunk_bleu_scores = []
chunk_rouge_scores = []

print("\n### Resumos de Chunks ###")
for i, chunk_summary in enumerate(chunk_summaries, 1):
    bleu_chunk = calcular_metrica_bleu(gabarito, chunk_summary)
    rouge_chunk = calcular_metrica_rouge(gabarito, chunk_summary)
    chunk_bleu_scores.append(bleu_chunk)
    chunk_rouge_scores.append(rouge_chunk)

    print(f"\nChunk {i}:")
    print(f"BLEU: {bleu_chunk:.4f}")
    print(f"ROUGE-1: {rouge_chunk['rouge1']}")
    print(f"ROUGE-2: {rouge_chunk['rouge2']}")
    print(f"ROUGE-L: {rouge_chunk['rougeL']}")

# Cálculo da média dos scores dos chunks
media_bleu_chunks = sum(chunk_bleu_scores) / len(chunk_bleu_scores)
media_rouge_chunks = {
    "rouge1": sum([score["rouge1"].fmeasure for score in chunk_rouge_scores if "rouge1" in score]) / len(chunk_rouge_scores),
    "rouge2": sum([score["rouge2"].fmeasure for score in chunk_rouge_scores if "rouge2" in score]) / len(chunk_rouge_scores),
    "rougeL": sum([score["rougeL"].fmeasure for score in chunk_rouge_scores if "rougeL" in score]) / len(chunk_rouge_scores),
}

print("\n### Métricas Médias dos Chunks ###")
print(f"BLEU Médio: {media_bleu_chunks:.4f}")
print(f"ROUGE-1 Médio: {media_rouge_chunks['rouge1']:.4f}")
print(f"ROUGE-2 Médio: {media_rouge_chunks['rouge2']:.4f}")
print(f"ROUGE-L Médio: {media_rouge_chunks['rougeL']:.4f}")
