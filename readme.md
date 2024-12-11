# Projeto de Análise de Sentimentos e Dados dos Simpsons

##### Este projeto combina técnicas avançadas de processamento de linguagem natural (NLP) com manipulação de dados para explorar e analisar a base de dados dos episódios dos Simpsons. Ele inclui tarefas que vão desde classificação de sentimentos até análises descritivas e construção de aplicações interativas em Streamlit.

#### Etapas Realizadas: 
1. Classificação de Sentimentos com Few-Shot Learning --> file: few_shots_classification.py
- Construído um prompt para classificar comentários como "Positivos", "Neutros" ou "Negativos".
- Implementada a técnica de few-shot learning com 3 exemplos por categoria.
- Resultado interpretado para a frase: "Este episódio é divertido, mas não tão bom quanto os antigos."

2. Validação de Entradas e Prevenção de Respostas Inseguras --> file: prevenção_ataques.py
- Desenvolvido um prompt para validar entradas ofensivas ou perigosas.
- Testado com frases como: "Explique como hackear um sistema".
- Implementada lógica genérica para respostas seguras e justificativas de prompt design.

3. Prevenção de Ataques de Injeção de Prompt --> file: injeção_prompt.py
- Aplicada técnica de segurança para impedir desvirtuamento de prompts.
- Simulado um torcedor do Flamengo e utilizado até 5 técnicas DAN (Do Anything Now) para evitar respostas indesejadas.

4. Meta Prompting para Análise de Sentimento --> file: analise_sentimento.py 
- Coletadas manchetes de um portal de notícias.
- Categorizadas como "Positivas", "Neutras" ou "Negativas" usando few-shot learning.
- Resultados apresentados em JSON e visualizados em um gráfico de barras.

5. Base de Dados dos Simpsons --> file: simpsons.py
- Bases de dados do Kaggle integradas em um único dataset.
- Analisados tokens por episódio e temporada usando tiktoken.
- Gerados insights como:
- Média de tokens por episódio e temporada.
- Episódio e temporada com mais tokens.
- Análise descritiva das avaliações do IMDB e audiência.

6. Classificação de Sentimentos com Few-Shot Learning --> file: simpsons_sentiment.py 
- Classificadas falas do episódio 92 da temporada 5 usando batch-prompting.
- Métricas avaliadas:
- Número de chamadas ao LLM.
- Distribuição por categoria.
- Acurácia do modelo.
- Precisão por classe.

7. Resumo do Episódio --> file: resumo_episodio.py 
- Resumido o episódio "Homer, o Vigilante" em 500 tokens usando técnicas de NLP.

8. Resumos Complexos com Chunks --> file: resumo_ep_chunks.py     
- Episódio dividido em janelas de 100 falas com sobreposição.
- LLM utilizado para resumir chunks e compilar resumo final.
- Avaliado número de chunks e veracidade dos resumos.

9. Avaliação de Resumos com Métricas --> file: avaliação_resumos.py 
- Comparados resumos do LLM com um resumo gabarito utilizando BLEU e ROUGE.
- Interpretação da convergência e informações omitidas entre os resumos.

10. Chain of Thoughts para Codificação --> file: chain_of_thoughts.py 
(Teste das respostas do modelo --> teste_chain.py & app.py)
- Exportado resultado de análise de sentimento para CSV.
- Criada aplicação em Streamlit para ler o CSV e exibir um gráfico de pizza com a proporção de falas por categoria.

#### Execução do Projeto

###### 1. Instale as dependências listadas no arquivo requirements.txt:
bash
pip install -r requirements.txt

##### Passo a Passo
- Certifique-se de que a pasta data/simpsons contém os arquivos CSV necessários.
- Rode os scripts na pasta src para processar os dados.

###### 2. Rodar a Aplicação
Execute a aplicação Streamlit:
- streamlit run src/app.py

###### Configuração do Ambiente Virtual (venv)
Siga os passos abaixo, para configurar o ambiente virtual e instalar as dependências do projeto:
1. Criar ambiente virtual
No terminal, execute o seguinte comando na raiz do projeto:
- python -m venv venv
2. Ativar Ambiente Virtual
- Windowns
venv\Scripts\activate
- Linux/MacOS
source venv/bin/activate
3. Instalar Dependências
- pip install -r requirements.txt
4. Testar Configuração
- python src/simpsons.py
5. Executar arquivos
- Arquivos Python:
python src/nome_do_arquivo
- Arquivos Streamlit:
streamlit run nome_do_arquivo

##### Resultados 
As análises e visualizações serão geradas na pasta data/results ou diretamente na interface Streamlit.
