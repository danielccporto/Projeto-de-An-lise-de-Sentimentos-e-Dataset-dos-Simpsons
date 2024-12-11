import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Título da aplicação
st.title("Análise de Sentimento - Proporção de Falas por Categoria")

try:
    # Lê o arquivo CSV
    df = pd.read_csv("data/results/sentiment_analysis.csv")

    # Verifica se a coluna 'classification' existe
    if 'classification' not in df.columns:
        st.error("O arquivo CSV não contém a coluna 'classification'.")
    else:
        # Calcula a proporção de falas por categoria
        category_counts = df['classification'].value_counts()
        category_proportions = category_counts / category_counts.sum()

        # Cria o gráfico de pizza interativo
        fig = go.Figure(data=[go.Pie(labels=category_proportions.index, values=category_proportions.values)])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title_text='Proporção de Falas por Categoria')

        # Exibe o gráfico
        st.plotly_chart(fig)

except FileNotFoundError:
    st.error("O arquivo 'data/results/sentiment_analysis.csv' não foi encontrado.")
except pd.errors.EmptyDataError:
    st.error("O arquivo 'data/results/sentiment_analysis.csv' está vazio.")
except Exception as e:
    st.error(f"Ocorreu um erro: {e}")
