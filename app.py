import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

st.write("# Classificação de Iris")
st.write("## Exemplo com comprimento de petala e sépala")

st.sidebar.write("## Parâmetros")
comp_sepala = st.sidebar.slider("Comprimento da sépala", 4.0, 8.0, 5.8, 0.1)
larg_sepala = st.sidebar.slider("Largura da sépala", 2.0, 4.0, 3.0, 0.1)
comp_petala = st.sidebar.slider("Comprimento da pétala", 0.9, 7.0, 3.8, 0.1)
larg_petala = st.sidebar.slider("Largura da pétala", 0.1, 2.0, 1.0, 0.1)

with open("objetos.pkl", "rb") as arquivo:
  ss, dtc = pickle.load(arquivo)

estrutura = { 'comp_sepala': comp_sepala,
              'larg_sepala': larg_sepala,
              'comp_petala': comp_petala,
              'larg_petala': larg_petala}

df= pd.DataFrame(estrutura, index=[0])

st.write("### Parâmetros de entrada")
st.write(df)

df = ss.transform(df)
st.write(df)

predicao = dtc.predict(df)
st.write(f"A classe dessa flor é: **{predicao[0]}**")
#st.write(predicao)

predicao = dtc.predict_proba(df)
predicao = pd.DataFrame(predicao)
predicao.rename({
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}, axis=1, inplace=True)

st.write("Probabilidades")
st.write(predicao)
