import os

import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

from text_analysis.embeddings import calculate_embedding_similarity, get_embedding

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
)
model_name = os.getenv("AZURE_DEPLOYMENT_NAME", "")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

base_strings = [
    (
        "Die Korrespondenzanalyse ist ein Verfahren, Struktur in Häufigkeitstabellen "
        "zu erkennen und damit auf Ähnlichkeiten oder Unähnlichkeiten von Objekten "
        "bzw. Merkmalsausprägungen zu schließen. Dies geschieht durch möglichst gute "
        "Darstellung der Zeilen- und Spaltenprofile der Häufigkeitstabelle in einem "
        "niedrigdimensionalen Unterraum in Form weniger Basisvektoren."
    ),
    (
        "Die grafische Darstellung von Daten gehört zu den zentralen Schritten der "
        "multivariaten Datenanalyse."
    ),
    (
        "Es gibt verschiedene Konstruktionsverfahren solcher Cluster, hier wird ein "
        "kurzer Überblick über partionierende Clusterverfahren, deren Ergebnis eine "
        "disjunkte Klassifikation der Objekte ist, und hierarchische Verfahren "
        "gegeben. Die hierarchischen Verfahren ermitteln disjunkte Klassifikationen "
        "in Form eines gröber oder feiner werdenden Baumes, beginnend mit der "
        "feinsten (agglomerative Verfahren) oder gröbsten (divisive Verfahren) "
        "Klassifikation."
    ),
]

embedding_vecs = [get_embedding(user_input=base_str, client=client, model_name=model_name) for base_str in base_strings]

# Add a Streamlit area with three radio buttons
option = st.radio(
    "Wähle einen Text zum Vergleichen:",
    options=["Text 1", "Text 2", "Text 3"],
    captions=base_strings,
)


def get_text_number(text: str) -> int:
    mapping = {"Text 1": 0, "Text 2": 1, "Text 3": 2}
    return mapping.get(text, -1)


def generate_response(prompt: str):
    embedding_prompt = get_embedding(user_input=prompt, client=client, model_name=model_name)
    distance = calculate_embedding_similarity(
        embedding_1=embedding_vecs[get_text_number(option)],
        embedding_2=embedding_prompt,
    )
    return (
        f"Cosinus-Ähnlichkeit: {distance:.6f}\n\nTokens verbraucht: ("
        f"input: {embedding_prompt.usage.prompt_tokens}, "
        f"total: {embedding_prompt.usage.total_tokens})\n\n"
        f"Vergleichstext: {base_strings[get_text_number(option)]}"
    )


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Was ist der Text, den ich embedden soll?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = generate_response(prompt=prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
