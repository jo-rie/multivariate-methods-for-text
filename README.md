# Python Code für die Vorlesung "Multivariate Verfahren für Texte"

Dieses Repository enthält eine Streamlit-App, ein Jupyter Notebook und ein Python-Skript, die Inhalte aus der Vorlesung "Multivariate Verfahren für Texte" darstellt und auswertbar macht. Die App nutzt OpenAI's API, um verschiedene Funktionen bereitzustellen.

## Einrichtung

### Voraussetzungen

- Python 3.11
- [Poetry](https://python-poetry.org/) für das Dependency-Management (alternativ kann auch die venv händisch über das bereitgestellte `requirements.txt` erstellt werden)

### Schritte zur Einrichtung

1. **Repository klonen:**

   ```sh
   git clone <repository-url>
   cd streamlit-app
   ````

2. **Abhängigkeiten installieren:**

```sh
poetry install
```

3. **.env Datei erstellen:**

Erstelle eine .env Datei im Hauptverzeichnis des Projekts mit folgendem Inhalt:

```
AZURE_OPENAI_ENDPOINT="https://<your-azure-openai-endpoint>"
AZURE_OPENAI_KEY="<your-azure-openai-key>"
AZURE_OPENAI_DEPLOYMENT_NAME="text-embedding-3-large"
AZURE_OPENAI_API_VERSION="2023-05-15"
```

Hinweis: Ersetze <your-azure-openai-endpoint> und <your-azure-openai-key> durch deine tatsächlichen Azure OpenAI Endpoint und API Key.

Als Student bekommt man relativ viel Azure-Kontingent geschenkt (meines Wissens 100 $ im Monat). 
Darüber kann man sich im Azure OpenAI Service einen Endpunkt und Key erzeugen (eine grobe Anleitung gibt es zum Beispiel [hier](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal))


## Starten der verschiedenen Services

### Streamlit-App

Die Streamlit-App startet man über folgendes Kommando in der Konsole (ausgeführt im root-Verzeichnis des Repo und mit aktivierter venv)

```sh
poetry run streamlit run streamlit.py
```

### Jupyter Notebook

Das Jupyter Notebook startet man wie üblich in vielen IDEs wird es auch direkt angezeigt. 
Damit alles wie gewünscht läuft, sollte man die virtual env auch in Jupyter hinterlegen.

### Python-Skript

Das Python-Skript kann man einfach wie bekannt mit aktivierter venv ausführen:

```sh
python test_classical_methods.py
```


## Quellen

Die verwendeten Stop-Words stammen von https://github.com/stopwords-iso/stopwords-de?tab=readme-ov-file.
