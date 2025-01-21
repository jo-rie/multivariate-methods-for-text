FROM debian:12 AS build
ENV DEVIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev python3.11 python3-pip python3-setuptools python3-dev

RUN pip install --break-system-packages --upgrade pip
RUN ln -s /usr/bin/python3.11 /usr/bin/python


WORKDIR /app

# Install dependencies
COPY poetry.lock pyproject.toml poetry.toml /app/
COPY core /app/core
COPY text_analysis /app/text_analysis
COPY streamlit_embedding_app.py /app


RUN pip install --break-system-packages poetry && poetry install

ENV PATH=/app/.venv/bin:$PATH

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Copy files into container
ENTRYPOINT ["streamlit", "run", "streamlit_embedding_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
