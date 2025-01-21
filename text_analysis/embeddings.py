# Function to get embedding
from __future__ import annotations

import numpy as np
from openai import AzureOpenAI
from openai.types import CreateEmbeddingResponse


def get_embedding(user_input: str, client: AzureOpenAI, model_name: str) -> CreateEmbeddingResponse:
    return client.embeddings.create(
        model=model_name,
        input=user_input,
    )


def calculate_embedding_similarity(
    embedding_1: CreateEmbeddingResponse, embedding_2: CreateEmbeddingResponse
) -> float:
    a = embedding_1.data[0].embedding
    b = embedding_2.data[0].embedding
    return calculate_cosine_similarity(a=a, b=b)


def calculate_cosine_similarity(a: list | np.ndarray, b: list | np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
