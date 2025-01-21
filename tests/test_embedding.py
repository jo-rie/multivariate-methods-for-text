import os
from pathlib import Path

import numpy as np
import pytest
from dotenv import load_dotenv
from openai import AzureOpenAI
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage

from text_analysis.embeddings import calculate_embedding_similarity, get_embedding


@pytest.fixture
def path_to_env() -> Path:
    return Path(__file__).parent.parent.joinpath(".env")


@pytest.fixture
def client(path_to_env: Path) -> AzureOpenAI:
    load_dotenv(path_to_env)
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    )


@pytest.fixture
def model_name(path_to_env: Path) -> str:
    load_dotenv(path_to_env)
    return os.getenv("AZURE_DEPLOYMENT_NAME", "")


def create_dummy_create_embedding_response(
    list_of_floats: list[float],
) -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse(
        usage=Usage(prompt_tokens=1, total_tokens=1),
        model="test-embedding-3-large",
        object="list",
        data=[Embedding(embedding=list_of_floats, index=1, object="embedding")],
    )


def test_get_embedding_returns_CreateEmbeddingResponse(
    client: AzureOpenAI, model_name: str
):
    result = get_embedding(user_input="Test", client=client, model_name=model_name)
    assert isinstance(result, CreateEmbeddingResponse)


def test_compute_distance_returns_right_value():
    embedding_1 = create_dummy_create_embedding_response(
        [0, 0, 0, 1, 1, 1, 1, 1, 2, 1, 2, 0, 1, 0]
    )
    embedding_2 = create_dummy_create_embedding_response(
        [0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1]
    )
    result = calculate_embedding_similarity(
        embedding_1=embedding_1, embedding_2=embedding_2
    )
    assert result == 0.6885303726590962
