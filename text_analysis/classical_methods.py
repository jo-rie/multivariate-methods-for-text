from __future__ import annotations

import os

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

DEFAULT_STOPWORDS_PATH = os.path.join(os.path.dirname(__file__), "STOPWORDS-DE.TXT")


class VectorizerResult(BaseModel):
    """Class representing the result of a text comparison vectorisation"""

    text_1: str
    text_2: str
    vector_1: np.ndarray
    vector_2: np.ndarray

    model_config = {"arbitrary_types_allowed": True}


class BOWVectorizerResult(VectorizerResult):
    feature_names: np.ndarray

    @property
    def data_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=self.feature_names, data=self.matrix)

    @property
    def matrix(self) -> np.ndarray:
        return np.vstack([self.vector_1, self.vector_2])


def read_stopwords(file_path: str = DEFAULT_STOPWORDS_PATH):
    with open(file_path, encoding="utf-8") as file:
        stopwords = file.read().splitlines()
    return stopwords


def bag_of_words_vectorizer(
    text_1: str,
    text_2: str,
    ngram_range: tuple[float, float] = (1, 1),
) -> BOWVectorizerResult:
    stopwords_list = read_stopwords()

    count_vectorizer = CountVectorizer(
        stop_words=stopwords_list,
        ngram_range=ngram_range,
    )

    count_matrix = count_vectorizer.fit_transform([text_1, text_2]).toarray()

    return BOWVectorizerResult(
        text_1=text_1,
        text_2=text_2,
        vector_1=count_matrix[0, :],
        vector_2=count_matrix[1, :],
        feature_names=count_vectorizer.get_feature_names_out(),
    )


def bag_of_words_jaccard(
    text_1: str,
    text_2: str,
    ngram_range: tuple[float, float] = (1, 1),
) -> float:
    data_frame = bag_of_words_vectorizer(
        text_1=text_1,
        text_2=text_2,
        ngram_range=ngram_range,
    ).data_frame

    return data_frame.all(axis=0).sum() / data_frame.any(axis=0).sum()


def tf_idf_vectorizer(
    text_1: str,
    text_2: str,
    ngram_range: tuple[float, float] = (1, 1),
) -> BOWVectorizerResult:
    stopwords_list = read_stopwords()

    count_vectorizer = TfidfVectorizer(
        stop_words=stopwords_list,
        ngram_range=ngram_range,
    )

    count_matrix = count_vectorizer.fit_transform([text_1, text_2]).toarray()

    return BOWVectorizerResult(
        text_1=text_1,
        text_2=text_2,
        vector_1=count_matrix[0, :],
        vector_2=count_matrix[1, :],
        feature_names=count_vectorizer.get_feature_names_out(),
    )


def latent_semantic_analysis_vectorizer(
    text_1: str,
    text_2: str,
    nb_dimensions: int,
    ngram_range: tuple[float, float] = (1, 1),
) -> VectorizerResult:
    tf_idf_matrix = tf_idf_vectorizer(
        text_1=text_1,
        text_2=text_2,
        ngram_range=ngram_range,
    ).matrix
    transformed_matrix = TruncatedSVD(n_components=2).fit_transform(tf_idf_matrix)

    return VectorizerResult(
        text_1=text_1,
        text_2=text_2,
        vector_1=transformed_matrix[0, :],
        vector_2=transformed_matrix[1, :],
    )
