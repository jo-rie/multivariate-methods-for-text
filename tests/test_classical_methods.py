import numpy as np

from text_analysis.classical_methods import (
    bag_of_words_jaccard,
    bag_of_words_vectorizer,
    latent_semantic_analysis_vectorizer,
    read_stopwords,
    tf_idf_vectorizer,
)


def test_read_stopwords():
    stopwords = read_stopwords()
    assert isinstance(stopwords, list)
    assert len(stopwords) > 0

def test_bag_of_words_vectorizer():
    text_1 = "This is a test."
    text_2 = "This is another test."
    result = bag_of_words_vectorizer(text_1, text_2)
    assert isinstance(result.vector_1, np.ndarray)
    assert isinstance(result.vector_2, np.ndarray)
    assert result.vector_1.shape == result.vector_2.shape
    assert len(result.feature_names) > 0
    np.testing.assert_allclose(result.vector_1, np.array([0,1,1,1]))
    np.testing.assert_allclose(result.vector_2, np.array([1,1,1,1]))

def test_bag_of_words_jaccard():
    text_1 = "This is a test."
    text_2 = "This is another test."
    jaccard_index = bag_of_words_jaccard(text_1, text_2)
    assert isinstance(jaccard_index, float)
    assert 0 <= jaccard_index <= 1
    assert np.isclose(jaccard_index, 0.75)

def test_tf_idf_vectorizer():
    text_1 = "This is a test."
    text_2 = "This is another test."
    result = tf_idf_vectorizer(text_1, text_2)
    assert isinstance(result.vector_1, np.ndarray)
    assert isinstance(result.vector_2, np.ndarray)
    assert result.vector_1.shape == result.vector_2.shape
    assert len(result.feature_names) > 0
    np.testing.assert_allclose(result.vector_1, np.array([0., 0.57735027, 0.57735027, 0.57735027]))
    np.testing.assert_allclose(result.vector_2, np.array([0.6300993445179441, 0.44832087319911734, 0.44832087319911734, 0.44832087319911734]))

def test_latent_semantic_analysis_vectorizer():
    text_1 = "This is a test."
    text_2 = "This is another test."
    result = latent_semantic_analysis_vectorizer(text_1, text_2, nb_dimensions=2)
    assert isinstance(result.vector_1, np.ndarray)
    assert isinstance(result.vector_2, np.ndarray)
    assert result.vector_1.shape == (2,)
    assert result.vector_2.shape == (2,)
    np.testing.assert_allclose(result.vector_1, np.array([0.9424740130302042, -0.33427942617328743]))
    np.testing.assert_allclose(result.vector_2, np.array([0.942474013030204, 0.33427942617328743]))
