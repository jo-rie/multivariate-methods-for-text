from __future__ import annotations

from text_analysis.classical_methods import (
                                             bag_of_words_jaccard,
                                             bag_of_words_vectorizer,
                                             tf_idf_vectorizer,
)

text_1 = "grafisch darstell dat zentral schritt multivariat datenanalys"
text_2 = "grafisch darstell dat multivariat datenanalys"


df = bag_of_words_vectorizer(text_1=text_1, text_2=text_2)

value = bag_of_words_jaccard(text_1=text_1, text_2=text_2)

print(df.data_frame)

print(value)

df_tfidf = tf_idf_vectorizer(text_1=text_1, text_2=text_2)

print(df_tfidf.data_frame)
