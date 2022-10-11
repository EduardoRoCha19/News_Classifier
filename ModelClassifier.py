from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import unidecode
import copy

def text_processing(noticia):
    #crear y añadir las stopwords
    stop_words = set()
    with open("Stopwords.csv") as stopwords_word:
        for word in stopwords_word:
            stop_words.add(word)

    #limpiar el texto
    cleanString = re.sub('\W+',' ', noticia )
    unaccented_string = unidecode.unidecode(cleanString)

    #utilizar el countvectorizer para hacer los tokens
    coun_vect_ = CountVectorizer(ngram_range=(1,3),stop_words=stop_words)
    coun_vect_.fit_transform([unaccented_string])
    arr_words = np.asarray(coun_vect_.get_feature_names_out())

    #diccionario a utilizar - abrir la bolsa de palabras de 1000 palabras
    tokens = {}
    with open("BoW1000UniBiTri.csv") as file:
        for word in file:
            tokens[word.replace('\n','')] = 0

    arr_vector = copy.deepcopy(tokens)
    #comparar las palabras de la noticia con la bolsa de palabras y añadir 1 valor en cada palabra encontrada.
    for i, words in enumerate(arr_words):
        if words in tokens.keys():
            arr_vector[words] += 1

    vector = []
    #guardar los valores en un arreglo.
    vector = (list(arr_vector.values()))

    return np.asarray(vector)


