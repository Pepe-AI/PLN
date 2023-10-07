# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XYR2J07J69YuD-XTG7hUfnFKeE4Y-JFv
"""

#!python -m spacy download es_core_news_sm
#!pip install datasets
#!pip install sumy

import nltk
from datasets import load_dataset
import spacy
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import numpy as np
from tensorflow.keras.utils import to_categorical
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import gensim
from gensim import corpora
from gensim.models import LsiModel
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, Input
from tensorflow.keras.models import Model

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))
nlp = spacy.load('es_core_news_sm')
stemmer = SnowballStemmer("spanish")

def eda(df):
    #dimensiones
    filas, columnas = df.shape
    print("Num.filas: ",filas, "Num.columnas: ", columnas)


    text_data = df['text']
    text_data = text_data[0:2]
    print("\nContenido detallado de la noticia, tipo texto\n","Tipo de dato:",type(text_data[0]))

    summary_data = df['summary']
    summary_data = summary_data[0:2]
    print("\nContenido resuumido de la noticia, tipo texto\n","Tipo de dato:",type(summary_data[0]))

    topic_data = df['topic']
    topic_data = topic_data[0:2]
    print("\nnombre de la revista que publica la noticia\n","Tipo de dato:",type(topic_data[0]))


    url_data = df['url']
    url_data = url_data[0:2]
    print("\nURL de la revista\n","Tipo de dato:",type(url_data[0]))


    title_data = df['title']
    title_data = title_data[0:2]
    print("\nTitulo de la noticia\n","Tipo de dato:",type(title_data[0]))


    date_data = df['date']
    date_data = date_data[0:2]
    print("\nFecha de publicación de la noticia\n","Tipo de dato:",type(date_data[0]))


def frecuency_term(df):
    text_data = df['text']
    text_data = text_data[0:2]


    #terminos mas frecuentes y menos frecuentes
    palabras = word_tokenize(text_data[0])
    fdist = FreqDist(palabras)


    # Imprime las 10 palabras más comunes
    print("Las palabras más comunes son:")
    for palabra, frecuencia in fdist.most_common(5):
        print(f"{palabra}: {frecuencia}")


    # Imprime las palabras menos comunes
    print("Las palabras menos comunes son:")
    for palabra, frecuencia in list(fdist.items())[-5:]:  # ajusta el número según la cantidad que quieras mostrar
        print(f"{palabra}: {frecuencia}")


def preprocess_text(text):
    text = re.sub(r'(https?\S*|http?\S*|www\S*|\d+|\[[^]]*\]|[^\w\s])', '', text.lower())
    if not text:
        return ''
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if not w in stop_words]
    filtered_text =" ".join(filtered_text)

    doc = nlp(filtered_text)
    lemmatized = [token.lemma_ for token in doc]

    stemmed_words = [stemmer.stem(word) for word in lemmatized]
    return stemmed_words

def word_to_index_mapping(tokenized_text):
    word_index = {}
    for sentence in tokenized_text:
        for word in sentence:
            if word not in word_index:
                word_index[word] = len(word_index) + 1
    return word_index

def convert_to_sequences(tokenized_text, word_index):
    return [[word_index[word] for word in sentence] for sentence in tokenized_text]



dataset = load_dataset('mlsum', 'es')
df = dataset['train']

eda = eda(df)
frecuency = frecuency_term(df)

#columnas indispensables para el programa
X = df['text']
X = X[0]

y = df["summary"]
y = y[0]
y_tokenize = word_tokenize(y)

preprocessed = preprocess_text(X)
preprocessed_y = preprocess_text(y)

word_index_X = word_to_index_mapping([preprocessed])
word_index_Y = word_to_index_mapping([preprocessed_y])

data_X = convert_to_sequences([preprocessed], word_index_X)
data_Y = convert_to_sequences([preprocessed_y], word_index_Y)

# Tamaños de las secuencias de entrada y salida
max_seq_length_X = max(len(s) for s in data_X)
max_seq_length_Y = max(len(s) for s in data_Y)

# Número de palabras únicas en los textos de entrada y resumen
num_words_X = len(word_index_X) + 1
num_words_Y = len(word_index_Y) + 1

embedding_dim = 50

# Definición del encoder
encoder_inputs = Input(shape=(max_seq_length_X,))
enc_emb = Embedding(num_words_X, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(embedding_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Definición del decoder
decoder_inputs = Input(shape=(max_seq_length_Y-1,))
dec_emb_layer = Embedding(num_words_Y, embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(embedding_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = TimeDistributed(Dense(num_words_Y, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outputs)

# Creación del modelo
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

data_X = np.array(data_X)
data_Y = np.array(data_Y)
decoder_input_data = data_Y[:,:-1]
decoder_target_data = to_categorical(data_Y[:,1:], num_classes=num_words_Y)


model.fit([np.array(data_X), np.array(data_Y)[:,:-1]], to_categorical(np.array(data_Y)[:,1:], num_classes=num_words_Y), epochs=100)

print(model)