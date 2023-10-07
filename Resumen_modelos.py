# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 21:14:33 2023

@author: pepeh
"""

import nltk
from datasets import load_dataset
import spacy
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from transformers import pipeline
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LsiModel
from collections import defaultdict
from nltk.tokenize import sent_tokenize

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
    #filtered_text =" ".join(filtered_text)
    #doc = nlp(filtered_text)
    #lemmatized = [token.lemma_ for token in doc]
    #stemmed_words = [stemmer.stem(word) for word in lemmatized]

    return filtered_text

def term_frequency_summarizer(text, summary_length):

    # Calcula la frecuencia de los términos
    words = word_tokenize(text)
    freq_table = defaultdict(int)
    for word in words:
        if word not in stop_words:
            freq_table[word] += 1

    # Normaliza las frecuencias de los términos
    max_freq = max(freq_table.values())
    for word in freq_table.keys():
        freq_table[word] = freq_table[word]/max_freq

    # Asigna una puntuación a cada frase en función de las frecuencias de sus términos
    sentences = sent_tokenize(text)
    sentence_scores = defaultdict(int)
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in freq_table.keys():
                sentence_scores[sent] += freq_table[word]

    # Ordena las frases por puntuación y selecciona las frases más importantes para el resumen
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    summary = ' '.join(summary_sentences[:summary_length])

    return summary


dataset = load_dataset('mlsum', 'es')
df = dataset['train']

eda = eda(df)
frecuency = frecuency_term(df)

#columnas indispensables para el programa
X = df['text']
X = X[0]

y = df["summary"]
y = y[0]

preprocessed = preprocess_text(X)
preprocessed_text = ' '.join(preprocessed)

#------------------------------Frecuencia de términos normalizada-------------------------------------#
summary = term_frequency_summarizer(preprocessed_text, summary_length=8)
#print("resumen:\n",summary)
#------------------------------Frecuencia de términos normalizada-------------------------------------#



#-----------------------------TextRank--------------------------------------#
#preprocessed = " ".join(preprocessed)
parser = PlaintextParser.from_string(preprocessed_text, Tokenizer("spanish"))

# Crea un resumidor de TextRank
summarizer = TextRankSummarizer()

# Resumir el texto
# El segundo argumento determina el número de oraciones en el resumen
#summary_textrank = summarizer(parser.document, 5)
#for sentence in summary_textrank:
    #print(sentence)
#-----------------------------TextRank--------------------------------------#


#--------------------------------LSA-----------------------------------#
dictionary = corpora.Dictionary([preprocessed])

# Crear un corpus
corpus = [dictionary.doc2bow(preprocessed)]

# Crear el modelo LSA
lsamodel = LsiModel(corpus, num_topics=2, id2word=dictionary)

# Imprimir los temas
print(lsamodel.print_topics())

summarizer = pipeline("summarization")

text = " ".join(preprocessed)

summary = summarizer(text, max_length=50, min_length=25, do_sample=False)

print(summary[0]['summary_text'])
#--------------------------------LSA-----------------------------------#


