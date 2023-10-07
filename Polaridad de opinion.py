import nltk
from sklearn.utils import resample
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import re
from sklearn.linear_model import LogisticRegression
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#------------------------------------------------------------------------#
nltk.download('punkt')
nltk.download("stopwords")
nltk.download("wordnet")
#------------------------------------------------------------------------#

#------------------------------------------------------------------------#
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
#------------------------------------------------------------------------#



def eda():
    #dimensiones
    filas, columnas = data.shape
    print("filas: ",filas, "columnas: ", columnas)
    #caracterisitcas
    print("id sirve como identificador y su tipo de datos es(numerica):  ", data["Id"].dtype)
    print("Productid sirve como clave de registro de un prodcuto(categorica):  ", data["ProductId"].dtype)
    print("UserID sirve como clave de registro de un usuario(categorica):  ", data["UserId"].dtype)
    print("ProfileName sirve como nombre de un usuario(categorica):  ", data["ProfileName"].dtype)
    print("HelpfulnessNumerator(numerica):  ", data["HelpfulnessNumerator"].dtype)
    print("HelpfulnessDenominator(numerica):  ", data["HelpfulnessDenominator"].dtype)
    print("Score calificaicon dada por el usuario(numerica):  ", data["Score"].dtype)
    print("Time tiempo en el que usuario hizo el comentario(numerica):  ", data["Time"].dtype)
    print("Summary(categorica) titulo o encabezado de la opinion del usuario:  ", data["Summary"].dtype)
    print("Text la opinion escrita por le usuario(categorica):  ", data["Text"].dtype)

def standard_distribution(X):
    plt.hist(X["Score"], bins=10)
    plt.xlabel("Valores")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de la columna " + "score")
    plt.show()
    desviacion_estandar = X["Score"].std()
    return desviacion_estandar

def preprocess_text(text):
    text = re.sub(r'\d+|http\S+|www\S+|https\S+|<.*?>|\d+|\[[^]]*\]|[^\w\s]', '', text.lower())
    if not text:
        return ''
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    stemmed_words = [stemmer.stem(word) for word in lemmatized_tokens]
    return " ".join(stemmed_words)

def data_balance():
    #identificar la clase dominante para ajustarla a las demas
    clase_dominante = data[data["Score"] == 1]
    clase_minoritaria_1 = data[data["Score"] == -1]
    clase_minoritaria_2 = data[data["Score"] == 0]

    # Definir la cantidad de muestras deseada
    n_samples = 400  # para clase1 (positivos)
    clase1_resampled = resample(clase_dominante, replace=False, n_samples=n_samples, random_state=42)

    n_samples = 400  # para clase2 (negativos)
    clase2_resampled = resample(clase_minoritaria_1, replace=False, n_samples=n_samples, random_state=42)

    n_samples = 400  # para clase3 (neutrales)
    clase3_resampled = resample(clase_minoritaria_2, replace=False, n_samples=n_samples, random_state=42)
    
    return clase1_resampled, clase2_resampled, clase3_resampled


#exploracion del dataset
data = pd.read_csv("E:\pln aplicaciones\practica 3\Reviews.csv")
eda = eda()
stand = standard_distribution(data)


#Eliminacion de columnas innecesarias
data = data.drop(data.columns[0:6], axis=1)
data = data.drop("Time", axis=1)


#mapear los valores en -, 0, + 
data["Score"] = data["Score"].map({1: -1, 2: -1, 3: 0, 4: 1, 5: 1})
stand_v2 = standard_distribution(data)


clase1_resampled,clase2_resampled,clase3_resampled = data_balance()


# Combinar los ejemplos submuestreados de la clase dominante con las clases minoritarias
data_sub = pd.concat([clase1_resampled, clase2_resampled, clase3_resampled])


# Guardar los cambios en un nuevo archivo CSV
nuevo_archivo = "Reviews_balanced.csv"
data_sub.to_csv(nuevo_archivo, index=False)


#mapear las clases a lo solicitado en la practica con negativo, neutral y positivo 
#data_sub["Score"] = data["Score"].map({-1: "negativo", -1: "negativo", 0: "neutral", 1: "positivo", 1: "positivo"})
stand_v3 = standard_distribution(data_sub)


#concatenar las columnas summary y text que son los datos caracteristicos(X) en una sola columna
data_sub['Summary'] = data_sub['Summary'].astype(str)
data_sub['Text'] = data_sub['Text'].astype(str)
data_sub['columna_concatenada'] = data_sub['Summary'] + ' ' + data_sub['Text']


#limpiar
data_sub["processed"] = data_sub["columna_concatenada"].apply(preprocess_text)

y=data_sub["Score"]

# Aplicar TfidfVectorizer
#vectorizer = TfidfVectorizer()
#X = vectorizer.fit_transform(data_sub["processed"]).toarray()


# Aplicar one-hot-encoded
X = pd.get_dummies(data_sub['processed'])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 

# ----------Entrenar un modelo con k-fold cross-validation----------------------------#
log_reg_classifier = LogisticRegression(random_state=42)
log_reg_scores = cross_val_score(log_reg_classifier, X, y, cv=5)

svc  = SVC(random_state=42)
svc_scores = cross_val_score(svc, X, y, cv=5)

decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree_scores = cross_val_score(decision_tree, X, y, cv=5)
# -----------------------------------------------------------------------------------#

print("---------------Accuracy (Regresion logistica):-----------------")
print(log_reg_scores.mean())

print("----------------Accuracy (Support Vector Machine):------------------")
print(svc_scores.mean())

print("-------------Accuracy (Decision Trees):--------------")
print(decision_tree_scores.mean())
# -----------------------------------------------------------------------------------#








""""stat, p = shapiro(data["Score"])

alpha = 0.05  # Nivel de significancia

if p > alpha:
    print("La columna", "Score", "sigue una distribución normal")
else:
    print("La columna", "Score", "no sigue una distribución normal")
"""