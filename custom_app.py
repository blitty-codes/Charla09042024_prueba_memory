import os
import time

import numpy as np
import pandas as pd
import psutil
from flask import Flask, request

# Configuración de Flask
app = Flask(__name__)

# Nombre de los archivos de datos
data_path = "movielens1M.csv"
movies_path = "movies_data.csv"

movies = pd.read_csv(
    movies_path, sep="::", names=["item_id", "title", "genres"], engine="python"
)

# Total de resultados a mostrar
N = 5
ITEMS_SIZE = 3706


def seek_item(pos: int, n_elemens):
    start = time.time()

    item_size = np.dtype(np.float64).itemsize

    with open("similarities_cos.bin", "rb") as f:
        f.seek(pos * item_size * n_elemens)
        data = f.read(item_size * n_elemens)
        x = np.frombuffer(data, dtype=np.float64)

    print(f"Time wasted on reading one row of cosine {time.time() - start}s")
    return x


def get_uniq_items_id(n_data: int):
    start = time.time()
    item_size = np.dtype(np.int16).itemsize
    with open("item_id_map.bin", "rb") as f:
        data = f.read(n_data * item_size)
        x = np.frombuffer(data, dtype=np.int16)

    print(f"Time wasted on reading unique items {time.time() - start}s")

    return list(x)


@app.route("/", methods=["POST"])
def barra():
    raise Exception()


##############################################
############### Item to Item #################
##############################################

################## Síncrono #################


@app.route("/cosine", methods=["POST"])
def cosine():
    """
    Calculate the similarity of similar movies
    """
    if request.is_json:
        # Obtener datos JSON de la solicitud
        data = request.get_json()

        item_id_map = get_uniq_items_id(ITEMS_SIZE)
        pos = item_id_map.index(data["movie"])

        # Obtener cos del item
        item_similarity = seek_item(pos, ITEMS_SIZE)

        # Obtener los resultados más altos del cálculo del coseno
        # print(item_similarity)
        similar_indices = item_similarity.argsort()[-N - 1 : -1][::-1]
        similar_similarities = item_similarity[similar_indices]

        # Transformar el tipo de dato para poder convertirlo a JSON
        similar_similarities = similar_similarities.astype(float)
        similar_indices = similar_indices.astype(int)

        # Crear el JSON en base a los resultados
        similar_movies_json = []
        for index, similarity in zip(similar_indices, similar_similarities):
            title = movies.loc[movies["item_id"] == item_id_map[index], "title"].values[
                0
            ]
            similar_movie_json = {
                "movie_id": int(item_id_map[index]),
                "similarity": float(similarity),
                "title": title,
            }
            similar_movies_json.append(similar_movie_json)

        # Obtener el consumo de memoria del proceso ejecutado
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024**2

        print(f"RAM : {memory_usage}")
        # Crear el diccionario final con el formato JSON
        title = movies.loc[movies["item_id"] == data["movie"], "title"].values[0]
        result_json = {
            "movie_id": data["movie"],
            "title": title,
            "similarities": similar_movies_json,
            "RAM": memory_usage,
        }

        return result_json


##############################################
#################### Main ####################
##############################################

if __name__ == "__main__":
    app.run(host="localhost", port=9090, debug=False)
