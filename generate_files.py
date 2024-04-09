import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Nombre de los archivos de datos
data_path = "movielens1M.csv"
movies_path = "movies_data.csv"

# Cargar los archivos de datos
ratings = pd.read_csv(
    data_path, sep=":", names=["user_id", "item_id", "rating", "timestamp"]
)
movies = pd.read_csv(
    movies_path, sep="::", names=["item_id", "title", "genres"], engine="python"
)
# Calcular la matriz de co-ocurrencia de películas (matriz de correlación)
co_ocurrence_matrix = pd.pivot_table(
    ratings, values="rating", index="user_id", columns="item_id", fill_value=0
)
# Obtener un mapa para los identificadores de las películas
item_id_map = sorted(ratings["item_id"].unique())

# Total de resultados a mostrar
N = 5


def create_file():
    start = time.time()
    similarities = cosine_similarity(co_ocurrence_matrix.T)

    with open("similarities_cos.bin", "wb") as f:
        similarities.tofile(f)
    print(f"Time wasted on save {time.time() - start}")

    return similarities.shape[1]


def seek_item(pos: int, n_elemens):
    start = time.time()

    item_size = np.dtype(np.float64).itemsize

    with open("similarities_cos.bin", "rb") as f:
        f.seek(pos * item_size * n_elemens)
        data = f.read(item_size * n_elemens)
        x = np.frombuffer(data, dtype=np.float64)
        # print(x)

    print(f"Time wasted on reading {time.time() - start}")

    return x


def create_uniq_items_id():
    start = time.time()
    item_id_map = np.array(sorted(ratings["item_id"].unique()), dtype=np.int16)
    with open("item_id_map.bin", "wb") as f:
        print(item_id_map.shape)
        item_id_map.tofile(f)
    print(f"Time wasted on save {time.time() - start}")

    return item_id_map.shape[0]


def get_uniq_items_id(n_data: int):
    start = time.time()
    item_size = np.dtype(np.int16).itemsize
    with open("item_id_map.bin", "rb") as f:
        data = f.read(n_data * item_size)
        x = np.frombuffer(data, dtype=np.int16)
        print(x.shape)

    print(f"Time wasted on save {time.time() - start}")

    return list(x)


# num of cols
item_size = create_file()
print(item_size)
item_size_ids = create_uniq_items_id()
seek_item(2, item_size)
get_uniq_items_id(item_size_ids)
