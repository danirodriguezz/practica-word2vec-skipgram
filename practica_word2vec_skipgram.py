import random
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def fijar_semilla(semilla=42):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)

fijar_semilla(42)


frases = []
with open("dataset_word2vec.txt", encoding="utf-8") as f:
    for linea in f:
        linea = linea.strip().lower()
        if linea != "":
            frases.append(linea.split())

print("Número de frases:", len(frases))
print("Ejemplo:", frases[0])
print("Ejemplo:", frases[140])


conteo = {}
for frase in frases:
    for palabra in frase:
        if palabra in conteo:
            conteo[palabra] += 1
        else:
            conteo[palabra] = 1

vocabulario = sorted(conteo.keys())

palabra_a_indice = {}
for i, palabra in enumerate(vocabulario):
    palabra_a_indice[palabra] = i

indice_a_palabra = vocabulario

print("Total de tokens:", sum(conteo.values()))
print("Tamaño del vocabulario:", len(vocabulario))


def generar_pares(frases, ventana):
    pares = []
    for frase in frases:
        for i in range(len(frase)):
            centro = frase[i]
            inicio = max(0, i - ventana)
            fin = min(len(frase), i + ventana + 1)
            for j in range(inicio, fin):
                if j != i:
                    pares.append((centro, frase[j]))
    return pares

for par in generar_pares([frases[0]], ventana=2):
    print(par)


VENTANA = 2
pares = generar_pares(frases, VENTANA)
print("Número de pares (centro, contexto):", len(pares))


class SkipGram(nn.Module):
    def __init__(self, tam_vocab, dim_embedding):
        super().__init__()
        self.embeddings = nn.Embedding(tam_vocab, dim_embedding)
        self.salida = nn.Linear(dim_embedding, tam_vocab)

    def forward(self, centro):
        vector = self.embeddings(centro)
        puntuaciones = self.salida(vector)
        return puntuaciones


def entrenar(ventana, dim_embedding, epocas, lr=0.01, tam_lote=64, semilla=42, mostrar=True):
    fijar_semilla(semilla)

    pares = generar_pares(frases, ventana)
    centros = torch.tensor([palabra_a_indice[c] for c, _ in pares])
    contextos = torch.tensor([palabra_a_indice[x] for _, x in pares])

    modelo = SkipGram(len(vocabulario), dim_embedding)
    funcion_perdida = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)

    historial = []
    for epoca in range(epocas):
        orden = torch.randperm(len(pares))
        perdida_total = 0

        for k in range(0, len(pares), tam_lote):
            lote = orden[k:k + tam_lote]
            x = centros[lote]
            y = contextos[lote]

            optimizador.zero_grad()
            prediccion = modelo(x)
            perdida = funcion_perdida(prediccion, y)
            perdida.backward()
            optimizador.step()

            perdida_total += perdida.item() * len(lote)

        perdida_media = perdida_total / len(pares)
        historial.append(perdida_media)
        if mostrar and (epoca + 1) % 10 == 0:
            print(f"Época {epoca + 1:3d} | pérdida = {perdida_media:.4f}")

    return modelo, historial


DIM = 10
EPOCAS = 100

modelo, historial = entrenar(VENTANA, DIM, EPOCAS)


plt.figure(figsize=(7, 4))
plt.plot(range(1, EPOCAS + 1), historial)
plt.xlabel("Época")
plt.ylabel("Pérdida media")
plt.title("Evolución de la pérdida durante el entrenamiento")
plt.grid(True)
plt.show()


def obtener_embeddings(modelo):
    E = modelo.embeddings.weight.detach().numpy()
    normas = np.linalg.norm(E, axis=1, keepdims=True)
    return E / normas

def vecinos_mas_cercanos(E, palabra, k=5):
    vector = E[palabra_a_indice[palabra]]
    similitudes = E @ vector
    orden = np.argsort(-similitudes)
    resultado = []
    for i in orden:
        if indice_a_palabra[i] != palabra:
            resultado.append((indice_a_palabra[i], similitudes[i]))
        if len(resultado) == k:
            break
    return resultado

E = obtener_embeddings(modelo)
print("Forma de la matriz de embeddings:", E.shape)
print("Vector de 'perro':", np.round(E[palabra_a_indice["perro"]], 2))
print()
for palabra, sim in vecinos_mas_cercanos(E, "perro"):
    print(f"{palabra:10s} {sim:.3f}")


palabras_consulta = ["perro", "perra", "gato", "gata", "coche", "parís", "francia",
                     "madrid", "rey", "niño", "café", "mañana"]

filas = {}
for palabra in palabras_consulta:
    vecinos = vecinos_mas_cercanos(E, palabra, k=5)
    filas[palabra] = [f"{v} ({s:.2f})" for v, s in vecinos]

tabla_vecinos = pd.DataFrame.from_dict(filas, orient="index",
                                       columns=["1º", "2º", "3º", "4º", "5º"])
tabla_vecinos.to_csv("tabla_vecinos.csv")
tabla_vecinos


def analogia(E, a, b, c, k=3):
    vector = E[palabra_a_indice[b]] - E[palabra_a_indice[a]] + E[palabra_a_indice[c]]
    vector = vector / np.linalg.norm(vector)
    similitudes = E @ vector
    orden = np.argsort(-similitudes)
    resultado = []
    for i in orden:
        if indice_a_palabra[i] not in (a, b, c):
            resultado.append(indice_a_palabra[i])
        if len(resultado) == k:
            break
    return resultado

print("parís : francia :: madrid : ?  ->", analogia(E, "parís", "francia", "madrid"))


analogias = [
    ("parís", "francia", "madrid", "españa"),
    ("madrid", "españa", "roma", "italia"),
    ("perro", "perra", "gato", "gata"),
    ("niño", "niña", "padre", "madre"),
    ("rey", "reina", "padre", "madre"),
    ("abuelo", "abuela", "hermano", "hermana"),
    ("médico", "médica", "enfermero", "enfermera"),
    ("perro", "perra", "niño", "niña"),
    ("ingeniero", "ingeniera", "programador", "programadora"),
    ("el", "la", "perro", "perra"),
]

def evaluar_analogias(E, analogias):
    filas = []
    aciertos = 0
    for a, b, c, esperada in analogias:
        top3 = analogia(E, a, b, c, k=3)
        acierto = (top3[0] == esperada)
        aciertos += acierto
        filas.append([f"{a} : {b} :: {c} : ?", esperada, ", ".join(top3), "✔" if acierto else "✘"])
    tabla = pd.DataFrame(filas, columns=["Analogía", "Esperada", "Top-3 del modelo", "Acierto (top-1)"])
    return tabla, aciertos

tabla_analogias, aciertos = evaluar_analogias(E, analogias)
tabla_analogias.to_csv("tabla_analogias.csv", index=False)
print(f"Aciertos: {aciertos} de {len(analogias)}")
tabla_analogias


E_centrado = E - E.mean(axis=0)
U, S, Vt = np.linalg.svd(E_centrado)
E_2d = E_centrado @ Vt[:2].T

palabras_dibujo = ["perro", "perra", "gato", "gata", "niño", "niña", "padre", "madre",
                   "rey", "reina", "parís", "madrid", "roma", "francia", "españa", "italia",
                   "coche", "autobús", "café", "té", "leche", "agua"]

plt.figure(figsize=(9, 7))
for palabra in palabras_dibujo:
    x, y = E_2d[palabra_a_indice[palabra]]
    plt.scatter(x, y, color="steelblue")
    plt.annotate(palabra, (x, y), xytext=(4, 4), textcoords="offset points")
plt.title("Embeddings proyectados a 2D con PCA")
plt.grid(True)
plt.show()


def experimento(ventana, dim, epocas):
    modelo_exp, hist = entrenar(ventana, dim, epocas, mostrar=False)
    E_exp = obtener_embeddings(modelo_exp)
    _, aciertos = evaluar_analogias(E_exp, analogias)
    vecinos = [v for v, _ in vecinos_mas_cercanos(E_exp, "perro", k=3)]
    return [ventana, dim, epocas, round(hist[-1], 3), f"{aciertos}/{len(analogias)}", ", ".join(vecinos)]

columnas = ["Ventana", "Dimensión", "Épocas", "Pérdida final", "Analogías", "Vecinos de 'perro'"]


filas = []
for dim in [2, 5, 10, 20, 50, 100]:
    filas.append(experimento(ventana=2, dim=dim, epocas=100))
pd.DataFrame(filas, columns=columnas)


filas = []
for ventana in [1, 2, 3, 5]:
    filas.append(experimento(ventana=ventana, dim=10, epocas=100))
pd.DataFrame(filas, columns=columnas)


filas = []
for epocas in [10, 30, 100, 200]:
    filas.append(experimento(ventana=2, dim=10, epocas=epocas))
pd.DataFrame(filas, columns=columnas)


filas = []
for semilla in [0, 1, 2, 3, 42]:
    fila = [semilla]
    for dim in [10]:
        modelo_s, _ = entrenar(2, dim, 100, semilla=semilla, mostrar=False)
        _, aciertos = evaluar_analogias(obtener_embeddings(modelo_s), analogias)
        fila.append(f"{aciertos}/{len(analogias)}")
    filas.append(fila)
pd.DataFrame(filas, columns=["Semilla", "Analogías (dim = 10)"])
