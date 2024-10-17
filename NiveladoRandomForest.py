import math
import itertools
import multiprocess
import time 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

# Función para resolver ecuación cuadrática
def solucion_positiva_ecuacion_cuadratica(a, b, c):
    discriminante = b**2 - 4*a*c
    if discriminante < 0:
        return "No hay soluciones reales"

    # Calcular ambas soluciones
    raiz_discriminante = math.sqrt(discriminante)
    x1 = (-b + raiz_discriminante) / (2*a)
    x2 = (-b - raiz_discriminante) / (2*a)

    # Devolver solo la solución positiva
    if x1 >= 0:
        return x1
    elif x2 >= 0:
        return x2
    else:
        return "No hay soluciones positivas"

# Función para saber el número total de árboles
def calcular_total_num_trees(minor_num_trees, max_num_trees):
    return sum(range(minor_num_trees, max_num_trees + 1))

# Función que almacena los hiperparámetros
def crear_param_grid_rf(minor_num_trees, s):
    return {
        'n_estimators': [x for x in range(minor_num_trees, s + 1)],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'min_samples_split': [2, 5, 10]
    }

# Función  para generar las combinaciones
def generar_combinaciones(param_grid):
    keys, values = zip(*param_grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

# Función para crear las listas que se enviaran a los procesadores
def calcular_combinaciones_random_forest(minor_num_trees, max_num_trees, N_THREADS):
    
    total_num_trees = calcular_total_num_trees(minor_num_trees, max_num_trees)
    #print("Total de arboles:", total_num_trees)

    trees_each_processor = total_num_trees / N_THREADS
    #print("Arboles aproximados por procesador:", trees_each_processor)

    out = []
    i = 1
    inferior_limit = minor_num_trees

    while True:
        c = -(2 * trees_each_processor + (inferior_limit**2) - inferior_limit)

        superior_limit = round(solucion_positiva_ecuacion_cuadratica(1, 1, c))

        if i == (N_THREADS):
            #print(f'Rango de [{inferior_limit},{max_num_trees}] con un total de {sum(range(inferior_limit, max_num_trees + 1))} arboles')
            param_grid_rf = crear_param_grid_rf(inferior_limit, max_num_trees)
            combinations_rf = generar_combinaciones(param_grid_rf)
            out.append(combinations_rf)
            break

        #print(f'Rango de [{inferior_limit},{superior_limit}] con un total de {sum(range(inferior_limit, superior_limit + 1))} arboles')

        # Generar parámetros y combinaciones para Random Forest
        param_grid_rf = crear_param_grid_rf(inferior_limit, superior_limit)
        combinations_rf = generar_combinaciones(param_grid_rf)
        out.append(combinations_rf)

        inferior_limit = superior_limit + 1

        i += 1

    return out

# Función a paralelizar
def evaluate_set(hyperparameter_set, lock, return_dict):

    # Obesity dataset 
    dataset = fetch_ucirepo(id=544) 

    # Data (as pandas DataFrames) 
    X = dataset.data.features 
    y_y = dataset.data.targets 
    y = y_y.to_numpy().ravel()  

    # Esto convierte las columnas categóricas en variables numéricas    
    X = pd.get_dummies(X)  

    # Dividir datos 
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20)

    for idx, s in enumerate(hyperparameter_set):
        clf = RandomForestClassifier()
        clf.set_params(n_estimators=s['n_estimators'], 
                       criterion=s['criterion'],
                       min_samples_split=s['min_samples_split'])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Exclusión mutua
        accuracy = accuracy_score(y_test, y_pred)
        lock.acquire()
        return_dict[str(s)] = accuracy  
        lock.release()

if __name__ == '__main__':
    # Ahora evaluaremos con más hilos
    threads = []
    N_THREADS = 11
    minor_num_trees = 10
    max_num_trees = 100
    splits = calcular_combinaciones_random_forest(minor_num_trees, max_num_trees , N_THREADS)
    lock = multiprocess.Lock()
    manager = multiprocess.Manager()
    return_dict = manager.dict()  # Diccionario para almacenar resultados

    log_file = 'NiveladoRandomForestResult.txt'  
    with open(log_file, 'w') as f:  
        f.write("Hyperparameter Evaluation Log\n\n")

    for i in range(N_THREADS):
        # Se generan los hilos de procesamiento
        threads.append(multiprocess.Process(target=evaluate_set, args=(splits[i], lock, return_dict)))

    start_time = time.perf_counter()
    
    # Se lanzan a ejecución
    for thread in threads:
        thread.start()

    # y se espera a que todos terminen
    for thread in threads:
        thread.join()
                
    finish_time = time.perf_counter()

    # Encontrar los mejores hiperparámetros
    best_params = max(return_dict.items(), key=lambda x: x[1])

    # Guardar resultados en el archivo txt
    with open(log_file, 'a') as f:
        f.write(f"Program finished in {finish_time - start_time} seconds\n")
        f.write(f"Best parameters: {best_params[0]}, Accuracy: {best_params[1]}\n\n")
        f.write("All Results:\n")
        for k, v in return_dict.items():
            f.write(f"Params: {k}, Accuracy: {v}\n")

    print(f"Results saved to {log_file}")
    print(f"Program finished in {finish_time - start_time} seconds")
