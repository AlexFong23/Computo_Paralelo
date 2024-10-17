import itertools
import multiprocess
import time
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo
import pandas as pd


# Método para hacer nivelación de cargas
def nivelacion_cargas(D, n_p):
    s = len(D)%n_p
    n_D = D[:s]
    t = int((len(D)-s)/n_p)
    out=[]
    temp=[]
    for i in D[s:]:
        temp.append(i)
        if len(temp)==t:
            out.append(temp)
            temp = []
    for i in range(len(n_D)):
        out[i].append(n_D[i])
    return out

# Parámetros para SVM
param_grid_svm = {
    'C': np.logspace(-2, 3, num=20),  # Generar 20 valores de C en escala logarítmica entre 0.01 y 1000
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']  # Más valores para gamma
}

# Generar combinaciones para SVM
keys_svm, values_svm = zip(*param_grid_svm.items())
combinations_svm = [dict(zip(keys_svm, v)) for v in itertools.product(*values_svm)]

# Función a paralelizar
def evaluate_set(hyperparameter_set, return_dict, lock):
   
    # Cargamos el conjunto de datos y lo dividimos
    dataset = fetch_ucirepo(id=544) 
    X = dataset.data.features 
    y_y = dataset.data.targets

    y = y_y.to_numpy().ravel() 
    X = pd.get_dummies(X)

    # Se particiona el conjunto en 80-20 para la evaluación
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20)


    for s in hyperparameter_set:
        clf = SVC()
        # Establecemos los hiperparámetros
        clf.set_params(C=s['C'], kernel=s['kernel'], gamma=s['gamma'], random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        # Almacenamos el resultado en el diccionario compartido
        lock.acquire()
        return_dict[str(s)] = acc
        lock.release()

if __name__ == '__main__':

    # Ahora evaluamos con múltiples procesos
    processes = []
    N_THREADS = 7
    splits = nivelacion_cargas(combinations_svm, N_THREADS)
    lock = multiprocess.Lock()
    manager = multiprocess.Manager()
    return_dict = manager.dict()

    # Archivo donde se guardarán los resultados
    output_file = 'SVMresults.txt'

    for i in range(N_THREADS):
        # Se generan los procesos
        p = multiprocess.Process(target=evaluate_set, args=(splits[i], return_dict, lock))
        processes.append(p)

    start_time = time.perf_counter()

    # Se lanzan a ejecución
    for process in processes:
        process.start()

    # Se espera a que todos terminen
    for process in processes:
        process.join()

    finish_time = time.perf_counter()

    # Encontrar los mejores hiperparámetros
    best_params = max(return_dict.items(), key=lambda x: x[1])

    # Guardar resultados en un archivo txt
    with open(output_file, 'w') as f:
        f.write(f"Program finished in {finish_time - start_time} seconds\n")
        f.write(f"Best parameters: {best_params[0]}, Accuracy: {best_params[1]}\n\n")
        f.write("All Results:\n")
        for k, v in return_dict.items():
            f.write(f"Params: {k}, Accuracy: {v}\n")
    
    print(f"Results saved to {output_file}")