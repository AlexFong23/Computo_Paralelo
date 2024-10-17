import itertools
from multiprocessing import Process, Lock, Manager
import time
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split

def nivelacion_cargas(D, n_p):
    """
    Distribuye equitativamente las combinaciones de hiperparámetros entre los procesos.

    Args:
    D: Lista de combinaciones de hiperparámetros.
    n_p: Número de procesos.

    Returns:
    out: Lista de listas, donde cada sublista contiene las combinaciones asignadas a un proceso.
    """
    s = len(D) % n_p
    n_D = D[:s]
    t = int((len(D) - s) / n_p)
    out = []
    temp = []
    for i in D[s:]:
        temp.append(i)
        if len(temp) == t:
            out.append(temp)
            temp = []
    for i in range(len(n_D)):
        out[i].append(n_D[i])
    return out

def evaluate_set(hyperparameter_set, X_train, X_test, y_train, y_test, return_dict, lock):
    """
    Evalúa un conjunto de hiperparámetros para SVM y almacena los resultados.

    Args:
    hyperparameter_set: Lista de combinaciones de hiperparámetros a evaluar.
    X_train, X_test, y_train, y_test: Conjuntos de datos de entrenamiento y prueba.
    return_dict: Diccionario compartido para almacenar los resultados.
    lock: Bloqueo para asegurar acceso exclusivo al diccionario compartido.
    """
    for s in hyperparameter_set:
        clf = SVC(
            C=s['C'],
            kernel=s['kernel'],
            gamma=s['gamma'],
            random_state=42
        )
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        lock.acquire()
        print(f"Evaluated params: {s}, Accuracy: {score}")
        lock.release()
        return_dict[str(s)] = score

if __name__ == '__main__':
    # Definimos la rejilla de hiperparámetros para SVM
    param_grid_svm = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }

    # Generamos todas las combinaciones posibles de hiperparámetros
    keys_svm, values_svm = zip(*param_grid_svm.items())
    combinations_svm = [dict(zip(keys_svm, v)) for v in itertools.product(*values_svm)]
    
    N_THREADS = 7  
    splits = nivelacion_cargas(combinations_svm, N_THREADS)
    lock = Lock()
    manager = Manager()
    return_dict = manager.dict()

    # Cargamos el conjunto de datos y realizamos la división fuera de los procesos
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.20, random_state=42
    )

    processes = []
    start_time = time.perf_counter()

    for i in range(N_THREADS):
        p = Process(target=evaluate_set, args=(
            splits[i], X_train, X_test, y_train, y_test, return_dict, lock
        ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")

    # Análisis de resultados
    best_params = max(return_dict.items(), key=lambda x: x[1])
    print(f"Best parameters: {best_params[0]}, Accuracy: {best_params[1]}")
