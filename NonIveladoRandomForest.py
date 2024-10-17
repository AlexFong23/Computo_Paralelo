import itertools
import multiprocess
import pandas as pd
import time

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

# Parámetros para Random Forest
param_grid_rf = {
    'n_estimators': [x for x in range(10, 101)],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'min_samples_split': [2, 5, 10] 
}

# Generar combinaciones para Random Forest
keys_rf, values_rf = zip(*param_grid_rf.items())
combinations_rf = [dict(zip(keys_rf, v)) for v in itertools.product(*values_rf)]


# Función a paralelizar
def evaluate_set(hyperparameter_set, lock, return_dict):
    """
    Evaluate a set of hyperparameters
    Args:
    hyperparameter_set: a list with the set of hyperparameters to be evaluated
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from ucimlrepo import fetch_ucirepo

    # Fetch dataset 
    dataset = fetch_ucirepo(id=544) 

    # Data (as pandas DataFrames) 
    X = dataset.data.features 
    y_y = dataset.data.targets 

    # Convertir X y y a arreglos de NumPy
    y = y_y.to_numpy().ravel()  

    # Esto convierte las columnas categóricas en variables numéricas
    X = pd.get_dummies(X)  

    # se particiona el conjunto en 80-20 para la evaluación
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y, 
                                                        test_size=0.20)
    for s in hyperparameter_set:
        clf = RandomForestClassifier()
        clf.set_params(n_estimators=s['n_estimators'], 
                       criterion=s['criterion'],
                       min_samples_split=s['min_samples_split'])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Exclusión mutua para guardar resultados
        lock.acquire()
        return_dict[str(s)] = accuracy  # Guardar el resultado en el diccionario compartido
        lock.release()


if __name__ == '__main__':
    # Ahora evaluaremos con más hilos
    threads = []
    N_THREADS = 11
    splits = nivelacion_cargas(combinations_rf, N_THREADS)
    lock = multiprocess.Lock()
    manager = multiprocess.Manager()
    return_dict = manager.dict()  # Diccionario para almacenar resultados

    log_file = 'NaivRandomForestResult.txt'  # Nombre del archivo de log
    with open(log_file, 'w') as f:  # Crear o vaciar el archivo antes de iniciar
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
