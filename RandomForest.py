import itertools
#from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocess
import time 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# método para hacer nivelación de cargas
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
    'n_estimators': [x for x in range(10,101)],
    'criterion': ['gini', 'entropy', 'log_loss']
}

# Generar combinaciones para Random Forest
keys_rf, values_rf = zip(*param_grid_rf.items())
combinations_rf = [dict(zip(keys_rf, v)) for v in itertools.product(*values_rf)]


# Función a paralelizar
def evaluate_set(hyperparameter_set, lock):
    """
    Evaluate a set of hyperparameters
    Args:
    hyperparameter_set: a list with the set of hyperparameters to be evaluated
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    # We load the dataset, here we use 80-20 for training and testing splits
    iris=datasets.load_iris()
    X=iris.data
    y=iris.target
    # se particiona el conjunto en 80-20 para la evaluación
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y, 
                                                        test_size=0.20)
    for s in hyperparameter_set:
        clf=RandomForestClassifier()
        #print(s)
        clf.set_params(n_estimators=s['n_estimators'], criterion=s['criterion'])
        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        # Exclusión mutua
        lock.acquire()
        print('Accuracy en el proceso:',accuracy_score(y_test,y_pred))
        lock.release()

if __name__=='__main__':
    # Now we will evaluated with more threads
    threads=[]
    N_THREADS=1
    splits=nivelacion_cargas(combinations_rf, N_THREADS)
    lock=multiprocess.Lock()
    for i in range(N_THREADS):
        # Se generan los hilos de procesamiento
        threads.append(multiprocess.Process(target=evaluate_set, args=(splits[i], lock)))


    start_time = time.perf_counter()
    # Se lanzan a ejecución
    for thread in threads:
        thread.start()

    # y se espera a que todos terminen
    for thread in threads:
        thread.join()
                
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")