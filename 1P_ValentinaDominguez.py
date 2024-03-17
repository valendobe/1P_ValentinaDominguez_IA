# IMPORTACION DE LIBRERIAS
import pandas as pd
import numpy as np
import matplotlib as plt

# FUNCIONES A UTILIZAR
def regresion_manual(X, y): # para determinar los coeficientes de la regresion
    X = np.c_[np.ones((X.shape[0], 1)), X] # para agregar columna de unos a la matriz dato
    coeficientes = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) 
    return coeficientes 
def predecir(X, coeficientes): # para predecir los valores de "y"
    Xm = np.c_[np.ones((X.shape[0], 1)), X]  # para agregar columna de unos a la matriz dato
    return  Xm@coeficientes # el @ funciona como producto punto
def rmse(y_true, y_pred): # para calcular metricas de evaluacion
    error=y_true-y_pred
    return np.sqrt(np.mean((error) ** 2))
def r2F(y_true, y_pred): # coeficiente de determinacion
    numerador = ((y_true - y_pred) ** 2).sum()
    denominador = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - (numerador / denominador)
def ajustar_evaluar_modelo(X, y): # para ajustar modelo y evaluar
    coeficientes = regresion_manual(X, y)
    y_pred = predecir(X, coeficientes)
    r2_ =[r2F(y,y_pred)]#completar
    rmse_val = [rmse(y,y_pred)]#completar
    return coeficientes, y_pred, r2_, rmse_val

# RECIBIR  ENTRADA POR PARTE DEL USUARIO Y CARGA DE DATOS
opcion=int(input())
data = pd.read_csv('Mediciones.csv') # carga de datos en formato .csv

# OPCIONES DE OPERACION
if opcion==1: 
    num_filas,num_columnas=data.shape # guarda en las variables los num. de fila y columna
    print("Numero de filas: ", num_filas)
    print("Numero de columnas: ", num_columnas)
    caracteristicas = data.columns[0:7 and 9] # .columns para ver el indice de la columna
    objetivo = data.columns[7]
    print(caracteristicas)
    print(objetivo)
elif opcion==2: 
    X = data['VTI_F']
    y = data['Pasos']
    X = X[:-1] # para eliminar la ultima fila con componentes NaN
    y = y[:-1] # para eliminar la ultima fila con componentes NaN
    coef =[regresion_manual(X, y)]
    print(coef)
elif opcion==3: 
    X = data['VTI_F']
    y = data['Pasos']
    X = X[:-1] # para eliminar la ultima fila con componentes NaN
    y = y[:-1] # para eliminar la ultima fila con componentes NaN
    coef = regresion_manual(X, y)
    print( coef)
    y_pred = predecir(X,coef)
    r2_ = r2F(y, y_pred)
    rmse_val = rmse(y, y_pred)
    print(y[:3],  y_pred [:3]) 
    print(r2_,  rmse_val )
elif opcion==4: 
    X_todo = data['VTI_F']
    y = data['Pasos']
    X_todo = X_todo[:-1]
    y = y[:-1]
    coeficientes_todo, y_pred_todo, r2_todo, rmse_todo = ajustar_evaluar_modelo(X_todo, y)
    print(r2_todo, rmse_todo)
elif opcion==5:
    models = {
        'Modelo_1': ['VTI_F'],
        'Modelo_2': ['VTI_F', 'BPM'],
        'Modelo_3': ['VTI_F','PEEP'],
        'Modelo_4': ['VTI_F','PEEP','BPM'],
        'Modelo_5': ['VTI_F','PEEP','BPM','VTE_F'],
    }
    for nombre_modelo, lista_caracteristicas in models.items():
        X = data[lista_caracteristicas]
        y = data['Pasos']
        X = X[:-1]
        y = y[:-1]
        coeficientes, y_pred, r2, rmse_val = ajustar_evaluar_modelo(X, y)
        print(nombre_modelo,r2, rmse_val)
elif opcion==6:
    data = data[:-1]
    valores_peep_unicos = data['PEEP'].unique()
    valores_bpm_unicos = data['BPM'].unique()
    print(valores_peep_unicos)
    print(valores_bpm_unicos)
    predicciones_totales = []
    for peep in valores_peep_unicos:
        for bpm in valores_bpm_unicos:
            
            
            
            datos_subset = data[(data['PEEP'] == peep) & (data['BPM'] == bpm)] #completar el filtrado de datos, se deben filtrar los datos para cada para par de PEEP y BPM
            
            
            X_subset = datos_subset[['VTI_F']]
            y_subset = datos_subset['Pasos']
            coeficientes_subset, y_pred_subset, r2_subset, rmse_subset = ajustar_evaluar_modelo(X_subset, y_subset)
            print(peep, bpm, r2_subset, rmse_subset)
            predicciones_totales.append(y_pred_subset)
    predicciones_concatenadas = np.concatenate(predicciones_totales)
    y=data['Pasos']
    r2_global = r2F(y, predicciones_concatenadas)
    rmse_global = rmse(y, predicciones_concatenadas)
    print('Global', r2_global, rmse_global)

# GRAFICO
plt.xlabel('VTI Fluke') # para nombrar al eje X
plt.ylabel('Pasos') # para nombrar al eje Y
plt.ylim(16000, 32000)  # establecer el rango de x
plt.xlim(150, 650) # establecer el rango de x
# .scatter utilizado para que los resultados se representen como puntos
plt.scatter(data['VTI_F'][:6],predicciones_concatenadas[0:6], color='blue')
plt.scatter(data['VTI_F'][6:12],predicciones_concatenadas[6:12], color='orange')
plt.scatter(data['VTI_F'][12:18],predicciones_concatenadas[12:18], color='green')
plt.scatter(data['VTI_F'][18:24],predicciones_concatenadas[18:24], color='red')
plt.legend({"BPM=12  PEEP=0", "BPM=20  PEEP=0", "BPM=12  PEEP=10", "BPM=20  PEEP=10"})
# grafico de las rectas de ajuste
plt.plot(data['VTI_F'][:6], predicciones_concatenadas[0:6], color='grey', linewidth=2) # grafica la recta de regresion ajustada
plt.plot(data['VTI_F'][6:12], predicciones_concatenadas[6:12], color='grey', linewidth=2)
plt.plot(data['VTI_F'][12:18], predicciones_concatenadas[12:18], color='grey', linewidth=2)
plt.plot(data['VTI_F'][18:24], predicciones_concatenadas[18:24], color='grey', linewidth=2)
plt.show() # mostrar grafico