# Modelo LSTM para ['Predecir el precio del Oro']

Esta carpeta contiene los archivos necesarios para entender, utilizar y potencialmente replicar el entrenamiento de un modelo de Red Neuronal Recurrente Long Short-Term Memory (LSTM) diseñado para ['predecir el precio del oro en el mercado actual'].

## Contenido

* **`[GoldLSTM].ipynb`**: Este notebook de Jupyter contiene una explicación detallada del proceso de entrenamiento del modelo LSTM. Incluye:
    * **Preprocesamiento de datos:** Pasos realizados para preparar los datos para el entrenamiento.
    * **Arquitectura del modelo:** Definición de las capas LSTM y otras capas utilizadas.
    * **Entrenamiento:** Código para entrenar el modelo utilizando los datos preparados.
    * **Evaluación:** Métricas utilizadas para evaluar el rendimiento del modelo en datos de prueba.
    * **Visualizaciones:** Gráficos para ilustrar el proceso de entrenamiento y los resultados.

* **`[modelo_oro90].h5`**: Este archivo contiene los pesos entrenados del modelo LSTM guardado en formato HDF5. Este formato es comúnmente utilizado por bibliotecas como Keras para almacenar modelos de redes neuronales.

## Cómo utilizar

1.  **Explorar el Notebook:** Abre el archivo `[GoldLSTM].ipynb` para entender en detalle la implementación del modelo, el proceso de entrenamiento y los resultados obtenidos. Puedes ejecutar las celdas del notebook en un entorno de Jupyter para revisar el código y las salidas.

2.  **Cargar el Modelo Entrenado:** El archivo `[modelo_oro90].h5` puede ser cargado en un entorno de Python utilizando bibliotecas como TensorFlow o Keras para realizar inferencias o para seguir entrenando el modelo con nuevos datos.

    ```python
    from tensorflow.keras.models import load_model

    # Asegúrate de que la ruta al archivo .h5 sea correcta
    modelo_cargado = load_model('[modelo_oro90].h5')

    # Ahora puedes usar 'modelo_cargado' para hacer predicciones, evaluar, etc.
    # ejemplo: predicciones = modelo_cargado.predict(nuevos_datos)
    ```

## Requisitos

Para ejecutar el notebook y utilizar el modelo, necesitarás tener instaladas las siguientes bibliotecas de Python:

* **Python** (versión recomendada: 3.x)
* **TensorFlow** (con Keras)
* **NumPy**
* **Pandas**
* **yfinance** 
* **Matplotlib** (opcional, para visualizaciones en el notebook)
* **Scikit-learn** (opcional, para algunas tareas de preprocesamiento o evaluación)
  
Puedes instalar estas bibliotecas utilizando `pip`:

```bash
pip install numpy pandas tensorflow matplotlib scikit-learn yfinance
