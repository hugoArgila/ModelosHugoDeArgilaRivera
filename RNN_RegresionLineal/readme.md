# Modelo RNN para Predicción del Precio de Vivienda en Islas Baleares con API Flask

Este repositorio contiene los archivos necesarios para entender, utilizar y probar un modelo de Red Neuronal Recurrente (RNN) diseñado para predecir el precio de la vivienda en las Islas Baleares. Se incluye una API construida con Flask para interactuar con el modelo a través de una interfaz web sencilla y recibir predicciones basadas en datos de entrada en formato JSON.

## Contenido

* **`Modelos/`**:
    * **`modelo_baleares61R2.h5`**: Modelo RNN entrenado para la predicción del precio de vivienda, con un coeficiente de determinación (R²) de 0.61 en el conjunto de prueba.
    * **`modelo_baleares71R2.h5`**: Modelo RNN entrenado para la predicción del precio de vivienda, con un coeficiente de determinación (R²) de 0.71 en el conjunto de prueba.
    * **`scaler_X.pkl`**: Objeto `StandardScaler` de scikit-learn utilizado para escalar las características de entrada durante el entrenamiento de los modelos. Necesario para preprocesar nuevos datos antes de la predicción.
    * **`scaler_y.pkl`**: Objeto `StandardScaler` de scikit-learn utilizado para escalar la variable objetivo (precio de la vivienda) durante el entrenamiento de los modelos. Necesario para desescalar la predicción del modelo a su valor original.

* **`static/`**:
    * **`styles.css`**: Archivo de hojas de estilo en cascada para personalizar la apariencia de la interfaz web.

* **`templates/`**:
    * **`index.html`**: Archivo HTML que define la estructura y los elementos de la interfaz web para ingresar los datos de la vivienda y obtener la predicción del precio a través de la API.

* **`app.py`**: Este archivo de Python contiene el código de la aplicación Flask que define la API para interactuar con los modelos RNN. Incluye rutas para recibir datos de entrada en formato JSON a través de la interfaz web, cargar los modelos y los scalers, realizar la predicción y mostrar el resultado.

* **`input_data.json`**: Archivo JSON de ejemplo que muestra la estructura esperada de los datos de entrada que la API leerá para realizar una predicción. Puedes utilizar este archivo como guía para entender el formato requerido.

* **`json_structure.txt`**: Archivo de texto que explica detalladamente cada campo esperado en el archivo `input_data.json`, incluyendo su tipo de dato y una breve descripción. Esto ayuda a los usuarios a entender la estructura del JSON que la API espera recibir.

* **`requirements.txt`**: Este archivo lista las dependencias de Python necesarias para ejecutar la aplicación Flask y utilizar los modelos RNN. Se utiliza para instalar las bibliotecas requeridas con `pip`.

* **`data.csv`**: ([**Descargar Dataset**]([ENLACE_AL_ARCHIVO_CSV])) Este archivo CSV contiene el dataset utilizado para entrenar los modelos de predicción de precios de vivienda en las Islas Baleares. Incluye las características utilizadas como entrada y la variable objetivo (precio de la vivienda).

## Cómo utilizar

1.  **Explorar los Modelos:** La carpeta `Modelos/` contiene los modelos RNN pre-entrenados (`.h5`) junto con los objetos `StandardScaler` (`.pkl`) utilizados para el escalado de las características de entrada y la variable objetivo. Los nombres de los modelos indican su coeficiente de determinación (R²) en el conjunto de prueba.

2.  **Descargar el Dataset:** Puedes descargar el dataset completo en formato CSV haciendo clic en el enlace [**Descargar Dataset**]([ENLACE_AL_ARCHIVO_CSV]) en la sección de Contenido. Este archivo puede ser útil para explorar los datos utilizados en el entrenamiento de los modelos.

3.  **Entender la Estructura del JSON de Entrada:** Lee el archivo `json_structure.txt` para comprender cada campo requerido en el archivo `input_data.json` y la estructura general del JSON que la API espera recibir para realizar una predicción.

4.  **Ejecutar la API de Flask:**
    * Asegúrate de tener instaladas todas las dependencias listadas en `requirements.txt`. Puedes instalarlas ejecutando el siguiente comando en tu terminal:
        ```bash
        pip install -r requirements.txt
        ```
    * Navega hasta el directorio raíz del repositorio (donde se encuentra el archivo `app.py`) en tu terminal.
    * Ejecuta la aplicación Flask utilizando el siguiente comando:
        ```bash
        python app.py
        ```
    * Una vez que la aplicación se esté ejecutando, podrás acceder a la interfaz web abriendo tu navegador y navegando a la dirección que se mostrará en la terminal (normalmente `http://127.0.0.1:5000/`).

5.  **Interactuar con la Interfaz Web:**
    * Abre la URL de la aplicación Flask en tu navegador.
    * Deberías ver la interfaz definida en `templates/index.html`.
    * Sigue las instrucciones en la página web para ingresar las características de la vivienda para la cual deseas predecir el precio. Los datos que ingreses se enviarán a la API en formato JSON (siguiendo la estructura definida en `input_data.json` y explicada en `json_structure.txt`).
    * La API utilizará el modelo seleccionado (implícito en la `app.py` o posiblemente seleccionable en la interfaz) para realizar la predicción y mostrar el resultado en la página web.

## Requisitos

Para ejecutar la aplicación Flask y utilizar los modelos RNN, necesitarás tener instaladas las siguientes bibliotecas de Python (listadas en `requirements.txt`):

* **Python** (versión recomendada: 3.x)
* **Flask**
* **TensorFlow** (con Keras)
* **scikit-learn**
* **joblib** (para cargar los archivos `.pkl` de los scalers)
* Otras bibliotecas que puedan estar listadas en `requirements.txt`.

## Notas

* Los modelos RNN (`.h5`) fueron entrenados utilizando datos de precios de vivienda en las Islas Baleares, disponibles en el archivo `data.csv`.
* Los archivos `scaler_X.pkl` y `scaler_y.pkl` son cruciales para asegurar que los nuevos datos de entrada se preprocesen de la misma manera que los datos de entrenamiento y que la predicción se desescale correctamente a la escala original del precio.
* La API de Flask (`app.py`) carga los modelos pre-entrenados y los scalers al inicio. Luego, recibe los datos de entrada a través de la interfaz web, los preprocesa utilizando el `scaler_X`, realiza la predicción con el modelo seleccionado y desescala la salida utilizando `scaler_y` antes de mostrar el resultado.
* El archivo `input_data.json` es un ejemplo de cómo deben estructurarse los datos de entrada para la API.
* El archivo `json_structure.txt` proporciona una guía detallada sobre el significado y el tipo de cada campo esperado en el JSON de entrada.

## Posibles mejoras

* Permitir al usuario seleccionar el modelo (`modelo_baleares61R2.h5` o `modelo_baleares71R2.h5`) a través de la interfaz web.
* Agregar validación de los datos de entrada en la API para asegurar que cumplen con la estructura esperada.
* Implementar manejo de errores más robusto en la API.
* Mejorar la interfaz web con más información sobre los modelos y los datos de entrada requeridos.

**Recuerda reemplazar `[ENLACE_AL_ARCHIVO_CSV]` con la URL real donde se encuentra tu archivo `data.csv` si lo estás alojando externamente, o si el archivo está directamente en tu repositorio, puedes usar una ruta relativa como `data.csv` para que el enlace funcione correctamente en la vista del repositorio en plataformas como GitHub.**

Si el archivo `data.csv` está en la raíz de tu repositorio, el enlace debería ser simplemente:

```markdown
* **`data.csv`**: ([**Descargar Dataset**](data.csv))
