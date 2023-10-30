import pandas as pd
import matplotlib.pyplot as plt


def eliminar_nan_100p(df):
    """
    Elimina las columnas que contienen el 100% de datos nulos en un DataFrame y devuelve una copia modificada.

    Parámetros:
    - df (pandas.DataFrame): El DataFrame original.

    Retorna:
    - pandas.DataFrame: Una copia del DataFrame original sin las columnas que tienen el 100% de datos nulos.
    """
    # Se crea una copia del df para evitar modificaciones en el original
    df_copy = df.copy()
    # Se genera una lista que contenga el 100% de datos nulos
    lista_columnas_100_datos_nulos = [i for i in df_copy.columns if (df_copy[i].isnull().sum()/df_copy.shape[0])== 1]
    # Se dropean las columnas con el 100% de los datos nulos
    df_copy = df_copy.drop(columns=lista_columnas_100_datos_nulos)
    return df_copy


# Función para obtener las columnas según su tipo
def separar_columnas_por_tipo(dataframe):
    """
    Separa las columnas de un DataFrame en listas según su tipo (int, float, object).

    Parámetros:
    - dataframe (pandas.DataFrame): El DataFrame del cual se desean separar las columnas.

    Retorna:
    - columnas_int (list): Lista de nombres de columnas de tipo int.
    - columnas_float (list): Lista de nombres de columnas de tipo float.
    - columnas_object (list): Lista de nombres de columnas de tipo object.
    """
  # Crear listas vacías para almacenar los nombres
    columnas_int = []
    columnas_float = []
    columnas_object = []
  # Recorrer las columnas
    for col in dataframe.columns:
      # Agregar a la lista según corresponda
        if dataframe[col].dtype == 'int64':
            columnas_int.append(col)
        elif dataframe[col].dtype == 'float64':
            columnas_float.append(col)
        else:
            columnas_object.append(col)

    return columnas_int, columnas_float, columnas_object


# Se crea función para pasar el tipo de dato a formato tiempo
def convert_to_timedelta(time_value):
    """
    Convierte un valor de tiempo en formato de cadena a formato de tiempo (Timedelta) de pandas.

    Parámetros:
    - time_value (str): Valor de tiempo en formato de cadena (por ejemplo, 'hh:mm:ss').

    Retorna:
    - pd.Timedelta: Un objeto Timedelta que representa el tiempo en horas, minutos y segundos.
      Si el valor no es una cadena, devuelve pd.NaT (Not-a-Time) para indicar un valor nulo.
    """
    if isinstance(time_value, str):
        hours, minutes, seconds = map(int, time_value.split(':'))
        return pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds)
    else:
        return pd.NaT  # Devuelve NaN si el valor no es una cadena
    
    
# Función para asignar categorías de duración
def categorize_duration(duration):
    """
    Asigna categorías de duración a un valor de duración en segundos.

    Parámetros:
    - duration (pd.Timedelta): Valor de duración representado como un objeto Timedelta de pandas.

    Retorna:
    - int: El número de categoría asignado según la duración proporcionada.
      - 1: Duración igual a cero.
      - 2: Duración mayor a cero y menor a 20 segundos.
      - 3: Duración mayor o igual a 20 segundos y menor a 2 minutos (120 segundos).
      - 4: Duración mayor o igual a 2 minutos y menor a 5 minutos (300 segundos).
      - 5: Duración igual o mayor a 5 minutos.
      - 6: Llamadas sin registro (si el valor de duración es NaN).
    """
    if pd.isnull(duration):
        return 6  # Llamadas sin registro
    elif 0 == duration.total_seconds():
        return 1
    elif 0 < duration.total_seconds() < 20:
        return 2
    elif 20 <= duration.total_seconds() < 120:
        return 3
    elif 120 <= duration.total_seconds() < 300:
        return 4
    else:
        return 5


def sum_y_merge(df, columns_to_sum, columna_original, id_column='Associated Deal IDs'):
    """
    Agrupa un DataFrame por una columna, suma y renombra las columnas sumadas,
    y devuelve el DataFrame modificado.

    Parámetros:
    - df (pandas.DataFrame): El DataFrame original.
    - columns_to_sum (list): Lista de nombres de columnas a sumar.
    - columna_original (str): Nombre de la columna original que se eliminará.
    - id_column (str, opcional): Nombre de la columna de identificación para la agrupación (por defecto es 'Associated Deal IDs').

    Retorna:
    - pandas.DataFrame: El DataFrame original actualizado con las columnas sumadas y la columna original eliminada.
    """
    # Se agrupan los datos del DataFrame por la columna de identificación (id_column) y se suman las columnas seleccionadas.
    grouped = df.groupby(id_column)[columns_to_sum].sum()

    # Se crean nuevos nombres de columna para las columnas sumadas, agregando 'Suma ' al inicio.
    new_column_names = ['Suma ' + col for col in columns_to_sum]

    # Se renombran las columnas del DataFrame agrupado con los nuevos nombres de columna.
    grouped.columns = new_column_names

    # Se realiza una operación de fusión (merge) entre el DataFrame original y el DataFrame agrupado.
    # Se utiliza la columna de identificación como la columna izquierda y los índices del DataFrame agrupado como la columna derecha.
    df = df.merge(grouped, left_on=id_column, right_index=True)

    # Se eliminan del DataFrame las columnas originales que fueron sumadas.
    df = df.drop(columns=(columns_to_sum + [columna_original]))

    # Se retorna el DataFrame actualizado.
    return df



    
# Función que binariza una lista de columnas
def binarizador(columnas_list, df, borrar=False, minimo=False):
    '''
    Crea nuevas columnas binarizadas a partir de una lista de columnas y un dataframe.

    Parámetros:
    -----------
    columnas_list (list): Lista de nombres de columnas a binarizar.
    df (pandas.DataFrame): Dataframe original.
    borrar (bool): Opcional. Si es True, las columnas originales se eliminan. Por defecto es False.
    minimo (bool): Opcional. Si es True, se generan las columnas binarias en la menor cantidad de columnas posibles. Por defecto es False.

    Returns:
    --------
    pandas.DataFrame: Dataframe con las nuevas columnas binarizadas agregadas.

    Ejemplo:
    --------
    df = binarizador(['sexo', 'ciudad'], df, borrar=True, minimo=True)

    '''

    for col in columnas_list:
        df = df.copy()  # Creamos una copia del dataframe para no modificar el original.
        if minimo:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)  # se binarizan las columnas en el menor número posible y se guardan en un nuevo DataFrame
            df = pd.concat([df, dummies], axis=1) # se concatenan ambos DataFrames
        else:    
            dummies = pd.get_dummies(df[col], prefix=col)  # se binarizan las columnas y se guardan en un nuevo DataFrame
            df = pd.concat([df, dummies], axis=1) # se concatenan ambos DataFrames
        if borrar: # opción para borrar las columnas
            df.drop(col, axis=1, inplace=True)
            
    # Validación para convertir True y False en 0 y 1
    df = df.astype(int)
    
    return df


def despliega_histogramas(data_frame, column_list):
    """
    Genera histogramas para una lista de columnas de un DataFrame y muestra los gráficos en una sola figura.

    Parámetros:
    - data_frame (pandas.DataFrame): El DataFrame que contiene los datos a representar.
    - column_list (list): Lista de nombres de las columnas para las cuales se generarán histogramas.

    Retorna:
    - None: Muestra los histogramas en una figura.
    """
    # Calcula el número total de columnas y el número de columnas por fila
    num_columns = len(column_list)
    num_cols = 2  # Número de columnas por fila en el arreglo de gráficos

    # Calcula el número de filas necesarias redondeando hacia arriba
    num_rows = -(-num_columns // num_cols)

    # Crea una figura y un arreglo de gráficos con el tamaño adecuado
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6 * num_rows))

    # Si solo hay una fila de gráficos, convierte 'axes' en un arreglo de dos dimensiones
    if num_rows == 1:
        axes = np.array(axes).reshape(1, -1)

    # Itera sobre las columnas y sus índices
    for i, column in enumerate(column_list):
        row = i // num_cols
        col = i % num_cols

        # Obtiene el gráfico correspondiente en la posición actual
        ax = axes[row, col]

        # Itera sobre los valores de 'Cliente_Perdido'
        for perdido in [0, 1]:
            # Filtra los datos según 'Cliente_Perdido' y la columna actual
            data_perdido = data_frame[data_frame['Cliente_Perdido'] == perdido][column]

            # Crea el histograma con los datos filtrados
            ax.hist(data_perdido, bins=20, alpha=0.7, label=f'Cliente_Perdido={perdido}')

        # Configura el título, etiquetas de ejes y leyenda del gráfico
        ax.set_title(f'Histograma de {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frecuencia')
        ax.legend()
        ax.grid(True)

    # Elimina los ejes en blanco si no se utilizan todas las ubicaciones de subgráficos
    if len(column_list) < num_rows * num_cols:
        for i in range(len(column_list), num_rows * num_cols):
            fig.delaxes(axes.flat[i])

    # Ajusta el espaciado entre los gráficos y muestra la figura
    plt.tight_layout()
    plt.show()
    
    
def despliega_histogramas2(data_frame, column_list):
    """
    Genera histogramas para una lista de columnas de un DataFrame y muestra los gráficos en una sola figura.

    Parámetros:
    - data_frame (pandas.DataFrame): El DataFrame que contiene los datos a representar.
    - column_list (list): Lista de nombres de las columnas para las cuales se generarán histogramas.

    Retorna:
    - None: Muestra los histogramas en una figura.
    """
    num_columns = len(column_list)
    # Número de columnas por fila en el arreglo de gráficos
    num_cols = 2
    # Cálculo de filas redondeado hacia arriba
    num_rows = -(-num_columns // num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6 * num_rows))

    # Calculo de la posición (fila y columna) en una cuadrícula de subgráficos
    for i, column in enumerate(column_list):
        row = i // num_cols
        col = i % num_cols

        # Si solo hay una fila de gráficos, calcula el índice adecuado
        if num_rows == 1:
            ax = axes[i]
        else:
            ax = axes[row, col]

        # Itera sobre los valores de 'Cliente_Perdido'
        for perdido in [0, 1]:
            data_perdido = data_frame[data_frame['Cliente_Perdido'] == perdido][column]
            ax.hist(data_perdido, bins=20, alpha=0.7, label=f'Cliente_Perdido={perdido}')

        # Configura el título, etiquetas de ejes y leyenda del gráfico
        ax.set_title(f'Histograma de {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frecuencia')
        ax.legend()
        ax.grid(True)

    # Elimina los ejes en blanco si no se utilizan todas las ubicaciones de subgráficos
    if len(column_list) < num_rows * num_cols:
        for i in range(len(column_list), num_rows * num_cols):
            fig.delaxes(axes.flat[i])

    plt.tight_layout()
    plt.show()
    
    
# Define una función para resaltar las celdas con correlaciones altas sore el 60%
def resalta_correlacion_alta(s):
    """
    Resalta las celdas con correlaciones altas mayores o iguales al 60% en un DataFrame.

    Parámetros:
    - s (pandas.Series): Una serie que representa las correlaciones entre columnas.

    Retorna:
    - list: Una lista de cadenas de estilo CSS para resaltar las celdas con correlaciones altas.
    """
    is_high = (s.abs() >= 0.6) & (s.abs() < 1)
    return ['background-color: yellow' if v else '' for v in is_high]