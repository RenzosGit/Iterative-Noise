import os
import shutil

def borrar_datos_en_carpeta(carpeta):
    # Existe?
    if os.path.exists(carpeta):
        # Itera sobre los archivos y sus subdirectorios
        for archivo in os.listdir(carpeta):
            archivo_path = os.path.join(carpeta, archivo)
            # Si es archivo, elimina
            if os.path.isfile(archivo_path):
                os.remove(archivo_path)
            # Si es carpeta, eliminar recursivamente
            elif os.path.isdir(archivo_path):
                shutil.rmtree(archivo_path)
        print(f"Todos los archivos y subdirectorios en {carpeta} han sido eliminados.")
    else:
        print(f"La carpeta {carpeta} no existe.")
