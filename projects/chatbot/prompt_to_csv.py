import csv
from pathlib import Path
import sys

def obtener_nombres_relativos(root_path: Path) -> list[Path]:
    """
    Función recursiva para explorar directorios y encontrar archivos .md 
    hasta un máximo de 2 niveles de subdirectorios intermedios.
    Devuelve una lista de objetos con la información extraída.
    """

    # Contador global o dentro del scope de llamada? 
    # Para mantener modularidad, pasamos el estado de profundidad.

    def explorar(current_dir: Path, profundidad_actual: int):
        # Validación básica de ruta actual (aunque path.exists() ya debería ser cierto)
        if not current_dir.is_dir():
            return []

        lista_resultados = []

        # Iterar sobre los elementos del directorio actual
        for item in current_dir.iterdir():
            try:
                # Calcular profundidad relativa al ROOT (cuántas carpetas hemos cruzado para llegar aquí)
                # La longitud de las partes del path relativo indica la profundidad.
                # Ejemplo: ./root -> [] (0), ./root/a -> ['a'] (1), ./root/a/b -> ['a','b'] (2)

                ruta_relativa = item.relative_to(root_path)
                partes_ruta = list(ruta_relativa.parts)

                # Si es un archivo .md:
                if item.is_file() and item.suffix == '.md':
                    # 1. ID Incremental (usaremos un contador externo o acumulador aquí si fuera necesario, 
                    # pero para simplificar la estructura pasamos una referencia mutable o generamos IDs al final)
                    # Aquí simplemente lo guardamos y ordenamos después.

                    # Generar Nombre:
                    # Separa las partes por '-', ignorando el archivo raíz (empty string inicial si aplica)
                    # Ejemplo path: a/b/file.md -> partes ['a', 'b', 'file.md']
                    # Queremos: a-b-file

                    nombre_archivo = item.stem  # Elimina la extensión .md
                    carpetas_intermedias = partes_ruta[:-1] # Todos menos el último (el archivo)

                    # Construir la cadena con guiones
                    if carpetas_intermedias:
                        nombre_concatenado = "-".join(carpetas_intermedias) + "-" + str(nombre_archivo)
                    else:
                        # Archivo directo en root (sin subcarpetas intermedias)
                        nombre_concatenado = str(nombre_archivo)

                    lista_resultados.append({
                        "name": nombre_concatenado,
                        "prompt": item.read_text(encoding="utf-8")
                    })

                elif item.is_dir():
                    # Es un subdirectorio: Verificar si podemos bajar más
                    # Si estamos en profundidad 2 (carpetas intermedias), no debemos entrar al siguiente nivel 
                    # porque eso haría 3 carpetas intermedias.
                    if profundidad_actual < 2:
                        lista_resultados.extend(explorar(item, profundidad_actual + 1))

            except PermissionError:
                # Saltar directorios sin permisos de lectura para evitar bloqueos
                continue

        return lista_resultados

    return explorar(root_path, 0)


def escribir_csv(datos: list[dict], archivo_salida: str = "output.csv"):
    """
    Escribe la lista de diccionarios en un archivo CSV con codificación UTF-8.
    Maneja caracteres especiales mediante csv.writer que maneja escapado interno.
    """
    with open(archivo_salida, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Id', 'name', 'prompt'])

        # Escribir encabezados obligatorios
        writer.writeheader()

        # Escribir filas con ID incrementales
        for index, fila in enumerate(datos):
            row_data = {
                'Id': index + 501,
                'name': fila['name'],
                'prompt': fila['prompt'].replace("\n", "\\n")
            }
            writer.writerow(row_data)


def main():
    # Punto de entrada del script
    if len(sys.argv) < 2:
        print("Uso: python script.py <ruta_del_directorio_raiz>")
        sys.exit(1)

    ruta_base = Path(sys.argv[1])

    # Verificación de existencia y tipo
    if not ruta_base.exists():
        raise FileNotFoundError(f"El directorio especificado no existe: {ruta_base}")

    if not ruta_base.is_dir():
        raise NotADirectoryError(f"La ruta especificada es un archivo, no una carpeta: {ruta_base}")

    # Ejecutar exploración
    try:
        datos_extraidos = obtener_nombres_relativos(ruta_base)

        # Escribir al CSV
        escribir_csv(datos_extraidos)
        print(f"Proceso completado. Se extrajeron {len(datos_extraidos)} archivos .md.")
        print(f"Archivo de salida generado: output.csv")

    except PermissionError as e:
        print(f"Error de permisos durante la lectura de directorios: {e}")
    except UnicodeDecodeError as e:
        print(f"Error de codificación UTF-8 en uno de los archivos leídos: {e}")
    except Exception as e:
        # Manejo genérico para errores inesperados
        raise RuntimeError(f"Ocurrió un error durante la ejecución del script: {e}")

if __name__ == "__main__":
    main()