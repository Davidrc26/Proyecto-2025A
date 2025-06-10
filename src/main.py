from src.controllers.manager import Manager
import pandas as pd
import re
from src.strategies.q_nodes import QNodes
from src.strategies.geometric import GeometricSIA


def iniciar():
    df = read_tests()
    data = df["10_nodos_a"]

    estado_inicial = "1000000000"

    condiciones =    "1111111111" 

    alcance, mecanismo = extraer_cadenas(data[35], len(estado_inicial))

    gestor_sistema = Manager(estado_inicial)

    ### Ejemplo de solución mediante módulo de fuerza bruta ###
    analizador_fb = GeometricSIA(gestor_sistema)

    sia_uno = analizador_fb.aplicar_estrategia(
        condiciones,
        alcance,
        mecanismo,
    )
    print(sia_uno)



def save_results(solutions, filename):
    """Guarda el resultado en un archivo CSV"""
    data = [obj.__dict__ for obj in solutions]
    df = pd.DataFrame(data=data)
    df.to_excel(filename, index=False)

def read_tests():
    """Función para leer los tests desde un archivo CSV"""
    df = pd.read_excel("./pruebasAEjecutar.xlsx", sheet_name="Hoja1")
    return df

def extraer_cadenas(texto, tamaño_red):
    abecedario = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    abecedario = abecedario[:tamaño_red]
    patron = r'([A-Z]+)_\{t\+1\}\|([A-Z]+)_\{t\}'
    coincidencia = re.match(patron, texto)
    if coincidencia:
        cadena1 = ''.join('1' if letra in coincidencia.group(1) else '0' for letra in abecedario)
        cadena2 = ''.join('1' if letra in coincidencia.group(2) else '0' for letra in abecedario)
        return cadena1, cadena2
    else:
        return None, None




def save_results(solutions, filename):
    """Guarda el resultado en un archivo CSV"""
    data = [obj.__dict__ for obj in solutions]
    df = pd.DataFrame(data=data)
    df.to_excel(filename, index=False)

def read_tests():
    """Función para leer los tests desde un archivo CSV"""
    df = pd.read_excel("./pruebasAEjecutar.xlsx", sheet_name="Hoja1")
    return df

def extraer_cadenas(texto, tamaño_red):
    abecedario = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    abecedario = abecedario[:tamaño_red]
    patron = r'([A-Z]+)_\{t\+1\}\|([A-Z]+)_\{t\}'
    coincidencia = re.match(patron, texto)
    if coincidencia:
        cadena1 = ''.join('1' if letra in coincidencia.group(1) else '0' for letra in abecedario)
        cadena2 = ''.join('1' if letra in coincidencia.group(2) else '0' for letra in abecedario)
        return cadena1, cadena2
    else:
        return None, None


