from src.controllers.manager import Manager

from src.controllers.strategies.geometric import GeometricSIA
from src.controllers.strategies.q_nodes import QNodes
def iniciar():
    """Punto de entrada principal"""
                    # ABCD #
    estado_inicial = "1000000000"

    condiciones =    "1111111111"
                     #ABCDEFGHIJKLMNO# t+1
    alcance =        "0111111111"
                     #ABCDEFGHIJKLMNO# t
    mecanismo =      "1111111111"

    gestor_sistema = Manager(estado_inicial)

    ### Ejemplo de solución mediante módulo de fuerza bruta ###
    analizador_fb = GeometricSIA(gestor_sistema)

    sia_uno = analizador_fb.aplicar_estrategia(
        condiciones,
        alcance,
        mecanismo,
    )
    print(sia_uno)

