from src.controllers.manager import Manager

from src.controllers.strategies.geometric import GeometricSIA
from src.controllers.strategies.q_nodes import QNodes
def iniciar():
    """Punto de entrada principal"""
                    # ABCD #
    estado_inicial = "10000000000000000000"

    condiciones =    "11111111111111111111"
                     #ABCDEFGHIJKLMNO# t+1
    alcance =        "11111111111111111111"
                     #ABCDEFGHIJKLMNO# t
    mecanismo =      "11111111111111111111"

    gestor_sistema = Manager(estado_inicial)

    ### Ejemplo de solución mediante módulo de fuerza bruta ###
    analizador_fb = QNodes(gestor_sistema)

    sia_uno = analizador_fb.aplicar_estrategia(
        condiciones,
        alcance,
        mecanismo,
    )
    print(sia_uno)


# 