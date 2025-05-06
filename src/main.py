from src.controllers.manager import Manager

from src.controllers.strategies.geometric import GeometricSIA

def iniciar():
    """Punto de entrada principal"""
                    # ABCD #
    estado_inicial = "000"
    condiciones =    "111"
    alcance =        "111"
    mecanismo =      "111"

    gestor_sistema = Manager(estado_inicial)

    ### Ejemplo de solución mediante módulo de fuerza bruta ###
    analizador_fb = GeometricSIA(gestor_sistema)
    sia_uno = analizador_fb.aplicar_estrategia(
        condiciones,
        alcance,
        mecanismo,
    )
    print(sia_uno)



# from src.controllers.manager import Manager
# from src.controllers.strategies.GeneticGA import GeneticGA
# from src.controllers.strategies.force import BruteForce
# from src.controllers.strategies.one_partition import OneFuturePartition
# from src.controllers.strategies.q_nodes import QNodes
# import pandas as pd
# import re

# def iniciar():
#     """Punto de entrada para probar la metaheurística genética"""
#     # 1) Configuración inicial
#     df = read_tests()
#     data = df["15B_nodos"]
#     estado_inicial = "100000000000000"
#     condiciones =    "111111111111111"
#     gestor_sistema = Manager(estado_inicial)
#     analisis = GeneticGA(gestor_sistema, len(estado_inicial))
#     soluciones = []
#     for i in range(len(data)):
#         alcance, mecanismo = extraer_cadenas(data[i], 15)
#         if alcance is None or mecanismo is None:
#             print(f"Error en la cadena: {data[i]}")
#             continue
#         sia = analisis.aplicar_estrategia(condiciones, alcance, mecanismo)
#         soluciones.append(sia)
#         print(sia)

#     save_results(soluciones, "resultados.xlsx")

#     # 2) Crear el gestor del sistema
    

#     #analisis = BruteForce(gestor_sistema)

    

# #     ga = GeneticGA(
# #     gestor=gestor_sistema,
# #     pop_size=150,
# #     generations=300,
# #     crossover_rate=0.8,
# #     mutation_rate=0.1,
# #     elitism=2,
# #     patience=25,
# #     verbose=True
# # )

# #     resultado_ga = ga.aplicar_estrategia(condiciones, alcance, mecanismo)

# #     # 5) Mostrar resultados
# #     print("Resultado de GA:")
# #     print(f"  Estrategia: {resultado_ga.estrategia}")
# #     print(f"  Pérdida (φ): {resultado_ga.perdida:.6f}")
# #     print(f"  Partición: {resultado_ga.particion}")
# #     print(f"  Tiempo total: {resultado_ga.tiempo_ejecucion:.4f} segundos")



# def save_results(solutions, filename):
#     """Guarda el resultado en un archivo CSV"""
#     data = [obj.__dict__ for obj in solutions]
#     df = pd.DataFrame(data=data)
#     df.to_excel(filename, index=False)

# def read_tests():
#     """Función para leer los tests desde un archivo CSV"""
#     df = pd.read_excel("./pruebasAEjecutar.xlsx", sheet_name="Hoja1")
#     return df


# def extraer_cadenas(texto, tamaño_red):
#     abecedario = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#     abecedario = abecedario[:tamaño_red]
#     patron = r'([A-Z]+)_\{t\+1\}\|([A-Z]+)_\{t\}'
#     coincidencia = re.match(patron, texto)
#     if coincidencia:
#         cadena1 = ''.join('1' if letra in coincidencia.group(1) else '0' for letra in abecedario)
#         cadena2 = ''.join('1' if letra in coincidencia.group(2) else '0' for letra in abecedario)
#         return cadena1, cadena2
#     else:
#         return None, None
    
# iniciar()