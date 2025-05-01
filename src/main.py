from src.controllers.manager import Manager
from src.controllers.strategies.GeneticGA import GeneticGA
from src.controllers.strategies.force import BruteForce
from src.controllers.strategies.one_partition import OneFuturePartition
from src.controllers.strategies.q_nodes import QNodes

def iniciar():
    """Punto de entrada para probar la metaheurística genética"""
    # 1) Configuración inicial
    estado_inicial = "10000000000000000000"
    condiciones =    "11111111111111111111"
    alcance =        "11111111111111111111"
    mecanismo =      "11111111111111111111"

    # 2) Crear el gestor del sistema
    gestor_sistema = Manager(estado_inicial)

    analisis = QNodes(gestor_sistema)

    sia = analisis.aplicar_estrategia(condiciones, alcance, mecanismo)

    print("Resultado de QNodes:")
    print(f"  Estrategia: {sia.estrategia}")
    print(f"  Pérdida (φ): {sia.perdida:.6f}")
    print(f"  Partición: {sia.particion}")
#     ga = GeneticGA(
#     gestor=gestor_sistema,
#     pop_size=150,
#     generations=300,
#     crossover_rate=0.8,
#     mutation_rate=0.1,
#     elitism=2,
#     patience=25,
#     verbose=True
# )

#     resultado_ga = ga.aplicar_estrategia(condiciones, alcance, mecanismo)

#     # 5) Mostrar resultados
#     print("Resultado de GA:")
#     print(f"  Estrategia: {resultado_ga.estrategia}")
#     print(f"  Pérdida (φ): {resultado_ga.perdida:.6f}")
#     print(f"  Partición: {resultado_ga.particion}")
#     print(f"  Tiempo total: {resultado_ga.tiempo_ejecucion:.4f} segundos")