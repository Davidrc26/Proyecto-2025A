from src.controllers.manager import Manager
from src.controllers.strategies.GeneticGA import GeneticGA
def iniciar():
    """Punto de entrada para probar la metaheurística genética"""
    # 1) Configuración inicial
    estado_inicial = "1000000000"
    condiciones    = "1111111111"
    alcance        = "1111111111"
    mecanismo      = "1111111111"

    # 2) Crear el gestor del sistema
    gestor_sistema = Manager(estado_inicial)

    # 4) Instanciar y ejecutar el GA
    ga = GeneticGA(
    gestor=gestor_sistema,
    pop_size=30,
    generations=70,
    crossover_rate=0.7,
    mutation_rate=0.05,
    elitism=1,
    verbose=True
)

    resultado_ga = ga.aplicar_estrategia(condiciones, alcance, mecanismo)

    # 5) Mostrar resultados
    print("Resultado de GA:")
    print(f"  Estrategia: {resultado_ga.estrategia}")
    print(f"  Pérdida (φ): {resultado_ga.perdida:.6f}")
    print(f"  Partición: {resultado_ga.particion}")
    print(f"  Tiempo total: {resultado_ga.tiempo_ejecucion:.4f} segundos")

