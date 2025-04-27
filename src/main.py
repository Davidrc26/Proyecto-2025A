from src.controllers.manager import Manager
from src.controllers.strategies.ACO import ACO

def iniciar():
    """Punto de entrada principal"""
    # Configuración inicial
    estado_inicial = "1000"
    condiciones = "1110"
    alcance = "1110"
    mecanismo = "1110"

    # Crear el gestor del sistema
    gestor_sistema = Manager(estado_inicial)

    # Parámetros para ACO
    num_hormigas = 10
    alpha = 1.0
    beta = 2.0
    rho = 0.5
    iteraciones = 50

    # Instanciar ACO
    analizador_aco = ACO(gestor_sistema, num_hormigas, alpha, beta, rho, iteraciones)

    # Ejecutar la estrategia ACO
    resultado_aco = analizador_aco.aplicar_estrategia(
        condiciones,
        alcance,
        mecanismo,
    )

    # Mostrar resultados
    print("Resultado de ACO:")
    print(f"Estrategia: {resultado_aco.estrategia}")
    print(f"Pérdida: {resultado_aco.perdida}")
    print(f"Partición: {resultado_aco.particion}")
    print(f"Tiempo total: {resultado_aco.tiempo_total:.4f} segundos")