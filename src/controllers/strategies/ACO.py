import time
import numpy as np

from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import ACO_LABEL
from src.constants.base import TYPE_TAG
from src.middlewares.profile import profile


class ACO(SIA):
    """Ant Colony Optimization (ACO) algorithm."""

    def __init__(self, gestor, num_hormigas: int, alpha: float, beta: float,
                 rho: float, iteraciones: int):
        """
        Inicializa el algoritmo ACO.

        Args:
            gestor (Manager): Gestor del sistema.
            num_hormigas (int): Número de hormigas.
            alpha (float): Influencia de la feromona.
            beta (float): Influencia de la heurística.
            rho (float): Tasa de evaporación de la feromona.
            iteraciones (int): Número de iteraciones.
        """
        super().__init__(gestor)
        self.num_hormigas = num_hormigas
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.iteraciones = iteraciones

        # Se inicializarán en aplicar_estrategia
        self.feromonas = None
        self.mejor_particion = None
        self.mejor_perdida = float('inf')

    def inicializar_feromonas(self, num_nodos: int):
        """Inicializa la matriz de feromonas NxN."""
        self.feromonas = np.ones((num_nodos, num_nodos), dtype=float)

    def calcular_probabilidades(self, nodo_actual: int, nodos_disponibles: list,
                                heuristica: np.ndarray):
        """
        Calcula las probabilidades de transición para las hormigas.

        Args:
            nodo_actual (int): Nodo actual de la hormiga.
            nodos_disponibles (list): Nodos que aún no han sido visitados.
            heuristica (np.ndarray): Matriz de heurística.

        Returns:
            list: Probabilidades de transición.
        """
        probs = []
        for nodo in nodos_disponibles:
            tau = self.feromonas[nodo_actual, nodo]
            eta = heuristica[nodo_actual, nodo]
            probs.append((tau ** self.alpha) * (eta ** self.beta))
        total = sum(probs)
        return [p / total for p in probs] if total > 0 else [1/len(probs)]*len(probs)

    def construir_solucion(self, heuristica: np.ndarray):
        """
        Construye una solución (recorrido) para una hormiga.

        Args:
            heuristica (np.ndarray): Matriz de heurística.

        Returns:
            list: Secuencia de nodos visitados.
        """
        num_nodos = heuristica.shape[0]
        solucion = []
        nodos_disponibles = list(range(num_nodos))
        # Elige nodo inicial al azar
        nodo_actual = np.random.choice(nodos_disponibles)
        solucion.append(nodo_actual)
        nodos_disponibles.remove(nodo_actual)

        while nodos_disponibles:
            probs = self.calcular_probabilidades(nodo_actual,
                                                 nodos_disponibles,
                                                 heuristica)
            nodo_siguiente = np.random.choice(nodos_disponibles, p=probs)
            solucion.append(nodo_siguiente)
            nodos_disponibles.remove(nodo_siguiente)
            nodo_actual = nodo_siguiente

        return solucion

    def actualizar_feromonas(self, soluciones: list, costos: list):
        """
        Actualiza la matriz de feromonas: evaporación y refuerzo.

        Args:
            soluciones (list): Lista de recorridos de hormigas.
            costos (list): Costos asociados a cada recorrido.
        """
        # Evaporación
        self.feromonas *= (1 - self.rho)

        # Refuerzo: cada hormiga deposita en su recorrido
        for sol, costo in zip(soluciones, costos):
            delta = 1.0 / (costo + 1e-9)
            for i in range(len(sol) - 1):
                u, v = sol[i], sol[i+1]
                self.feromonas[u, v] += delta
                self.feromonas[v, u] += delta

    @profile(context={TYPE_TAG: ACO_LABEL})
    def aplicar_estrategia(self, condiciones: str,
                           alcance: str,
                           mecanismo: str) -> Solution:
        """
        Aplica la estrategia ACO para encontrar una buena bipartición.

        Args:
            condiciones (str): Condiciones iniciales.
            alcance (str): Alcance del sistema.
            mecanismo (str): Mecanismo del sistema.

        Returns:
            Solution: Solución con la mejor bipartición encontrada.
        """
        # 1) Preparar el subsistema
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)

        # 2) Determinar N = m + n
        futuros = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos
        m, n = futuros.size, presentes.size
        N = m + n

        # 3) Inicializar estructura ACO
        self.inicializar_feromonas(N)
        heuristica = np.random.rand(N, N)

        # 4) Bucle principal de iteraciones
        for _ in range(self.iteraciones):
            rutas = []
            costos = []
            for _ in range(self.num_hormigas):
                ruta = self.construir_solucion(heuristica)
                costo = self.calcular_costo(ruta)
                rutas.append(ruta)
                costos.append(costo)
                # Actualizar mejor global
                if costo < self.mejor_perdida:
                    self.mejor_perdida = costo
                    self.mejor_particion = ruta

            self.actualizar_feromonas(rutas, costos)

        # 5) Formatear la mejor partición encontrada
        particion_formateada = self.formatear_particion(self.mejor_particion)

        # 6) Devolver resultado
        return Solution(
            estrategia=ACO_LABEL,
            perdida=self.mejor_perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=self.sia_subsistema.distribucion_marginal(),
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=particion_formateada,
        )

    def calcular_costo(self, solucion: list) -> float:
        """
        Calcula el costo de una solución.

        Args:
            solucion (list): Recorrido de la hormiga.

        Returns:
            float: Costo asociado (reemplazar con lógica real).
        """
        # TODO: Implementar función de costo real usando tu métrica EMD
        return np.random.random()

    def formatear_particion(self, particion: list) -> str:
        """
        Formatea la partición (lista de nodos) para su representación.

        Args:
            particion (list): Lista de índices de nodos.

        Returns:
            str: Texto con la partición.
        """
        return " ".join(map(str, particion))
