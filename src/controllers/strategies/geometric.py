from src.funcs.base import emd_efecto
from src.constants.base import NET_LABEL, TYPE_TAG
from src.constants.models import GEOMETRIC_ANALYSIS_TAG, GEOMETRIC_STRAREGY_TAG
from src.models.base.sia import SIA
from src.controllers.manager import Manager
from src.middlewares.profile import profiler_manager, profile
from src.middlewares.slogger import SafeLogger
from src.models.core.solution import Solution
from src.funcs.format import fmt_biparte_q


import numpy as np
from itertools import combinations


def hamming_distance(a: int, b: int) -> int:
    """Distancia de Hamming entre enteros a y b."""
    return bin(a ^ b).count('1')

class GeometricSIA(SIA):
    """
    Estrategia geométrica-topológica para bipartición óptima.
    Implementa el cálculo recursivo de costos y selección de partición.
    """

    def __init__(self, gestor: Manager, **kwargs):
        super().__init__(gestor)
        session_name = f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}_GEOM"
        profiler_manager.start_session(session_name)
        self.logger = SafeLogger(GEOMETRIC_STRAREGY_TAG)
        # Caché para costos t(i,j) por variable
        self._cost_cache = {}

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(self, condiciones: str, alcance: str, mecanismo: str) -> Solution:
        """
        Ejecuta el algoritmo geométrico completo para encontrar la bipartición óptima.
        """
        # 1) Preparar subsistema con condicionamientos
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)
        # 2) Calcular tabla de costos T[v][i][j]
        T = self._calcular_tabla_costos()
        # 3) Identificar biparticiones candidatas
        candidates = self._identificar_candidatos(T)
        # 4) Evaluar y seleccionar mejor bipartición
        best = None
        best_phi = np.inf
        for sel in candidates:
            # sel: tuple de índices de variable en partición activa
            comp = [v for v in range(self.N) if v not in sel]
            part = self.sia_subsistema.bipartir(
                np.array(sel, dtype=np.int8),
                np.array(comp, dtype=np.int8)
            )
            dist_p = part.distribucion_marginal()
            phi = emd_efecto(dist_p, self.sia_dists_marginales)
            if phi < best_phi:
                best_phi = phi
                best = sel
                best_dist = dist_p
        # 5) Formatear solución
        seleccion = [(1,i) for i in best]
        complemento = [(1,i) for i in comp]
        particion_str = fmt_biparte_q(seleccion, complemento)
        return Solution(
            estrategia="Geometric",
            perdida=best_phi,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=best_dist,
            tiempo_total=0.0,
            particion=particion_str,
        )

    def _calcular_tabla_costos(self):
        """
        Construye T: dict variable->matrix[size x size] con costos t(i,j).
        """
        n = len(self.sia_gestor.estado_inicial)
        size = 1 << n
        ncubes = self.sia_subsistema.ncubos
        T = {}
        for v, nc in enumerate(ncubes):
            T[v] = np.zeros((size, size), dtype=float)
            for i in range(size):
                for j in range(size):
                    T[v][i,j] = self._calcular_transicion_coste(i, j, nc.data.flatten())
        return T

    def _calcular_transicion_coste(self, i: int, j: int, X: np.ndarray) -> float:
        """
        Calcula recursivamente t(i,j) según la ecuación 5.1.
        X: vector de probabilidades para esta variable, longitud 2^n.
        """
        key = (i,j,id(X))
        if key in self._cost_cache:
            return self._cost_cache[key]
        d = hamming_distance(i, j)
        gamma = 2**(-d)
        cost = gamma * abs(X[i] - X[j])
        # vecinos en caminos mínimos: estados que reducen dH
        if d > 1:
            for b in range(X.size.bit_length()):
                k = i ^ (1 << b)
                if hamming_distance(k, j) == d - 1:
                    cost += self._calcular_transicion_coste(k, j, X)
        self._cost_cache[key] = cost
        return cost

    def _identificar_candidatos(self, T):
        """
        Genera biparticiones candidatas basadas en heurística simple:
        por cada variable, incluirla o no.
        Para n_vars grandes, se podría filtrar por costos extremos.
        """
        n_vars = len(self.sia_subsistema.ncubos)
        # aquí tomamos todas las particiones con un solo bit activo y sus complementos
        cands = []
        for k in range(1, n_vars):
            for combo in combinations(range(n_vars), k):
                cands.append(combo)
        return cands
