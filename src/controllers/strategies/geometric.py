from src.constants.base import NET_LABEL, TYPE_TAG
from src.constants.models import GEOMETRIC_ANALYSIS_TAG, GEOMETRIC_STRAREGY_TAG
from src.models.base.sia import SIA
from src.controllers.manager import Manager
from src.middlewares.profile import profiler_manager, profile
from src.middlewares.slogger import SafeLogger
from src.models.core.solution import Solution

import numpy as np
import pandas as pd
from itertools import combinations
from src.funcs.base import emd_efecto
from src.funcs.format import fmt_biparte_q


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
    def aplicar_estrategia(
        self,
        condiciones: str,
        alcance: str,
        mecanismo: str
    ) -> Solution:
        """
        Ejecuta el algoritmo geométrico completo para encontrar la bipartición óptima.
        """
        # 1) Preparar subsistema con condicionamientos
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)

        # Inicializar dimensiones y número de estados
        self.N = len(self.sia_gestor.estado_inicial)
        self.size = 1 << self.N

        # 2) Calcular tabla de costos T[v][i][j]
        T = self._calcular_tabla_costos()

        # 3) Identificar biparticiones candidatas
        candidates = self._identificar_candidatos(T)

        # 4) Evaluar y seleccionar mejor bipartición
        best = None
        best_phi = np.inf
        best_dist = None
        for sel in candidates:
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
        seleccion = [(1, i) for i in best]
        complemento = [(1, i) for i in comp]
        particion_str = fmt_biparte_q(seleccion, complemento)

        return Solution(
            estrategia="Geometric",
            perdida=best_phi,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=best_dist,
            tiempo_total=0.0,
            particion=particion_str,
        )

    def _calcular_tabla_costos(self) -> dict:
        """
        Construye T: dict variable->matrix[size x size] con costos t(i,j).
        """
        T = {}
        for v, nc in enumerate(self.sia_subsistema.ncubos):
            M = np.zeros((self.size, self.size), dtype=float)
            X = nc.data.flatten()
            for i in range(self.size):
                for j in range(self.size):
                    M[i, j] = self._calcular_transicion_coste(i, j, X)
            T[v] = M
        return T

    def _calcular_transicion_coste(
        self,
        i: int,
        j: int,
        X: np.ndarray
    ) -> float:
        """
        Calcula recursivamente t(i,j) según la ecuación 5.1.
        X: vector de probabilidades para esta variable, longitud 2^n.
        """
        key = (i, j, id(X))
        if key in self._cost_cache:
            return self._cost_cache[key]

        d = hamming_distance(i, j)
        gamma = 2 ** (-d)
        base = abs(X[i] - X[j])
        sum_neighbors = 0.0
        if d > 1:
            # vecinos que disminuyen distancia Hamming en 1
            for bit in range(self.N):
                k = i ^ (1 << bit)
                if hamming_distance(k, j) == d - 1:
                    sum_neighbors += self._calcular_transicion_coste(k, j, X)

        cost = gamma * (base + sum_neighbors)
        self._cost_cache[key] = cost
        return cost

    def _identificar_candidatos(self, T: dict) -> list:
        """
        Genera biparticiones candidatas: todas las combinaciones no triviales.
        Para n_vars grandes, se podría filtrar por costos extremos.
        """
        n_vars = len(self.sia_subsistema.ncubos)
        cands = []
        for k in range(1, n_vars):
            for combo in combinations(range(n_vars), k):
                cands.append(combo)
        return cands

    def generar_tabla_T(
        self,
        condiciones: str,
        alcance: str,
        mecanismo: str
    ) -> pd.DataFrame:
        """
        Genera la tabla T de costos t(i,j) para todas las variables.
        Devuelve un DataFrame con multi-índice en columnas (variable, estado_j).
        Filas indexadas por estado_i (little-endian).
        """
        # Asegurar subsistema preparado
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)
        # Recalcular dimensiones si es necesario
        self.N = len(self.sia_gestor.estado_inicial)
        self.size = 1 << self.N

        T = self._calcular_tabla_costos()
        labels = [f"{i:0{self.N}b}" for i in range(self.size)]
        var_names = [nc.indice for nc in self.sia_subsistema.ncubos]

        # Construir DataFrame
        panels = [T[v] for v in range(len(var_names))]
        data = np.hstack(panels)
        col_arrays = [
            [vn for vn in var_names for _ in range(self.size)],
            labels * len(var_names)
        ]
        columns = pd.MultiIndex.from_arrays(col_arrays, names=["Variable", "Estado_j"])
        df = pd.DataFrame(data, index=labels, columns=columns)
        return df
