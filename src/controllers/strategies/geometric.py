import time
from typing import List, Tuple
from src.constants.base import NET_LABEL, TYPE_TAG
from src.constants.models import GEOMETRIC_ANALYSIS_TAG, GEOMETRIC_STRAREGY_TAG
from src.models.base.sia import SIA
from src.controllers.manager import Manager
from src.middlewares.profile import profiler_manager, profile
from src.middlewares.slogger import SafeLogger
from src.models.core.solution import Solution

import numpy as np
import random
from src.funcs.base import emd_efecto  # lo necesitarás después
from src.funcs.format import fmt_biparte_q

def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count('1')


class GeometricSIA(SIA):
    """
    Estrategia geométrica-topológica (esqueleto paso 1):
    sólo calcula t(i0→j) para j a un solo bit de distancia.
    """

    def __init__(self, gestor: Manager, **kwargs):
        super().__init__(gestor)
        session_name = f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}_GEOM"
        profiler_manager.start_session(session_name)
        self.logger = SafeLogger(GEOMETRIC_STRAREGY_TAG)
        self._cost_cache = {}

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condiciones: str,
        alcance: str,
        mecanismo: str
    ) -> Solution:
        # --- 1) Preparo subsistema y calculo i0 ---
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)

        self.N = len(self.sia_gestor.estado_inicial)
        # entero del estado inicial (bit-string → int)
        bits = "".join(str(b) for b in self.sia_gestor.estado_inicial)
        self.i0 = int(bits, 2)
        start_time = time.time()

        # --- 2) Calculo sólo las distancias t(i0→j) para flips de 1 bit ---
        one_bit_states = [self.i0 ^ (1 << b) for b in range(self.N)]
        T = {}  # dict var → dict(j → t(i0,j))
        for v, nc in enumerate(self.sia_subsistema.ncubos):
            X = nc.data.flatten()
            self._current_var = v
            self._cost_cache.clear()
            row = {}
            for j in one_bit_states:
                row[j] = self._calcular_transicion_coste(self.i0, j, X)
            T[v] = row
        sum_per_j = {}
        for v, row in T.items():
            for j, cost in row.items():
                sum_per_j[j] = sum_per_j.get(j, 0.0) + cost

# 3) Encontrar el j que tenga la suma mínima
        best_j = min(sum_per_j, key=sum_per_j.get)
        best_sum = sum_per_j[best_j]
        changed_bits = [
            idx
            for idx in range(self.N)
            if ((self.i0 >> idx) & 1) != ((best_j >> idx) & 1)
        ]
        print(f"Bits cambiados de {self.i0:0{self.N}b} → {best_j:0{self.N}b}: {changed_bits}")
        nonzero = []
        for v, row in T.items():
            cost = row.get(best_j, 0.0)
        if abs(cost) > 1e-12:
            # si quieres el literal, en lugar de 'v' podrías usar:
            # var_name = self.sia_subsistema.ncubos[v].indice
            nonzero.append(v)
            print(f"Variables con costo no-cero en t(i0 → {best_j:0{self.N}b}): {nonzero}")
        # 1) convierte changed_bits y nonzero en arrays int8
        subalcance  = np.array(changed_bits, dtype=np.int8)
        submecanismo = np.array(nonzero,    dtype=np.int8)

        # 2) llama a bipartir con esos dos vectores
        part = self.sia_subsistema.bipartir(subalcance, submecanismo)

        # 3) obtén su distribución marginal
        dist = part.distribucion_marginal()

        # 4) a partir de aquí ya puedes evaluar el φ o seguir tu lógica
        phi = emd_efecto(dist, self.sia_dists_marginales)

        # 5) formatea la partición para devolverla
        seleccion  = ([(1, i) for i in subalcance] + [(0, i) for i in submecanismo])   # futuros que migran
        complemento = (
            [(1, i) for i in self.sia_subsistema.indices_ncubos   if i not in subalcance] +
            [(0, i) for i in self.sia_subsistema.dims_ncubos      if i not in submecanismo]
        )

        particion_str = fmt_biparte_q(seleccion, complemento)

        return Solution(
            estrategia="Geometric",
            perdida=phi,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist,
            tiempo_total=time.time() - start_time,
            particion=particion_str,
        )

    def _calcular_transicion_coste(
        self,
        i: int,
        j: int,
        X: np.ndarray
    ) -> float:
        key = (i, j, id(X))
        if key in self._cost_cache:
            return self._cost_cache[key]

        d = hamming_distance(i, j)
        gamma = 2 ** (-d)
        base = abs(X[i] - X[j])
        suma = 0.0
        if d > 1:
            # recursión sólo si haces caminos más largos, pero ahora no será el caso
            for b in range(self.N):
                k = i ^ (1 << b)
                if hamming_distance(k, j) == d - 1:
                    suma += self._calcular_transicion_coste(k, j, X)

        cost = gamma * (base + suma)
        self._cost_cache[key] = cost
        return cost
    
