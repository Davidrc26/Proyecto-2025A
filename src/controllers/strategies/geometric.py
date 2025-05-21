import time
from src.constants.base import NET_LABEL, TYPE_TAG
from src.constants.models import GEOMETRIC_ANALYSIS_TAG, GEOMETRIC_STRAREGY_TAG
from src.models.base.sia import SIA
from src.controllers.manager import Manager
from src.middlewares.profile import profiler_manager, profile
from src.middlewares.slogger import SafeLogger
from src.models.core.solution import Solution
import threading

import numpy as np
from src.funcs.base import emd_efecto  # lo necesitarás después
from src.funcs.format import fmt_biparte_q


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count('1')


class GeometricSIA(SIA):
    """
    Estrategia geométrica-topológica (extensión paso 1):
    - calcula t(i0→j) para flips de 1 bit
    - y además para el “complemento completo” (todos los bits invertidos)
    """

    def __init__(self, gestor: Manager, **kwargs):
        super().__init__(gestor)
        session_name = f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}_GEOM"
        profiler_manager.start_session(session_name)
        self.logger = SafeLogger(GEOMETRIC_STRAREGY_TAG)
        self._cost_cache = {}
        self.T = {}

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condiciones: str,
        alcance: str,
        mecanismo: str
    ) -> Solution:
        # 1) preparo subsistema y estado inicial
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)
        start_time = time.time()
        self.N = len(self.sia_gestor.estado_inicial)
        bits = "".join(
            self.sia_gestor.estado_inicial[i]
            for i in range(self.N)
            if mecanismo[i] == "1"
        )
        self.i0 = int(bits, 2)
    
        # --- FASE 1: primer cubo según TPM ---
        tpm = self.sia_cargar_tpm()
        medias = np.mean(tpm, axis=0)
        referencia = self.sia_dists_marginales
        diffs = np.abs(referencia - medias)
        mejor_col = np.argmax(diffs)
    
        # guardar alcance/mecanismo de fase 1
        alcance_fase1   = np.array([mejor_col], dtype=np.int8)
        mecanismo_fase1 = np.array([],           dtype=np.int8)
        part1 = self.sia_subsistema.bipartir(alcance_fase1, mecanismo_fase1)
        dist1 = part1.distribucion_marginal()
        phi_fase1 = emd_efecto(dist1, referencia)
    
        # si pérdida es 0, retornamos ya
        if phi_fase1 == 0:
            seleccion1 = [(1, i) for i in alcance_fase1] + [(0, i) for i in mecanismo_fase1]
            complemento1 = (
                [(1, i) for i in self.sia_subsistema.indices_ncubos   if i not in alcance_fase1] +
                [(0, i) for i in self.sia_subsistema.dims_ncubos         if i not in mecanismo_fase1]
            )
            particion_str1 = fmt_biparte_q(seleccion1, complemento1)
            return Solution(
                estrategia="Geometric",
                perdida=phi_fase1,
                distribucion_subsistema=referencia,
                distribucion_particion=dist1,
                tiempo_total=time.time() - start_time,
                particion=particion_str1,
            )
    
        # --- FASE 2: cómputo completo si phi_fase1 != 0 ---
        one_bit_states = [self.i0 ^ (1 << b) for b in range(len(bits))]
        j_candidates = list(dict.fromkeys(one_bit_states))
    
        self.T = {}
        hilos = []
        for i, nc in enumerate(self.sia_subsistema.ncubos):
            hilo = threading.Thread(
                target=self.proccess_nc,
                args=(nc, self.sia_subsistema.indices_ncubos[i], j_candidates)
            )
            hilos.append(hilo); hilo.start()
        for hilo in hilos:
            hilo.join()
    
        # sumar costes y elegir best_j
        sum_per_j = {}
        for row in self.T.values():
            for j, cost in row.items():
                sum_per_j[j] = sum_per_j.get(j, 0.0) + cost
        best_j = min(sum_per_j, key=sum_per_j.get)
    
        # determinar bits cambiados y variables con coste
        changed_bits = [
            self.sia_subsistema.dims_ncubos[idx]
            for idx in range(len(bits))
            if ((self.i0 >> idx) & 1) != ((best_j >> idx) & 1)
        ]
        nonzero = [
            v for v, row in self.T.items()
            if abs(row.get(best_j, 0.0)) > 1e-12
        ]
    
        # bipartir fase 2
        alcance_fase2   = np.array(nonzero,      dtype=np.int8)
        mecanismo_fase2 = np.array(changed_bits,  dtype=np.int8)
        part2 = self.sia_subsistema.bipartir(alcance_fase2, mecanismo_fase2)
        dist2 = part2.distribucion_marginal()
        phi_fase2 = emd_efecto(dist2, referencia)
    
        # si pérdida en fase 2 es 0, retornamos también
        if phi_fase2 == 0:
            seleccion2 = [(1, i) for i in alcance_fase2] + [(0, i) for i in mecanismo_fase2]
            complemento2 = (
                [(1, i) for i in self.sia_subsistema.indices_ncubos   if i not in alcance_fase2] +
                [(0, i) for i in self.sia_subsistema.dims_ncubos         if i not in mecanismo_fase2]
            )
            particion_str2 = fmt_biparte_q(seleccion2, complemento2)
            return Solution(
                estrategia="Geometric",
                perdida=phi_fase2,
                distribucion_subsistema=referencia,
                distribucion_particion=dist2,
                tiempo_total=time.time() - start_time,
                particion=particion_str2,
            )
    
        # Si ambos valores de phi no son cero, retornar el de menor valor
        if phi_fase1 < phi_fase2:
            seleccion = [(1, i) for i in alcance_fase1] + [(0, i) for i in mecanismo_fase1]
            complemento = (
                [(1, i) for i in self.sia_subsistema.indices_ncubos   if i not in alcance_fase1] +
                [(0, i) for i in self.sia_subsistema.dims_ncubos         if i not in mecanismo_fase1]
            )
            particion_str = fmt_biparte_q(seleccion, complemento)
            return Solution(
                estrategia="Geometric",
                perdida=phi_fase1,
                distribucion_subsistema=referencia,
                distribucion_particion=dist1,
                tiempo_total=time.time() - start_time,
                particion=particion_str,
            )
        else:
            # finalmente, retorno solución de fase 2 (tiene menor phi o igual)
            seleccion = [(1, i) for i in alcance_fase2] + [(0, i) for i in mecanismo_fase2]
            complemento = (
                [(1, i) for i in self.sia_subsistema.indices_ncubos   if i not in alcance_fase2] +
                [(0, i) for i in self.sia_subsistema.dims_ncubos         if i not in mecanismo_fase2]
            )
            particion_str = fmt_biparte_q(seleccion, complemento)
            return Solution(
                estrategia="Geometric",
                perdida=phi_fase2,
                distribucion_subsistema=referencia,
                distribucion_particion=dist2,
                tiempo_total=time.time() - start_time,
                particion=particion_str,
            )

    def _calcular_transicion_coste(
        self,
        i: int,
        j: int,
        X: np.ndarray
    ) -> float:
        """
        Cálculo recursivo t(i,j) = 2^{-dH}·( |X[i]-X[j]| + Σ t(k,j) )
        """
        key = (i, j, id(X))
        if key in self._cost_cache:
            return self._cost_cache[key]

        d = hamming_distance(i, j)
        gamma = 2 ** (-d)
        base = abs(X[i] - X[j])
        suma = 0.0
        if d > 1:
            for b in range(self.N):
                k = i ^ (1 << b)
                if hamming_distance(k, j) == d - 1:
                    suma += self._calcular_transicion_coste(k, j, X)

        cost = gamma * (base + suma)
        self._cost_cache[key] = cost
        return cost

    def proccess_nc(self, nc, key, j_candidates):
        X = nc.data.flatten()
        self._current_var = key
        self._cost_cache.clear()
        row = {}
        for j in j_candidates:
            row[j] = self._calcular_transicion_coste(self.i0, j, X)
        self.T[key] = row


