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
        bits = ""
        for i in range(self.N):
            if mecanismo[i] == "1":
                bits += self.sia_gestor.estado_inicial[i]

        self.i0 = int(bits, 2)

        #calculo cual es el mejor cubo para esto hago el promedio por columna de la matriz tpm y escojo el que tiene menor diferencia con su valor de estado inicial
        # Versión vectorizada con NumPy
        tpm = self.sia_cargar_tpm()

        # Calcular promedio por columna
        column_averages = np.mean(tpm, axis=0)

        # Crear un vector de referencia basado en el estado inicial
        # (1 si el bit es 1, 0 si es 0)
        reference_bits = self.sia_dists_marginales

        # Calcular diferencias absolutas
        column_diffs = np.abs(reference_bits - column_averages)

        # Encontrar columna con menor diferencia
        min_diff_col = np.argmin(column_diffs)
        print(f"La columna con menor diferencia es: {min_diff_col}")
        print(f"Valor de la diferencia: {column_diffs[min_diff_col]}")


        one_bit_states = [
            self.i0 ^ (1 << b) for b in range(len(bits))
        ]

        # me aseguro de no duplicar en caso de N=1
        j_candidates = list(dict.fromkeys(one_bit_states))
        
        
        # 3) calculo T[v][j] = t(i0→j) sólo para esos j
        self.T = {}  # variable → { j → coste }
        hilos = []
        for i, nc in enumerate(self.sia_subsistema.ncubos):
            ##necesito inicializar hilo para cada cubo
            hilo = threading.Thread(target=self.proccess_nc, args=(nc, self.sia_subsistema.indices_ncubos[i], j_candidates))
            hilos.append(hilo)
            hilo.start()

        for hilo in hilos:
            hilo.join()

            # 4) sumo costes por cada j y elijo el j con suma mínima
        sum_per_j = {}
        for row in self.T.values():
            for j, cost in row.items():
                sum_per_j[j] = sum_per_j.get(j, 0.0) + cost

        best_j = min(sum_per_j, key=sum_per_j.get)

        # T es un diccionario  e diccionarios, quiero la suma de cada diccionario interno de T
        sum_per_dict = {}
        for k, row in self.T.items():
            for j, cost in row.items():
                sum_per_dict[k] = sum_per_dict.get(k, 0.0) + cost

        # 5) saco qué bits cambiaron, y qué variables tienen coste ≠ 0
        changed_bits = [
            self.sia_subsistema.dims_ncubos[idx]
            for idx in range(len(bits))
            if ((self.i0 >> idx) & 1) != ((best_j >> idx) & 1)
        ]

        nonzero = [
            v
            for v, row in self.T.items()
            if abs(row.get(best_j, 0.0)) > 1e-12
        ]

        

        # 6) construyo subalcance / submecanismo y biparto
        submecanismo = np.array(changed_bits, dtype=np.int8)
        subalcance = np.array(nonzero,      dtype=np.int8)
        part = self.sia_subsistema.bipartir(subalcance, submecanismo)
        dist = part.distribucion_marginal()

        # 7) calculo φ para la partición resultante
        phi = emd_efecto(dist, self.sia_dists_marginales)

        # 8) formateo cadena de partición
        seleccion = [(1, i) for i in subalcance] + [(0, i)
                                                    for i in submecanismo]
        complemento = (
            [(1, i) for i in self.sia_subsistema.indices_ncubos if i not in subalcance] +
            [(0, i)
             for i in self.sia_subsistema.dims_ncubos if i not in submecanismo]
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


