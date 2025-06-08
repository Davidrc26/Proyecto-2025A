import time
from src.constants.base import NET_LABEL, TYPE_TAG
from src.constants.models import GEOMETRIC_ANALYSIS_TAG, GEOMETRIC_STRAREGY_TAG, GA_LABEL
from src.models.base.sia import SIA
from src.controllers.manager import Manager
from src.middlewares.profile import profiler_manager, profile
from src.middlewares.slogger import SafeLogger
from src.models.core.solution import Solution
from typing import List, Tuple
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
        self.pop_size = -1
        self.generations: int = 10
        self.crossover_rate: float = 0.7
        self.mutation_rate: float = 0.05
        self.elitism: int = 1
        self.patience: int = 10
        self.N: int = 0
        self.m: int = 0
        self._cost_cache = {}
        self.T = {}
        self._cache_genetic: dict = {} 

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condiciones: str,
        alcance: str,
        mecanismo: str
    ) -> Solution:
        # 1) preparo subsistema y estado inicial
        self.pop_size = len(condiciones)
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)
        self.start_time = time.time()
        self.N = len(self.sia_gestor.estado_inicial)
        bits = "".join(
            self.sia_gestor.estado_inicial[i]
            for i in range(self.N)
            if mecanismo[i] == "1"
        )
        self.i0 = int(bits[::-1], 2)
    
        # --- FASE 1: primer cubo según TPM ---
        tpm = self.sia_subsistema.collapsed_ncubes
        values = [t[1] for t in tpm]
        referencia = self.sia_dists_marginales
        diffs = np.abs(referencia - values)
        best_col = np.argmin(diffs)
        # guardar alcance/mecanismo de fase 1
        alcance_fase1   = np.array([tpm[best_col][0]], dtype=np.int8)
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
                tiempo_total=time.time() - self.start_time,
                particion=particion_str1,
            )
    
        # --- FASE 2: cómputo completo si phi_fase1 != 0 ---
        one_bit_states = [self.i0 ^ (1 << b) for b in range(len(bits))]
        j_candidates = list(dict.fromkeys(one_bit_states))
    
        self.T = {}
        for i, nc in enumerate(self.sia_subsistema.ncubos):
            self.proccess_nc(
                nc, self.sia_subsistema.indices_ncubos[i], j_candidates)
    
        # sumar costes y elegir best_j
        sum_per_j = {}
        for row in self.T.values():
            for j, cost in row.items():
                sum_per_j[j] = sum_per_j.get(j, 0.0) + cost
        best_j = min(sum_per_j, key=sum_per_j.get)

        #sumar costes de cada diccionario 
        # suma_por_clave = {clave: sum(subdic.values()) for clave, subdic in self.T.items()}
        # suma_ordenada = dict(sorted(suma_por_clave.items(), key=lambda item: item[1]))
        # print(suma_ordenada)


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
                tiempo_total=time.time() - self.start_time,
                particion=particion_str2,
            )
    
        # Si ambos valores de phi no son cero, retornar el de menor valor
        if phi_fase1 < phi_fase2:
            #Genetico (columnas-futuro)
            solution = self.explorate_genetic(best_col, phi_fase1)

            if solution.perdida >= phi_fase1:
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
                    tiempo_total=time.time() - self.start_time,
                    particion=particion_str,
                )
            else:
                seleccion = solution.particion
            
                particion_str = fmt_biparte_q(seleccion, complemento)
                return Solution(
                    estrategia="Geometric",
                    perdida=solution.perdida,
                    distribucion_subsistema=referencia,
                    distribucion_particion=solution.distribucion_particion,
                    tiempo_total=time.time() - self.start_time,
                    particion=particion_str,
                ) 
        else:
            solution = self.explorate_neighbors(best_j=best_j, changed_bits=changed_bits)
            #Calcular los vecinos del mejor (filas- presente)
            if solution.perdida >= phi_fase2:
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
                tiempo_total=time.time() - self.start_time,
                particion=particion_str2,
            )
            else:
                return solution


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

    
    def nodes_complement(self, seleccion: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return [v for v in self.vertices if v not in seleccion]

    def _evaluar_individuo(self, key: Tuple[int]) -> float:
        ind = np.array(key, dtype=np.int8)
        count = ind.sum()
        if count == 0 or count == self.N:
            return -1e9
        phi = self._cache_genetic.get(key)
        if phi is None:
            subalcance = np.where(ind[:self.m] == 1)[0]
            submecanismo = np.where(ind[self.m:] == 1)[0]
            real_alcance = self.sia_subsistema.indices_ncubos[subalcance]
            real_mecanismo = self.sia_subsistema.dims_ncubos[submecanismo]
            seleccion_debug = [(1, int(idx)) for idx in real_alcance] + [(0, int(idx)) for idx in real_mecanismo]
            print(f"Evaluando partición: {fmt_biparte_q(seleccion_debug, self.nodes_complement(seleccion_debug))}")
            part = self.sia_subsistema.bipartir(
                np.array(real_alcance, dtype=np.int8),
                np.array(real_mecanismo, dtype=np.int8)
            )
            dist = part.distribucion_marginal()
            phi = emd_efecto(dist, self.dists_ref)
            print(phi)
            self._cache_genetic[key] = phi
        return -phi

    def explorate_genetic(self, best_ncube: int, best_loss: float) -> Solution:
        futuros = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos
        self.m = futuros.size
        n = presentes.size
        self.N = len(futuros) + len(presentes)
        self.indices_futuro = futuros
        self.indices_presente = presentes
        self.dists_ref = self.sia_dists_marginales
        self.vertices = [(1, int(idx)) for idx in futuros] + [(0, int(idx)) for idx in presentes]

        poblacion = np.zeros((self.pop_size, self.N), dtype=np.int8)
        
        # Individuo base: solo el best_ncube activado
        poblacion[0, best_ncube] = 1
        
        for i in range(1, self.pop_size):
            # Siempre incluir el best_ncube
            poblacion[i, best_ncube] = 1
            # Elegir cuántas posiciones adicionales activar (1 o 2)
            k = np.random.choice([1, 2])
            # Elegir índices adicionales distintos de best_ncube
            available_indexes = [j for j in range(self.N) if j != best_ncube]
            if len(available_indexes) >= k:
                adicionales = np.random.choice(available_indexes, k, replace=False)
                poblacion[i, adicionales] = 1
        no_improve = 0
        prev_phi = float('inf')
        global_best_phi = best_loss
        global_best_key = None

        for gen in range(self.generations):
            improve = False
            print(f"Generación {gen+1}/{self.generations}")
            keys = [tuple(ind.tolist()) for ind in poblacion]
            unique_keys = list(dict.fromkeys(keys))
            unique_fitness = {k: self._evaluar_individuo(k) for k in unique_keys}
            fitness_vals = np.array([unique_fitness[k] for k in keys])

            for k, f in unique_fitness.items():
                phi_val = -f
                if phi_val < global_best_phi:
                    global_best_phi = phi_val
                    global_best_key = k
                    improve = True


            gen_phi = min(-val for val in fitness_vals)
            # Early stop sólo por paciencia (mayor exploración)
            if not improve:
                print(f"Early stop por paciencia en gen {gen+1}")
                break
            if gen_phi < prev_phi:
                no_improve = 0
                prev_phi = gen_phi
            else:
                no_improve += 1

            elite_idx = np.argsort(fitness_vals)[-self.elitism:]
            elites = [poblacion[i].copy() for i in elite_idx]
            padres = np.empty_like(poblacion)
            for i in range(self.pop_size):
                a, b = np.random.choice(self.pop_size, 2, replace=False)
                padres[i] = poblacion[a] if fitness_vals[a] > fitness_vals[b] else poblacion[b]
            hijos = padres.copy()
            for i in range(0, self.pop_size-1, 2):
                if np.random.rand() < self.crossover_rate:
                    pt = np.random.randint(1, self.N)
                    hijos[i, :pt], hijos[i+1, :pt] = padres[i+1, :pt].copy(), padres[i, :pt].copy()
            for i in range(self.pop_size):
                if np.random.rand() < self.mutation_rate:
                    flip_count = np.random.choice([1, 2])
                    idxs = np.random.choice(self.N, size=flip_count, replace=False)
                    hijos[i, idxs] = 1 - hijos[i, idxs]
            poblacion = np.vstack((elites, hijos[self.elitism:]))

        # Usar mejor global
        if global_best_key is None:
            best_ind = poblacion[0]
            best_phi = -fitness_vals[0]
        else:
            best_ind = np.array(global_best_key, dtype=np.int8)
            best_phi = global_best_phi

        subalcance = np.where(best_ind[:self.m] == 1)[0]
        submecanismo = np.where(best_ind[self.m:] == 1)[0]
        part = self.sia_subsistema.bipartir(
            np.array(self.indices_futuro[subalcance], dtype=np.int8),
            np.array(self.indices_presente[submecanismo], dtype=np.int8)
        )
        dist = part.distribucion_marginal()

        # Construir partición final
        seleccion = [(1, int(idx)) for idx in self.indices_futuro[subalcance]] + \
                    [(0, int(idx)) for idx in self.indices_presente[submecanismo]]
        complemento = self.nodes_complement(seleccion)
        particion_str = fmt_biparte_q(seleccion, complemento)

        return Solution(
            estrategia=GA_LABEL,
            perdida=best_phi,
            distribucion_subsistema=self.dists_ref,
            distribucion_particion=dist,
            tiempo_total=time.time() - self.start_time,
            particion=particion_str,
        )

    def explorate_neighbors(self, best_j: int, changed_bits: list) -> Solution:
        """
        Fase 3: expansión local desde best_j,
        cambiando un bit adicional no modificado en i0 → best_j.
        """
        # Asegúrate de haber guardado best_j en self.best_j tras Fase 2
        # 1) Detectar bits ya cambiados en i0 → best_j
        bits_cambiados = {
            b for b in range(self.N)
            if ((self.i0 >> b) & 1) != ((best_j >> b) & 1)
        }

        # 2) Generar nuevos vecinos cambiando un bit distinto
        j_candidates = [
            best_j ^ (1 << b)
            for b in range(self.N)
            if b not in bits_cambiados
        ]
        # Eliminar posibles duplicados conservando orden
        j_candidates = list(dict.fromkeys(j_candidates))

        # 3) Calcular costes t(i0 → j) para cada vecino
        self.T.clear()
        for i, nc in enumerate(self.sia_subsistema.ncubos):
            self.proccess_nc(nc, self.sia_subsistema.indices_ncubos[i], j_candidates)

        # 4) Sumatoria de costes por cada j y selección de best_j2
        sum_per_j = {}
        for row in self.T.values():
            for j, cost in row.items():
                sum_per_j[j] = sum_per_j.get(j, 0.0) + cost
        best_j2 = min(sum_per_j, key=sum_per_j.get)

        # 5) Determinar bits cambiados adicionales y variables con coste no nulo
        changed_bits2 = [
            self.sia_subsistema.dims_ncubos[idx]
            for idx in range(self.N)
            if ((best_j >> idx) & 1) != ((best_j2 >> idx) & 1)
        ]

        changed_bits2.extend(changed_bits)
        nonzero2 = [
            v for v, row in self.T.items()
            if abs(row.get(best_j2, 0.0)) > 1e-12
        ]

        # 6) Bipartición y cálculo de phi
        alcance   = np.array(nonzero2,    dtype=np.int8)
        mecanismo = np.array(changed_bits2, dtype=np.int8)
        part      = self.sia_subsistema.bipartir(alcance, mecanismo)
        dist      = part.distribucion_marginal()
        phi       = emd_efecto(dist, self.sia_dists_marginales)

        seleccion1 = [(1, i) for i in alcance] + [(0, i) for i in mecanismo]
        complemento1 = (
                [(1, i) for i in self.sia_subsistema.indices_ncubos   if i not in alcance] +
                [(0, i) for i in self.sia_subsistema.dims_ncubos         if i not in mecanismo]
            )
        particion_str1 = fmt_biparte_q(seleccion1, complemento1)
        return Solution(
                estrategia="Geometric",
                perdida=phi,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=dist,
                tiempo_total=time.time() - self.start_time,
                particion=particion_str1,
        )




