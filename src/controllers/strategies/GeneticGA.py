import time
import numpy as np
from typing import List, Tuple
from multiprocessing import Pool

from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile
from src.funcs.base import emd_efecto, ABECEDARY
from src.funcs.format import fmt_biparte_q
from src.constants.models import GA_LABEL, GA_STRATEGY_TAG
from src.constants.base import TYPE_TAG, NET_LABEL
from src.controllers.manager import Manager
from src.models.base.sia import SIA
from src.models.core.solution import Solution


class GeneticGA(SIA):
    """
    Algoritmo Genético optimizado para encontrar particiones pequeñas y eficientes.
    """

    def __init__(
        self,
        gestor: Manager,
        pop_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.05,
        elitism: int = 1,
        patience: int = 10,
        verbose: bool = False,
    ):
        super().__init__(gestor)
        session_name = f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}_GA"
        profiler_manager.start_session(session_name)

        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.patience = patience
        self.verbose = verbose

        self.N: int = 0
        self.m: int = 0
        self.dists_ref: np.ndarray = None  # type: ignore
        self.indices_futuro: np.ndarray = None  # type: ignore
        self.indices_presente: np.ndarray = None  # type: ignore
        self.vertices: List[Tuple[int, int]] = []
        self._cache: dict = {}

        self.logger = SafeLogger(GA_STRATEGY_TAG)

    def nodes_complement(self, seleccion: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return [v for v in self.vertices if v not in seleccion]

    def _evaluar_individuo(self, key: Tuple[int]) -> float:
        ind = np.array(key, dtype=np.int8)
        ones = ind.sum()
        if ones == 0 or ones == self.N:
            return -1e9

        phi = self._cache.get(key)
        if phi is None:
            subalcance = np.where(ind[:self.m] == 1)[0]
            submecanismo = np.where(ind[self.m:] == 1)[0]
            part = self.sia_subsistema.bipartir(
                np.array(subalcance, dtype=np.int8),
                np.array(submecanismo, dtype=np.int8)
            )
            dist = part.distribucion_marginal()
            phi = emd_efecto(dist, self.dists_ref)
            self._cache[key] = phi

        return -phi

    @profile(context={TYPE_TAG: GA_LABEL})
    def aplicar_estrategia(
        self,
        condiciones: str,
        alcance: str,
        mecanismo: str,
    ) -> Solution:
        # Preparar subsistema
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)
        
        futuros = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos
        self.m = futuros.size
        n = presentes.size
        self.N = self.m + n
        self.indices_futuro = futuros
        self.indices_presente = presentes
        self.dists_ref = self.sia_dists_marginales
        self.vertices = [(1, int(idx)) for idx in futuros] + [(0, int(idx)) for idx in presentes]

        # Inicialización sesgada de población
        poblacion = np.zeros((self.pop_size, self.N), dtype=np.int8)
        for i in range(self.pop_size):
            if np.random.rand() < 0.7:
                k = np.random.choice([1, 2, 3])
                idx = np.random.choice(self.N, k, replace=False)
                poblacion[i, idx] = 1
            else:
                poblacion[i] = np.random.randint(0, 2, size=self.N, dtype=np.int8)

        best_phi = float('inf')
        best_ind = np.zeros(self.N, dtype=np.int8)
        start_time = time.time()

        no_improve = 0
        prev_phi = best_phi
        pool = Pool(processes=8)

        try:
            for gen in range(self.generations):
                keys = [tuple(ind.tolist()) for ind in poblacion]
                fitness_vals = pool.map(self._evaluar_individuo, keys)
                fitness_vals = np.array(fitness_vals)

                gen_best_idx = np.argmax(fitness_vals)
                gen_best_fitness = fitness_vals[gen_best_idx]
                if -gen_best_fitness < best_phi:
                    best_phi = -gen_best_fitness
                    best_ind = poblacion[gen_best_idx].copy()

                if self.verbose:
                    self.logger.info(f"Gen {gen+1}/{self.generations}: best_phi={best_phi:.6f}")

                # Early stopping por mejora mínima
                if abs(prev_phi - best_phi) < 1e-5:
                    if self.verbose:
                        self.logger.info(f"Early stop por mejora mínima en generación {gen+1}")
                    break

                # Early stopping normal
                if best_phi < prev_phi:
                    no_improve = 0
                    prev_phi = best_phi
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        if self.verbose:
                            self.logger.info(f"Early stop por paciencia en generación {gen+1}")
                        break

                # Reducción dinámica de población
                if no_improve >= self.patience // 2 and self.pop_size > 10:
                    self.pop_size = max(self.pop_size // 2, 10)
                    poblacion = poblacion[:self.pop_size]
                    fitness_vals = fitness_vals[:self.pop_size]

                # Torneo
                padres = np.empty_like(poblacion)
                for i in range(self.pop_size):
                    a, b = np.random.choice(self.pop_size, 2, replace=False)
                    padres[i] = poblacion[a] if fitness_vals[a] > fitness_vals[b] else poblacion[b]

                # Cruce
                hijos = padres.copy()
                for i in range(0, self.pop_size - 1, 2):
                    if np.random.rand() < self.crossover_rate:
                        pt = np.random.randint(1, self.N)
                        hijos[i, :pt], hijos[i+1, :pt] = padres[i+1, :pt].copy(), padres[i, :pt].copy()

                # Mutación localizada
                for i in range(self.pop_size):
                    if np.random.rand() < 0.5:
                        flip_count = np.random.choice([1, 2])
                        idx_to_flip = np.random.choice(self.N, size=flip_count, replace=False)
                        hijos[i, idx_to_flip] = 1 - hijos[i, idx_to_flip]

                # Elitismo
                elite_idx = np.argsort(fitness_vals)[-self.elitism:]
                elites = poblacion[elite_idx]
                poblacion = hijos
                poblacion[:self.elitism] = elites

        finally:
            pool.close()
            pool.join()

        # Construir partición
        seleccion = []
        for i in range(self.N):
            if best_ind[i] == 1:
                seleccion.append((1, int(self.indices_futuro[i])) if i < self.m else
                                  (0, int(self.indices_presente[i - self.m])))
        complemento = self.nodes_complement(seleccion)
        particion_str = fmt_biparte_q(seleccion, complemento)

        return Solution(
            estrategia=GA_LABEL,
            perdida=best_phi,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=None,
            tiempo_total=time.time() - start_time,
            particion=particion_str,
        )
