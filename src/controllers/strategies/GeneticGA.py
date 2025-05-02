import time
import numpy as np
from typing import List, Tuple

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
    Algoritmo Genético optimizado sin multihilos para facilitar el debug.
    Se rastrea el mejor global correctamente para evitar pérdida cero.
    """

    def __init__(
        self,
        gestor: Manager,
        pop_size: int,
        generations: int = 10,
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
        self._cache: dict = {}  # caché de fitness

        self.logger = SafeLogger(GA_STRATEGY_TAG)

    def nodes_complement(self, seleccion: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return [v for v in self.vertices if v not in seleccion]

    def _evaluar_individuo(self, key: Tuple[int]) -> float:
        ind = np.array(key, dtype=np.int8)
        count = ind.sum()
        if count == 0 or count == self.N:
            return -1e9
        phi = self._cache.get(key)
        if phi is None:
            subalcance = np.where(ind[:self.m] == 1)[0]
            submecanismo = np.where(ind[self.m:] == 1)[0]
            real_alcance = self.indices_futuro[subalcance]
            real_mecanismo = self.indices_presente[submecanismo]
            seleccion_debug = [(1, int(idx)) for idx in real_alcance] + [(0, int(idx)) for idx in real_mecanismo]
            print(f"Evaluando partición: {fmt_biparte_q(seleccion_debug, self.nodes_complement(seleccion_debug))}")
            part = self.sia_subsistema.bipartir(
                np.array(real_alcance, dtype=np.int8),
                np.array(real_mecanismo, dtype=np.int8)
            )
            dist = part.distribucion_marginal()
            phi = emd_efecto(dist, self.dists_ref)
            print(phi)
            self._cache[key] = phi
        return -phi

    @profile(context={TYPE_TAG: GA_LABEL})
    def aplicar_estrategia(
        self,
        condiciones: str,
        alcance: str,
        mecanismo: str,
    ) -> Solution:
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)
        print("aqui comienza")
        futuros = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos
        self.m = futuros.size
        n = presentes.size
        self.N = self.m + n
        self.indices_futuro = futuros
        self.indices_presente = presentes
        self.dists_ref = self.sia_dists_marginales
        self.vertices = [(1, int(idx)) for idx in futuros] + [(0, int(idx)) for idx in presentes]

        poblacion = np.zeros((self.pop_size, self.N), dtype=np.int8)
        for i in range(self.pop_size):
            if np.random.rand() < 0.7:
                k = np.random.choice([1, 2])
                idx = np.random.choice(self.N, k, replace=False)
                poblacion[i, idx] = 1
            else:
                poblacion[i] = np.random.randint(0, 2, size=self.N, dtype=np.int8)

        start_time = time.time()
        no_improve = 0
        prev_phi = float('inf')
        global_best_phi = float('inf')
        global_best_key = None

        for gen in range(self.generations):
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

            gen_phi = min(-val for val in fitness_vals)
            if self.verbose:
                self.logger.info(f"Gen {gen+1}/{self.generations}: gen_phi={gen_phi:.6f}")
            if abs(prev_phi - gen_phi) < 1e-5 or gen_phi >= prev_phi and no_improve >= self.patience:
                print(f"Early stop en gen {gen+1}")
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

        print("ya termine")
        return Solution(
            estrategia=GA_LABEL,
            perdida=best_phi,
            distribucion_subsistema=self.dists_ref,
            distribucion_particion=dist,
            tiempo_total=time.time() - start_time,
            particion=particion_str,
        )
