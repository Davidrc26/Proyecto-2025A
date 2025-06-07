import time
import numpy as np
from typing import Tuple

from src.controllers.manager import Manager
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile
from src.funcs.base import emd_efecto
from src.funcs.format import fmt_biparte_q
from src.constants.base import TYPE_TAG


class OneFuturePartition(SIA):
    """
    Estrategia para evaluar solo particiones de la forma:
        - Un único nodo del futuro está aislado.
        - Todos los presentes y demás futuros están agrupados.
    """
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        profiler_manager.start_session(f"1FUTUROVSTODOS_{len(gestor.estado_inicial)}")
        self.logger = SafeLogger("1FUTUROVSALL")

    @profile(context={TYPE_TAG: "ONE_FUTURE_PARTITION"})
    def aplicar_estrategia(self, condiciones: str, alcance: str, mecanismo: str) -> Solution:
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)

        futuros = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos
        dists_ref = self.sia_dists_marginales / self.sia_dists_marginales.sum()

        mejor_phi = float("inf")
        mejor_particion = None
        mejor_dist = None

        for idx_futuro in futuros:
            # Un solo nodo futuro aislado
            alcance_bin = np.array([i for i in futuros if i != idx_futuro], dtype=np.int8)
            mecanismo_bin = np.array(presentes, dtype=np.int8)

            part = self.sia_subsistema.bipartir(alcance_bin, mecanismo_bin)
            dist = part.distribucion_marginal()
            dist = dist / dist.sum() if dist.sum() > 0 else dist
            phi = emd_efecto(dist, dists_ref)

            seleccion = [(1, int(idx_futuro))]  # futuro aislado
            complemento = [(0, int(i)) for i in presentes] + [(1, int(i)) for i in futuros if i != idx_futuro]
            particion_str = fmt_biparte_q(seleccion, complemento)

            if phi < mejor_phi:
                mejor_phi = phi
                mejor_particion = particion_str
                mejor_dist = dist

        return Solution(
            estrategia="ONE_FUTURE_PARTITION",
            perdida=mejor_phi,
            distribucion_subsistema=dists_ref,
            distribucion_particion=mejor_dist,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=mejor_particion,
        )
