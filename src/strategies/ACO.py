import time
import numpy as np

from src.funcs.base import emd_efecto
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import ACO_LABEL

class ACO(SIA):
    """Ant Colony Optimization (ACO) for network bipartitioning, with enhanced heuristics and local search."""

    def __init__(self, gestor, num_hormigas=50, alpha=1.0, beta=2.0,
                 rho=0.05, iteraciones=200, Q=1.0, epsilon=1e-9, verbose=False):
        super().__init__(gestor)
        self.num_hormigas = num_hormigas
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.iteraciones = iteraciones
        self.Q = Q
        self.epsilon = epsilon
        self.verbose = verbose

        # Se inicializan tras preparar subsistema:
        self.N = None               # total de nodos (m + n)
        self.m = None               # dimensión futuro
        self.tau = None             # feromonas shape (N, 2)
        self.eta = None             # heurística shape (N, 2)
        self.dists_ref = None       # distribución de referencia
        self.indices_futuro = None  # m indices
        self.indices_presente = None# n indices

    def inicializar_parametros(self):
        # Tau y eta iniciales uniformes
        self.tau = np.ones((self.N, 2), dtype=float)
        self.eta = np.ones((self.N, 2), dtype=float)
        # Heurística informada: EMD individual para asignar el nodo i al grupo 1
        for i in range(self.N):
            if i < self.m:
                subalcance = np.array([self.indices_futuro[i]], dtype=np.int8)
                submecanismo = np.array([], dtype=np.int8)
            else:
                subalcance = np.array([], dtype=np.int8)
                submecanismo = np.array([self.indices_presente[i - self.m]], dtype=np.int8)
            part = self.sia_subsistema.bipartir(subalcance, submecanismo)
            dist = part.distribucion_marginal()
            emd = emd_efecto(dist, self.dists_ref)
            self.eta[i, 1] = 1.0 / (emd + self.epsilon)

    def construir_particion(self):
        """Construye una partición (bits) asignando cada nodo a grupo 0 o 1."""
        bits = np.zeros(self.N, dtype=np.int8)
        for i in range(self.N):
            w0 = self.tau[i, 0] ** self.alpha * self.eta[i, 0] ** self.beta
            w1 = self.tau[i, 1] ** self.alpha * self.eta[i, 1] ** self.beta
            total = w0 + w1
            p0 = w0 / total if total > 0 else 0.5
            bits[i] = np.random.choice([0, 1], p=[p0, 1 - p0])
        # Evitar partición trivial
        if bits.sum() == 0 or bits.sum() == self.N:
            flip = np.random.randint(0, self.N)
            bits[flip] = 1 - bits[flip]
        return bits

    def actualizar_feromonas(self, soluciones, costos, best_bits_global, best_phi_global):
        """Evaporación y refuerzo de la feromona."""
        # Evaporar
        self.tau *= (1 - self.rho)
        # Depositar por la mejor hormiga local
        idx_mejor = int(np.argmin(costos))
        mejor_bits = soluciones[idx_mejor]
        mejor_costo = costos[idx_mejor]
        delta = self.Q / (mejor_costo + self.epsilon)
        for i, g in enumerate(mejor_bits):
            self.tau[i, g] += delta
        # Depósito elitista global
        delta_g = self.Q / (best_phi_global + self.epsilon)
        for i, g in enumerate(best_bits_global):
            self.tau[i, g] += delta_g

    def compute_phi_dist(self, bits):
        """Calcula φ y la distribución marginal para un vector de bits."""
        subalcance = np.where(bits[:self.m] == 1)[0]
        submecanismo = np.where(bits[self.m:] == 1)[0]
        part = self.sia_subsistema.bipartir(
            np.array(subalcance, dtype=np.int8),
            np.array(submecanismo, dtype=np.int8)
        )
        dist = part.distribucion_marginal()
        phi = emd_efecto(dist, self.dists_ref)
        return phi, dist

    def local_search(self, bits):
        """Mejora localmente la partición invirtiendo bits para reducir φ."""
        phi, dist = self.compute_phi_dist(bits)
        improved = True
        while improved:
            improved = False
            for i in range(self.N):
                bits[i] ^= 1  # flip
                new_phi, new_dist = self.compute_phi_dist(bits)
                if new_phi < phi:
                    phi, dist = new_phi, new_dist
                    improved = True
                else:
                    bits[i] ^= 1  # revert
        return bits, phi, dist

    def aplicar_estrategia(self, condiciones, alcance, mecanismo) -> Solution:
        """Aplica ACO con heurística avanzada y búsqueda local."""
        # 1) Preparar el subsistema
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)
        futuros = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos
        self.m = futuros.size
        n = presentes.size
        self.N = self.m + n
        self.indices_futuro = futuros
        self.indices_presente = presentes
        self.dists_ref = self.sia_dists_marginales

        # 2) Inicializar ACO
        self.inicializar_parametros()

        best_phi = float('inf')
        best_bits = None
        best_dist = None
        start_time = time.time()

        # 3) Bucle principal
        for t in range(self.iteraciones):
            soluciones, costos = [], []
            for _ in range(self.num_hormigas):
                bits = self.construir_particion()
                bits, phi, dist = self.local_search(bits)
                soluciones.append(bits.copy())
                costos.append(phi)
                if phi < best_phi:
                    best_phi, best_bits, best_dist = phi, bits.copy(), dist
            # Actualizar feromonas con elitismo
            self.actualizar_feromonas(soluciones, costos, best_bits, best_phi)
            if self.verbose:
                print(f"Iter {t+1}/{self.iteraciones}: best_phi = {best_phi:.6f}")

        # 4) Formatear partición óptima
        grupos = []
        for i in range(self.N):
            if best_bits[i] == 1:
                if i < self.m:
                    grupos.append((1, int(futuros[i])))
                else:
                    grupos.append((0, int(presentes[i - self.m])))
        particion_str = " ".join(f"({t},{idx})" for t, idx in grupos)

        # 5) Devolver Solution
        return Solution(
            estrategia=ACO_LABEL,
            perdida=best_phi,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=best_dist,
            tiempo_total=time.time() - start_time,
            particion=particion_str,
        )
