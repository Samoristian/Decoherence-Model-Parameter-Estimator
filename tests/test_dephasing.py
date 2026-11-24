import os
import sys
import math
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import torch
from PySide6.QtWidgets import QApplication

# Ensure repository root is importable regardless of pytest working directory
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import (
    QuantumStateGenerator,
    DephasingPhysicsEngine,
    DephasingMainWindow,
    REAL_DTYPE,
    COMPLEX_DTYPE,
    DEVICE,
)


@pytest.fixture(scope="module")
def state_gen() -> QuantumStateGenerator:
    return QuantumStateGenerator()


@pytest.fixture(scope="module")
def physics_engine() -> DephasingPhysicsEngine:
    return DephasingPhysicsEngine(quad_order=128)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestQuantumStateGenerator:
    """Tests for Eqs. (B.3)-(B.13) parameter and initial density matrix generation."""

    @pytest.mark.parametrize("seed", [0, 42, 123, 999])
    def test_generate_a_parameters_constraints(self, state_gen: QuantumStateGenerator, seed: int):
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(seed)
        a_r1_sq, A = state_gen.generate_a_parameters(gen)

        assert a_r1_sq.shape == (4,)
        assert a_r1_sq.dtype == REAL_DTYPE
        assert A.shape == (4, 4)
        assert A.dtype == COMPLEX_DTYPE

        # Capped simplex constraints: 0 <= |a_{r1}|^2 <= 0.5 and sum_r |a_{r1}|^2 = 1
        assert torch.all(a_r1_sq >= -1e-15)
        assert torch.all(a_r1_sq <= 0.5 + 1e-15)
        assert torch.isclose(a_r1_sq.sum(), torch.tensor(1.0, dtype=REAL_DTYPE), atol=1e-12)

        # Eq. (B.10): |a_{r2}| = |a_{r3}| = 0.5 for every r
        assert torch.allclose(A[:, 1].abs(), torch.full((4,), 0.5, dtype=REAL_DTYPE), atol=1e-12)
        assert torch.allclose(A[:, 2].abs(), torch.full((4,), 0.5, dtype=REAL_DTYPE), atol=1e-12)

        # Eqs. (B.3) & (B.6): Unitarity of A (A^dagger A = I and A A^dagger = I)
        eye4 = torch.eye(4, dtype=COMPLEX_DTYPE, device=DEVICE)
        assert torch.allclose(A.conj().T @ A, eye4, atol=1e-12)
        assert torch.allclose(A @ A.conj().T, eye4, atol=1e-12)

    @pytest.mark.parametrize("seed", [7, 42, 2026])
    def test_generate_b_matrix_identity(self, state_gen: QuantumStateGenerator, seed: int):
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(seed)
        b_mat, G_mat, opt_loss = state_gen.generate_b_matrix(gen, mode="identity")

        assert opt_loss < 1e-12
        # Row normalization (Eq. B.5): sum_k |b_{rk}|^2 = 1
        row_norms = (b_mat.real.square() + b_mat.imag.square()).sum(dim=1)
        assert torch.allclose(row_norms, torch.ones(4, dtype=REAL_DTYPE), atol=1e-10)

        # Identity Gram operator: G = B^T B^* = I_4
        eye4 = torch.eye(4, dtype=COMPLEX_DTYPE, device=DEVICE)
        assert torch.allclose(G_mat, eye4, atol=1e-10)

    @pytest.mark.parametrize("seed", [7, 42, 2026])
    def test_generate_b_matrix_non_identity_diagonal(self, state_gen: QuantumStateGenerator, seed: int):
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(seed)
        b_mat, G_mat, opt_loss = state_gen.generate_b_matrix(gen, mode="diagonal", spread=0.75)

        assert opt_loss < 1e-12
        # Row normalization (Eq. B.5): sum_k |b_{rk}|^2 = 1
        row_norms = (b_mat.real.square() + b_mat.imag.square()).sum(dim=1)
        assert torch.allclose(row_norms, torch.ones(4, dtype=REAL_DTYPE), atol=1e-10)

        # Off-diagonal elements vanish (Eq. B.4)
        off_diag = G_mat - torch.diag(torch.diag(G_mat))
        assert off_diag.abs().max().item() < 1e-10

        # Trace(G) = 4 and G != I_4
        diag_real = torch.diag(G_mat.real)
        assert torch.isclose(diag_real.sum(), torch.tensor(4.0, dtype=REAL_DTYPE), atol=1e-10)
        assert (diag_real - 1.0).abs().max().item() > 0.05

    @pytest.mark.parametrize("mode", ["diagonal", "identity"])
    @pytest.mark.parametrize("beta,omega_0,K", [(0.2, 1.0, 0.3), (2.5, 1.5, 1.2), (8.0, 2.0, 5.0)])
    def test_initial_rho_0_physicality(
        self,
        state_gen: QuantumStateGenerator,
        mode: str,
        beta: float,
        omega_0: float,
        K: float,
    ):
        a_r1_sq, _, b_mat, G_mat, _ = state_gen.generate_all(seed=42, mode=mode, spread=0.7)
        bw = torch.tensor(beta * omega_0, dtype=REAL_DTYPE, device=DEVICE)
        rho_0 = state_gen.calculate_rho_0(b_mat, a_r1_sq, bw, beta, K, G=G_mat)

        # Hermiticity
        assert torch.allclose(rho_0, rho_0.conj().T, atol=1e-13)
        # Unit trace
        assert torch.isclose(torch.trace(rho_0).real, torch.tensor(1.0, dtype=REAL_DTYPE), atol=1e-12)
        assert abs(torch.trace(rho_0).imag.item()) < 1e-13
        # Positive semidefiniteness
        evals = torch.linalg.eigvalsh(rho_0)
        assert evals.min().item() >= -1e-12


class TestDephasingPhysicsEngine:
    """Tests for bath functions, analytical Appendix A formulas, and open-system invariants."""

    def test_bath_functions_initial_and_ohmic_limit(self, physics_engine: DephasingPhysicsEngine):
        t = torch.tensor([0.0, 0.5, 2.0, 200.0], dtype=REAL_DTYPE, device=DEVICE)
        lam = 0.4
        omega_c = 5.0

        # Smooth transition across s = 1.0
        bath_s1 = physics_engine.compute_bath_functions(t, beta=1.0, lam=lam, s=1.0, omega_c=omega_c)
        bath_s1_eps = physics_engine.compute_bath_functions(t, beta=1.0, lam=lam, s=1.0 + 1e-7, omega_c=omega_c)

        assert bath_s1["gamma_t"][0].item() == pytest.approx(0.0, abs=1e-15)
        assert bath_s1["Phi_t"][0].item() == pytest.approx(0.0, abs=1e-15)
        assert bath_s1["Delta_t"][0].item() == pytest.approx(0.0, abs=1e-15)
        assert torch.all(bath_s1["gamma_t"] >= 0.0)
        assert torch.allclose(bath_s1["Phi_t"], bath_s1_eps["Phi_t"], atol=1e-6)

        # For s = 1 (Ohmic), lim_{t -> inf} Phi(t) = (pi / 2) * lambda
        expected_phi_inf = 0.5 * math.pi * lam
        assert bath_s1["Phi_t"][-1].item() == pytest.approx(expected_phi_inf, rel=1e-3)

    @pytest.mark.parametrize("s", [0.6, 1.0, 1.5])
    def test_pure_plus_appendix_a_exact_scaling(
        self,
        state_gen: QuantumStateGenerator,
        physics_engine: DephasingPhysicsEngine,
        s: float,
    ):
        """
        Verifies Appendix A exact analytical results (Eqs. A.2, A.10, A.11) for |+,+><+,+|:
          N(t) = (1/8) * sqrt((1 - e^{-4 gamma})^2 + 16 e^{-2 gamma} sin^2 Delta) - (1/8)*(1 - e^{-4 gamma})
          C(t) = 2 * N(t)
        """
        a_r1_sq, _, b_mat, G_mat, _ = state_gen.generate_all(seed=42, mode="diagonal")
        t = torch.linspace(0.0, 8.0, 100, dtype=REAL_DTYPE, device=DEVICE)

        res = physics_engine.simulate(
            t=t,
            beta=1.2,
            omega_0=1.0,
            lam=0.35,
            s=s,
            omega_c=2.5,
            state_mode="pure_plus",
            b_mat=b_mat,
            G_mat=G_mat,
            a_r1_sq=a_r1_sq,
        )

        gamma = res["gamma_t"]
        delta = res["Delta_t"]
        one_minus_e4g = 1.0 - torch.exp(-4.0 * gamma)
        radical = torch.sqrt(one_minus_e4g.square() + 16.0 * torch.exp(-2.0 * gamma) * torch.sin(delta).square())
        neg_exact = torch.clamp((radical - one_minus_e4g) / 8.0, min=0.0)
        conc_exact = 2.0 * neg_exact

        assert torch.allclose(res["negativity"], neg_exact, atol=1e-11)
        assert torch.allclose(res["concurrence"], conc_exact, atol=1e-11)
        assert torch.allclose(res["concurrence"], 2.0 * res["negativity"], atol=1e-11)

    @pytest.mark.parametrize("mode", ["diagonal", "identity", "pure_plus"])
    @pytest.mark.parametrize("s", [0.5, 1.0, 2.5])
    def test_density_matrix_invariants_over_time(
        self,
        state_gen: QuantumStateGenerator,
        physics_engine: DephasingPhysicsEngine,
        mode: str,
        s: float,
    ):
        a_r1_sq, _, b_mat, G_mat, _ = state_gen.generate_all(
            seed=42,
            mode=mode if mode != "pure_plus" else "diagonal",
        )
        t = torch.linspace(0.0, 10.0, 80, dtype=REAL_DTYPE, device=DEVICE)

        res = physics_engine.simulate(
            t=t,
            beta=1.0,
            omega_0=1.0,
            lam=0.3,
            s=s,
            omega_c=2.0,
            state_mode=mode,
            b_mat=b_mat,
            G_mat=G_mat,
            a_r1_sq=a_r1_sq,
        )

        # Purity bounds: 0.25 <= P(t) <= 1.0
        assert torch.all(res["purity"] >= 0.25 - 1e-12)
        assert torch.all(res["purity"] <= 1.0 + 1e-12)

        # Entanglement & entropy bounds
        assert torch.all(res["concurrence"] >= -1e-12)
        assert torch.all(res["concurrence"] <= 1.0 + 1e-12)
        assert torch.all(res["negativity"] >= -1e-12)
        assert torch.all(res["vn_entropy"] >= -1e-12)

        if mode != "pure_plus":
            # Eq. (5.12): gamma_nm^cor(t) <= 0 at all times and vanishes at t = 0
            assert res["gamma_cor_12"][0].item() == pytest.approx(0.0, abs=1e-15)
            assert torch.all(res["gamma_cor_12"] <= 1e-15)


class TestDephasingGUI:
    """Headless PySide6 smoke tests for interactive controls and live updates."""

    def test_window_controls_and_modes(self, qapp):
        window = DephasingMainWindow()
        try:
            assert window.b_mat is not None
            assert window.G_mat is not None
            assert window.a_r1_sq is not None

            # Switch through all preparation modes
            for idx in range(window.combo_mode.count()):
                window.combo_mode.setCurrentIndex(idx)
                assert len(window.lbl_metrics.text()) > 0

            # Adjust thermostat sliders
            window.sl_beta.spinbox.setValue(2.5)
            window.sl_s.spinbox.setValue(1.5)
            window.sl_lambda.spinbox.setValue(0.45)
            assert "T = 1/β = 0.4000" in window.lbl_temp.text()
        finally:
            window.close()
