import sys
import torch
from torch import pi

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
                   QLabel, QSlider, QDoubleSpinBox, QSpinBox, QPushButton, QComboBox, QTabWidget, QTextEdit, QSplitter)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

REAL_DTYPE = torch.float64
COMPLEX_DTYPE = torch.complex128
DEVICE = torch.device("cpu")
MONOSPACE_STYLE = "font-family: 'Courier New', monospace; font-size: 12px;"
SUBLABEL_STYLE = "color: #334155; font-style: italic; margin-left: 4px;"


class QuantumStateGenerator:
    """
    Stateful PyTorch generator for non-selective measurement parameters:
      - Subset probabilities |a_{r1}|^2 and unitary matrix A = (a_{rk}) (Eqs. B.3, B.6, B.10)
      - Measurement vectors b_{rk} for Diagonal (G != I_4) or Identity (G = I_4) Gram operator
      - Initial reduced density matrix rho_S(0) (Eqs. B.11, B.13)
    """

    def __init__(self):
        self.ones4 = torch.ones(4, dtype=REAL_DTYPE, device=DEVICE)
        self.eye4_complex = torch.eye(4, dtype=COMPLEX_DTYPE, device=DEVICE)

        k_idx = torch.arange(4, dtype=REAL_DTYPE, device=DEVICE)
        angles = 2.0 * pi * torch.outer(k_idx, k_idx) / 4.0
        self.dft4 = 0.5 * torch.complex(torch.cos(angles), torch.sin(angles))

    def generate_a_parameters(self, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples |a_{r1}|^2 on the capped simplex {0 <= p_r <= 0.5, sum_r p_r = 1}
        and constructs a matching 4x4 unitary matrix A = (a_{rk}) with |a_{r2}| = |a_{r3}| = 0.5.
        """
        u = torch.rand(2, generator=gen, dtype=REAL_DTYPE, device=DEVICE) * 0.5
        p_base = torch.stack([u[0], 0.5 - u[0], u[1], 0.5 - u[1]])
        perm = torch.randperm(4, generator=gen, device=DEVICE)
        a_r1_sq = p_base[perm]

        s0, s1, s2, s3 = torch.sqrt(p_base)
        A_base = torch.tensor(
            [
                [s0, 0.5, 0.5, s1],
                [s1, 0.5, -0.5, -s0],
                [s2, 0.5, -0.5, s3],
                [s3, 0.5, 0.5, -s2],
            ],
            dtype=COMPLEX_DTYPE,
            device=DEVICE,
        )

        phase_angles = 2.0 * pi * torch.rand((2, 4), generator=gen, dtype=REAL_DTYPE, device=DEVICE)
        row_phases = torch.complex(torch.cos(phase_angles[0]), torch.sin(phase_angles[0])).unsqueeze(1)
        col_phases = torch.complex(torch.cos(phase_angles[1]), torch.sin(phase_angles[1])).unsqueeze(0)

        A = (A_base[perm] * row_phases) * col_phases
        return a_r1_sq, A

    @staticmethod
    def _unpack_b(params: torch.Tensor) -> torch.Tensor:
        return torch.complex(params[:16].view(4, 4), params[16:].view(4, 4))

    def _compute_b_loss(self, params: torch.Tensor, target_G: torch.Tensor) -> torch.Tensor:
        b = self._unpack_b(params)
        row_norms = (b.real.square() + b.imag.square()).sum(dim=1)
        loss_norm = (row_norms - 1.0).square().sum()

        G = b.T @ b.conj()
        diff = G - target_G
        loss_gram = (diff.real.square() + diff.imag.square()).sum()
        return loss_norm + loss_gram

    def generate_b_matrix(
        self,
        gen: torch.Generator,
        mode: str = "diagonal",
        spread: float = 0.65,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Optimizes 32 real parameters via PyTorch L-BFGS to produce 4x4 complex b_{rk}:
          - 'identity': G = B^T B^* = I_4
          - 'diagonal': G = B^T B^* = diag(B_1, B_2, B_3, B_4) != I_4 with sum_k B_k = 4
        """
        if mode == "identity":
            target_diag = self.ones4
            target_G = self.eye4_complex
        elif mode == "diagonal":
            raw = torch.rand(4, generator=gen, dtype=REAL_DTYPE, device=DEVICE) + 0.15
            raw = raw * (4.0 / raw.sum())
            target_diag = (1.0 - spread) * self.ones4 + spread * raw
            target_G = torch.diag(target_diag).to(COMPLEX_DTYPE)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        rand_angles = 2.0 * pi * torch.rand(4, generator=gen, dtype=REAL_DTYPE, device=DEVICE)
        rand_phases = torch.complex(torch.cos(rand_angles), torch.sin(rand_angles))
        b_init = self.dft4 * (rand_phases * torch.sqrt(target_diag)).unsqueeze(0)

        noise = 0.05 * torch.randn(32, generator=gen, dtype=REAL_DTYPE, device=DEVICE)
        params = (torch.cat([b_init.real.reshape(-1), b_init.imag.reshape(-1)]) + noise).detach().requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [params],
            lr=1.0,
            max_iter=500,
            tolerance_grad=1e-14,
            tolerance_change=1e-16,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = self._compute_b_loss(params, target_G)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            b_opt = self._unpack_b(params)
            G_opt = b_opt.T @ b_opt.conj()
            final_loss = float(self._compute_b_loss(params, target_G).item())

        return b_opt, G_opt, final_loss

    def generate_all(
        self,
        seed: int,
        mode: str = "diagonal",
        spread: float = 0.65,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Generates all initial-state tensors using a single seeded PyTorch RNG."""
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(seed)
        a_r1_sq, A_mat = self.generate_a_parameters(gen)
        b_mat, G_mat, opt_loss = self.generate_b_matrix(gen, mode=mode, spread=spread)
        return a_r1_sq, A_mat, b_mat, G_mat, opt_loss

    @staticmethod
    def calculate_rho_0(
        b: torch.Tensor,
        a_r1_sq: torch.Tensor,
        bw: torch.Tensor,
        beta: float,
        K: float,
        G: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes initial reduced density matrix [rho_S(0)]_{nm} via Eqs. (B.11) and (B.13)."""
        bK = torch.tensor(beta * K, dtype=REAL_DTYPE, device=DEVICE)
        tanh_bw = torch.tanh(bw)
        exp_neg_bK_sech = torch.exp(-bK) / torch.cosh(bw)
        denom = 1.0 + exp_neg_bK_sech

        pref_off = -tanh_bw / denom
        pref_diag_G = (1.0 + tanh_bw + exp_neg_bK_sech) / denom

        weighted_b = a_r1_sq.to(COMPLEX_DTYPE).unsqueeze(1) * b
        corr_matrix = weighted_b.T @ b.conj()

        gram = (b.T @ b.conj()) if G is None else G
        G_diag = torch.diag(gram.real).to(COMPLEX_DTYPE)

        rho_0 = pref_off * corr_matrix + 0.25 * pref_diag_G * G_diag
        return 0.5 * (rho_0 + rho_0.conj().T)


class DephasingPhysicsEngine:
    """
    Pure-PyTorch vectorized solver for the two-spin bosonic dephasing model:
      - Decoherence function gamma(t), dynamic phase Phi(t), indirect coupling Delta(t)
      - Correlational decoherence gamma_nm^cor(t) & phase chi_nm(t) (Eqs. 5.12-5.15)
      - Wootters concurrence C(t), Peres-Horodecki negativity N(t), purity P(t),
        von Neumann entropy S_vN(t), and characteristic relaxation times.
    """

    def __init__(self, quad_order: int = 128):
        x, w = self._gauss_legendre_nodes(quad_order)
        self.u = (1.0 + x) / (1.0 - x)
        self.log_jac_w = torch.log((2.0 / (1.0 - x).square()) * w)

        m_vals = torch.tensor([1.0, 0.0, 0.0, -1.0], dtype=REAL_DTYPE, device=DEVICE)
        self.dm = m_vals.unsqueeze(1) - m_vals.unsqueeze(0)
        self.dm2 = m_vals.square().unsqueeze(1) - m_vals.square().unsqueeze(0)
        self.dm_sq = self.dm.square()

        self.eye4_bool = torch.eye(4, dtype=torch.bool, device=DEVICE).unsqueeze(0)
        self.pure_plus_rho0 = torch.full((4, 4), 0.25, dtype=COMPLEX_DTYPE, device=DEVICE)
        self.sigma_yy = torch.tensor(
            [
                [0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
            ],
            dtype=COMPLEX_DTYPE,
            device=DEVICE,
        )

    @staticmethod
    def _gauss_legendre_nodes(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes Gauss-Legendre nodes and weights on [-1, 1] in pure PyTorch float64."""
        idx = torch.arange(1, n + 1, dtype=REAL_DTYPE, device=DEVICE)
        x = torch.cos(pi * (idx - 0.25) / (n + 0.5))

        for _ in range(12):
            p0 = torch.ones_like(x)
            p1 = x.clone()
            for k in range(2, n + 1):
                p2 = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k
                p0, p1 = p1, p2
            dp = n * (x * p1 - p0) / (x.square() - 1.0)
            x = x - p1 / dp

        # Final derivative evaluation at converged nodes for full float64 weight accuracy
        p0 = torch.ones_like(x)
        p1 = x.clone()
        for k in range(2, n + 1):
            p2 = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k
            p0, p1 = p1, p2
        dp = n * (x * p1 - p0) / (x.square() - 1.0)
        w = 2.0 / ((1.0 - x.square()) * dp.square())
        return x, w

    def compute_bath_functions(
        self,
        t: torch.Tensor,
        beta: float,
        lam: float,
        s: float,
        omega_c: float,
    ) -> dict[str, torch.Tensor | float]:
        """Computes K, gamma(t), Phi(t), and Delta(t) across time grid t."""
        K = lam * torch.exp(torch.lgamma(s))

        tau = omega_c * t
        theta_c = torch.atan(tau)
        amp = (1.0 + tau.square()).pow(0.5 * (1.0 - s))
        sinc_term = theta_c if abs(s - 1.0) < 1e-8 else torch.sin((s - 1.0) * theta_c) / (s - 1.0)
        Phi_t = K * amp * sinc_term
        Delta_t = Phi_t - K * t

        base_weight = lam * torch.exp((s - 2.0) * torch.log(self.u) - self.u + self.log_jac_w)
        y = beta * omega_c * self.u
        coth_factor = torch.where(
            y < 1e-6,
            2.0 / y + y / 6.0,
            1.0 + 2.0 / torch.expm1(torch.clamp(y, max=700.0)),
        )

        phase_ut = tau.unsqueeze(1) * self.u.unsqueeze(0)
        one_minus_cos = torch.where(
            phase_ut < 1e-4,
            0.5 * phase_ut.square() - phase_ut.pow(4) / 24.0,
            1.0 - torch.cos(phase_ut),
        )

        gamma_t = one_minus_cos @ (base_weight * coth_factor)
        return {
            "K": K,
            "gamma_t": gamma_t,
            "Phi_t": Phi_t,
            "Delta_t": Delta_t,
        }

    def simulate(
        self,
        t: torch.Tensor,
        beta: float,
        omega_0: float,
        lam: float,
        s: float,
        omega_c: float,
        state_mode: str,
        b_mat: torch.Tensor,
        G_mat: torch.Tensor,
        a_r1_sq: torch.Tensor,
    ) -> dict[str, torch.Tensor | float | None]:
        """Runs full vectorized simulation over time grid t."""
        bath = self.compute_bath_functions(t, beta, lam, s, omega_c)
        K = bath["K"]
        gamma_t = bath["gamma_t"]
        Phi_t = bath["Phi_t"]
        Delta_t = bath["Delta_t"]
        bw = torch.tensor(beta * omega_0, dtype=REAL_DTYPE, device=DEVICE)

        if state_mode == "pure_plus":
            rho_0 = self.pure_plus_rho0
            use_initial_corr = False
        else:
            rho_0 = QuantumStateGenerator.calculate_rho_0(b_mat, a_r1_sq, bw, beta, K, G=G_mat)
            use_initial_corr = True

        total_phase = -omega_0 * t[:, None, None] * self.dm[None, :, :] - Delta_t[:, None, None] * self.dm2[None, :, :]
        damp_bath = torch.exp(-gamma_t[:, None, None] * self.dm_sq[None, :, :])
        rho_tot = rho_0.unsqueeze(0) * damp_bath * torch.complex(torch.cos(total_phase), torch.sin(total_phase))

        if use_initial_corr:
            arg = 2.0 * Phi_t[:, None, None] * self.dm[None, :, :]
            sin_arg = torch.sin(arg)
            cos_arg = torch.cos(arg)
            sinh_bw = torch.sinh(bw)
            coth_bw = 1.0 / torch.tanh(bw)

            gamma_cor_12 = -0.5 * torch.log1p((sin_arg[:, 0, 1] / sinh_bw).square())
            chi_12 = -torch.atan2(coth_bw * sin_arg[:, 0, 1], cos_arg[:, 0, 1])
            gamma_total_12 = gamma_t + gamma_cor_12

            F_corr = torch.complex(cos_arg, -coth_bw * sin_arg)
            F_corr = torch.where(self.eye4_bool, torch.ones_like(F_corr), F_corr)
            rho_tot = rho_tot * F_corr
        else:
            gamma_cor_12 = torch.zeros_like(gamma_t)
            chi_12 = torch.zeros_like(gamma_t)
            gamma_total_12 = gamma_t

        rho_tot = 0.5 * (rho_tot + rho_tot.conj().transpose(-1, -2))

        # Purity P(t) and von Neumann entropy S_vN(t)
        purity = torch.real(torch.einsum("tij,tji->t", rho_tot, rho_tot))
        evals_rho = torch.clamp(torch.linalg.eigvalsh(rho_tot), min=1e-16)
        vn_entropy = -(evals_rho * torch.log(evals_rho)).sum(dim=-1)

        # Peres-Horodecki Negativity N(t)
        T_len = t.shape[0]
        rho_tb = rho_tot.view(T_len, 2, 2, 2, 2).permute(0, 1, 4, 3, 2).reshape(T_len, 4, 4)
        negativity = torch.clamp(-torch.linalg.eigvalsh(rho_tb), min=0.0).sum(dim=-1)

        # Wootters Concurrence C(t)
        rho_tilde = self.sigma_yy @ rho_tot.conj() @ self.sigma_yy
        eigvals_R = torch.linalg.eigvals(rho_tot @ rho_tilde)
        sqrt_lams = torch.sqrt(torch.clamp(eigvals_R.real, min=0.0))
        sqrt_lams_sorted, _ = torch.sort(sqrt_lams, dim=-1, descending=True)
        concurrence = torch.clamp(
            sqrt_lams_sorted[:, 0] - sqrt_lams_sorted[:, 1] - sqrt_lams_sorted[:, 2] - sqrt_lams_sorted[:, 3],
            min=0.0,
        )

        return {
            "t": t,
            "K": K,
            "rho_0": rho_0,
            "gamma_t": gamma_t,
            "gamma_cor_12": gamma_cor_12,
            "gamma_total_12": gamma_total_12,
            "Phi_t": Phi_t,
            "Delta_t": Delta_t,
            "chi_12": chi_12,
            "purity": purity,
            "vn_entropy": vn_entropy,
            "negativity": negativity,
            "concurrence": concurrence,
            "tau_rel_1": self._first_crossing_time(t, gamma_t, threshold=1.0),
            "tau_rel_4": self._first_crossing_time(t, gamma_t, threshold=0.25),
            "tau_ohmic_th": beta / (pi * max(lam, 1e-12)),
            "tau_bath_c": 1.0 / max(omega_c, 1e-12),
        }

    @staticmethod
    def _first_crossing_time(t: torch.Tensor, y: torch.Tensor, threshold: float) -> float | None:
        above = torch.nonzero(y >= threshold, as_tuple=False)
        if above.numel() == 0:
            return None
        idx = int(above[0].item())
        if idx == 0:
            return float(t[0].item())
        t0, t1 = float(t[idx - 1].item()), float(t[idx].item())
        y0, y1 = float(y[idx - 1].item()), float(y[idx].item())
        if abs(y1 - y0) < 1e-15:
            return t1
        return t0 + (threshold - y0) * (t1 - t0) / (y1 - y0)


class ParameterSliderRow(QWidget):
    """Synchronized QSlider + QDoubleSpinBox row for thermostat and system parameters."""

    def __init__(
        self,
        label_text: str,
        min_val: float,
        max_val: float,
        init_val: float,
        step: float,
        decimals: int = 3,
        callback=None,
    ):
        super().__init__()
        self.scale = round(1.0 / step)
        self.callback = callback

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        label = QLabel(label_text)
        label.setMinimumWidth(145)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(round(min_val * self.scale)), int(round(max_val * self.scale)))
        self.slider.setValue(int(round(init_val * self.scale)))

        self.spinbox = QDoubleSpinBox()
        self.spinbox.setDecimals(decimals)
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setSingleStep(step)
        self.spinbox.setValue(init_val)
        self.spinbox.setMinimumWidth(85)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)

        layout.addWidget(label)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.spinbox)

    def _on_slider_changed(self, int_val: int):
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(int_val / self.scale)
        self.spinbox.blockSignals(False)
        if self.callback:
            self.callback()

    def _on_spinbox_changed(self, val: float):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(val * self.scale)))
        self.slider.blockSignals(False)
        if self.callback:
            self.callback()

    def value(self) -> float:
        return float(self.spinbox.value())


class DephasingMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Two-Spin Bosonic Dephasing Model Simulator")
        self.resize(1380, 860)

        self.state_gen = QuantumStateGenerator()
        self.engine = DephasingPhysicsEngine(quad_order=128)

        self.a_r1_sq: torch.Tensor | None = None
        self.A_mat: torch.Tensor | None = None
        self.b_mat: torch.Tensor | None = None
        self.G_mat: torch.Tensor | None = None
        self.opt_loss: float = 0.0

        self._cached_tmax: float | None = None
        self._cached_t: torch.Tensor | None = None

        self._build_ui()
        self._init_plots()
        self.regenerate_state_parameters()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left control panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMinimumWidth(410)
        left_panel.setMaximumWidth(480)

        # 1. Thermostat & Spin Parameters
        thermo_group = QGroupBox("Thermostat & Two-Spin Parameters")
        thermo_layout = QVBoxLayout(thermo_group)

        self.sl_beta = ParameterSliderRow("Inverse Temp β (1/k_B T):", 0.05, 10.0, 1.0, 0.05, 2, self.update_simulation)
        self.lbl_temp = QLabel()
        self.lbl_temp.setStyleSheet(SUBLABEL_STYLE)

        self.sl_omega0 = ParameterSliderRow("Qubit Frequency ω₀:", 0.05, 5.0, 1.0, 0.05, 2, self.update_simulation)
        self.sl_lambda = ParameterSliderRow("Coupling Strength λ:", 0.01, 2.0, 0.25, 0.01, 3, self.update_simulation)
        self.sl_s = ParameterSliderRow("Ohmicity Index s:", 0.10, 3.50, 1.00, 0.05, 2, self.update_simulation)
        self.lbl_regime = QLabel()
        self.lbl_regime.setStyleSheet(SUBLABEL_STYLE)

        self.sl_omegac = ParameterSliderRow("Cut-off Frequency ω_c:", 0.20, 15.0, 2.0, 0.10, 2, self.update_simulation)
        self.sl_tmax = ParameterSliderRow("Max Evolution Time t_max:", 1.0, 40.0, 12.0, 0.5, 1, self.update_simulation)

        for widget in (
            self.sl_beta,
            self.lbl_temp,
            self.sl_omega0,
            self.sl_lambda,
            self.sl_s,
            self.lbl_regime,
            self.sl_omegac,
            self.sl_tmax,
        ):
            thermo_layout.addWidget(widget)
        left_layout.addWidget(thermo_group)

        # 2. Initial State Preparation & Gram Operator
        prep_group = QGroupBox("Initial State & Gram Operator (PyTorch L-BFGS)")
        prep_layout = QGridLayout(prep_group)

        prep_layout.addWidget(QLabel("Initial Preparation:"), 0, 0)
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("Mixed Correlated — Diagonal Gram (G ≠ I)", "diagonal")
        self.combo_mode.addItem("Mixed Correlated — Identity Gram (G = I)", "identity")
        self.combo_mode.addItem("Pure Benchmark State |+,+⟩⟨+,+| (App. A)", "pure_plus")
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        prep_layout.addWidget(self.combo_mode, 0, 1, 1, 2)

        prep_layout.addWidget(QLabel("RNG Seed:"), 1, 0)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(42)
        prep_layout.addWidget(self.spin_seed, 1, 1)

        btn_regen = QPushButton("Regenerate b_rk && |a_r1|²")
        btn_regen.clicked.connect(self.regenerate_state_parameters)
        prep_layout.addWidget(btn_regen, 1, 2)

        self.sl_spread = ParameterSliderRow("Diag G Anisotropy:", 0.10, 1.00, 0.70, 0.05, 2, None)
        prep_layout.addWidget(self.sl_spread, 2, 0, 1, 3)

        left_layout.addWidget(prep_group)

        # 3. Computed Metrics
        metrics_group = QGroupBox("Computed Relaxation & Entanglement Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        self.lbl_metrics = QLabel()
        self.lbl_metrics.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_metrics.setStyleSheet(MONOSPACE_STYLE)
        metrics_layout.addWidget(self.lbl_metrics)
        left_layout.addWidget(metrics_group, stretch=1)

        splitter.addWidget(left_panel)

        # Right panel: Tabs
        right_tabs = QTabWidget()

        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        self.fig = Figure(figsize=(9, 7), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        plot_layout.addWidget(NavigationToolbar2QT(self.canvas, plot_tab))
        plot_layout.addWidget(self.canvas, stretch=1)
        right_tabs.addTab(plot_tab, "Quantum Dynamics Plots")

        inspector_tab = QWidget()
        inspector_layout = QVBoxLayout(inspector_tab)
        self.txt_tensors = QTextEdit()
        self.txt_tensors.setReadOnly(True)
        self.txt_tensors.setStyleSheet(MONOSPACE_STYLE)
        inspector_layout.addWidget(self.txt_tensors)
        right_tabs.addTab(inspector_tab, "Generated Tensors (|a_r1|², b_rk, G, ρ_S(0))")

        splitter.addWidget(right_tabs)
        splitter.setStretchFactor(1, 1)

    def _init_plots(self):
        """Creates persistent Matplotlib axes and line artists once to avoid redraw allocations."""
        self.axes = self.fig.subplots(2, 2)
        ax1, ax2 = self.axes[0, 0], self.axes[0, 1]
        ax3, ax4 = self.axes[1, 0], self.axes[1, 1]

        (self.ln_conc,) = ax1.plot([], [], label="Concurrence C(t)", color="#1d4ed8", lw=2.0)
        (self.ln_neg,) = ax1.plot([], [], label="Negativity N(t)", color="#dc2626", ls="--", lw=1.8)
        ax1.set_title("Entanglement: Concurrence C(t) & Negativity N(t)")
        ax1.set_xlabel("Time t")
        ax1.set_ylabel("Measure")

        (self.ln_gamma,) = ax2.plot([], [], label="Bath γ(t)", color="#0f766e", lw=1.8)
        (self.ln_gamma_cor,) = ax2.plot([], [], label="Correlational γ₁₂^cor(t) ≤ 0", color="#9333ea", ls="--", lw=1.6)
        (self.ln_gamma_tot,) = ax2.plot([], [], label="Total γ₁₂^tot(t) = γ + γ^cor", color="#ea580c", lw=2.0)
        self.zero_line = ax2.axhline(0.0, color="black", lw=0.8, alpha=0.5)
        self.tau_line = ax2.axvline(0.0, color="#475569", ls=":", lw=1.4, label="τ_rel (γ=1)")
        ax2.set_title("Generalized Decoherence & Recoherence Dynamics")
        ax2.set_xlabel("Time t")
        ax2.set_ylabel("Decoherence Function")

        (self.ln_purity,) = ax3.plot([], [], label="Purity P(t) = Tr[ρ²]", color="#15803d", lw=2.0)
        (self.ln_entropy,) = ax3.plot([], [], label="Von Neumann Entropy S_vN(t)", color="#b45309", ls="-.", lw=1.8)
        ax3.set_title("System Purity P(t) & Von Neumann Entropy")
        ax3.set_xlabel("Time t")
        ax3.set_ylabel("Value")

        (self.ln_phi,) = ax4.plot([], [], label="Dynamic Phase Φ(t)", color="#2563eb", lw=1.8)
        (self.ln_delta,) = ax4.plot([], [], label="Indirect Coupling Δ(t) = Φ(t) - Kt", color="#64748b", ls="--", lw=1.6)
        (self.ln_chi,) = ax4.plot([], [], label="Correlational Phase χ₁₂(t)", color="#db2777", lw=1.6)
        ax4.set_title("Bath-Induced & Correlational Phase Shifts")
        ax4.set_xlabel("Time t")
        ax4.set_ylabel("Phase (rad)")

        for ax in self.axes.flat:
            ax.grid(True, alpha=0.3)

        self.fig.tight_layout()

    def _get_time_grid(self, t_max: float) -> torch.Tensor:
        if self._cached_t is None or self._cached_tmax != t_max:
            self._cached_tmax = t_max
            self._cached_t = torch.linspace(0.0, t_max, 400, dtype=REAL_DTYPE, device=DEVICE)
        return self._cached_t

    def _on_mode_changed(self):
        mode = self.combo_mode.currentData()
        self.sl_spread.setEnabled(mode == "diagonal")
        if mode in ("diagonal", "identity"):
            self.regenerate_state_parameters()
        else:
            self.update_simulation()

    def regenerate_state_parameters(self):
        mode = self.combo_mode.currentData()
        gram_mode = mode if mode in ("diagonal", "identity") else "diagonal"
        self.a_r1_sq, self.A_mat, self.b_mat, self.G_mat, self.opt_loss = self.state_gen.generate_all(
            seed=int(self.spin_seed.value()),
            mode=gram_mode,
            spread=self.sl_spread.value(),
        )
        self.update_simulation()

    def update_simulation(self):
        beta = self.sl_beta.value()
        omega_0 = self.sl_omega0.value()
        lam = self.sl_lambda.value()
        s = self.sl_s.value()
        omega_c = self.sl_omegac.value()
        t_max = self.sl_tmax.value()
        mode = self.combo_mode.currentData()

        self.lbl_temp.setText(f"Effective Temperature T = 1/β = {1.0 / beta:.4f}")
        if abs(s - 1.0) < 1e-6:
            regime_str = "Ohmic (s = 1.00, white noise)"
        elif s < 1.0:
            regime_str = f"Sub-Ohmic (0 < s = {s:.2f} < 1)"
        elif s <= 2.0:
            regime_str = f"Super-Ohmic complete decoherence (1 < s = {s:.2f} ≤ 2)"
        else:
            regime_str = f"Super-Ohmic partial decoherence (s = {s:.2f} > 2)"
        self.lbl_regime.setText(f"Regime: {regime_str}")

        res = self.engine.simulate(
            t=self._get_time_grid(t_max),
            beta=beta,
            omega_0=omega_0,
            lam=lam,
            s=s,
            omega_c=omega_c,
            state_mode=mode,
            b_mat=self.b_mat,
            G_mat=self.G_mat,
            a_r1_sq=self.a_r1_sq,
        )

        self._update_metrics_and_inspector(res, mode)
        self._update_plots(res, mode)

    def _update_metrics_and_inspector(self, res: dict, mode: str):
        tau1_str = f"{res['tau_rel_1']:.4f}" if res["tau_rel_1"] is not None else "> t_max (or partial)"
        tau4_str = f"{res['tau_rel_4']:.4f}" if res["tau_rel_4"] is not None else "> t_max (or partial)"

        C, N, P, g_tot = res["concurrence"], res["negativity"], res["purity"], res["gamma_total_12"]
        G_diag_vals = torch.diag(self.G_mat.real).tolist()
        G_off_max = float((self.G_mat - torch.diag(torch.diag(self.G_mat))).abs().max().item())

        self.lbl_metrics.setText(
            f"Renormalization K = λ·Γ(s)  : {res['K']:.5f}\n"
            f"Bath Correlation τ_c = 1/ω_c : {res['tau_bath_c']:.4f}\n"
            f"Ohmic Thermal τ_th = β/(πλ)  : {res['tau_ohmic_th']:.4f}\n"
            f"1/e Coherence τ_rel (γ=1)    : {tau1_str}\n"
            f"1/e Coherence τ_rel^(4)(4γ=1): {tau4_str}\n"
            f"-----------------------------------------\n"
            f"Concurrence  C(0) / C_max    : {C[0].item():.4f} / {C.max().item():.4f}\n"
            f"Negativity   N(0) / N_max    : {N[0].item():.4f} / {N.max().item():.4f}\n"
            f"Purity       P(0) / P_max    : {P[0].item():.4f} / {P.max().item():.4f}\n"
            f"Min Total γ_12^tot(t)        : {g_tot.min().item():.5f}\n"
            f"-----------------------------------------\n"
            f"Gram diag(G)                 : [{', '.join(f'{v:.3f}' for v in G_diag_vals)}]\n"
            f"Max off-diag |G_kk'| (k≠k')  : {G_off_max:.2e}\n"
            f"L-BFGS Optimization Loss     : {self.opt_loss:.2e}"
        )

        a_r4_sq = 0.5 - self.a_r1_sq
        rho_0 = res["rho_0"]
        tr_rho0 = float(torch.trace(rho_0).real.item())
        evals_rho0 = torch.linalg.eigvalsh(rho_0).tolist()

        self.txt_tensors.setPlainText(
            "\n".join(
                [
                    "=== Generated Subset Vector Probabilities (Eqs. B.3, B.10) ===",
                    f"|a_r1|^2 = {[round(v, 5) for v in self.a_r1_sq.tolist()]}  (sum = {self.a_r1_sq.sum().item():.6f})",
                    f"|a_r4|^2 = {[round(v, 5) for v in a_r4_sq.tolist()]}  (|a_r1|^2 + |a_r4|^2 = 0.5)",
                    "|a_r2|^2 = |a_r3|^2 = [0.25, 0.25, 0.25, 0.25]\n",
                    f"=== Gram Operator G = B^T B^* (Mode: {mode}, L-BFGS loss = {self.opt_loss:.3e}) ===",
                    self._format_complex_matrix(self.G_mat),
                    "\n=== Optimized Measurement Subset Matrix b_rk ===",
                    self._format_complex_matrix(self.b_mat),
                    f"\n=== Initial Reduced Density Matrix ρ_S(0) (Tr = {tr_rho0:.6f}) ===",
                    self._format_complex_matrix(rho_0),
                    f"\nEigenvalues of ρ_S(0): {[round(v, 6) for v in evals_rho0]}",
                ]
            )
        )

    @staticmethod
    def _format_complex_matrix(mat: torch.Tensor) -> str:
        return "\n".join(
            "  [ " + "  ".join(f"{mat[r, c].item().real:+.4f}{mat[r, c].item().imag:+.4f}j" for c in range(4)) + " ]"
            for r in range(4)
        )

    def _update_plots(self, res: dict, mode: str):
        t_np = res["t"].numpy()
        is_correlated = mode != "pure_plus"

        self.ln_conc.set_data(t_np, res["concurrence"].numpy())
        self.ln_neg.set_data(t_np, res["negativity"].numpy())

        self.ln_gamma.set_data(t_np, res["gamma_t"].numpy())
        self.ln_gamma_cor.set_data(t_np, res["gamma_cor_12"].numpy())
        self.ln_gamma_tot.set_data(t_np, res["gamma_total_12"].numpy())
        self.ln_gamma_cor.set_visible(is_correlated)
        self.ln_gamma_tot.set_visible(is_correlated)
        self.zero_line.set_visible(is_correlated)

        tau1 = res["tau_rel_1"]
        if tau1 is not None:
            self.tau_line.set_xdata([tau1, tau1])
            self.tau_line.set_label(f"τ_rel (γ=1) = {tau1:.2f}")
            self.tau_line.set_visible(True)
        else:
            self.tau_line.set_visible(False)

        self.ln_purity.set_data(t_np, res["purity"].numpy())
        self.ln_entropy.set_data(t_np, res["vn_entropy"].numpy())

        self.ln_phi.set_data(t_np, res["Phi_t"].numpy())
        self.ln_delta.set_data(t_np, res["Delta_t"].numpy())
        self.ln_chi.set_data(t_np, res["chi_12"].numpy())
        self.ln_chi.set_visible(is_correlated)

        for ax in self.axes.flat:
            ax.relim(visible_only=True)
            ax.autoscale_view()
            handles = [h for h in ax.get_legend_handles_labels()[0] if h.get_visible()]
            labels = [h.get_label() for h in handles]
            ax.legend(handles, labels, loc="best", fontsize=8.5)

        self.canvas.draw_idle()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DephasingMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
