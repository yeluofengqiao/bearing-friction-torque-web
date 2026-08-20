import csv
import io
import math
import os
from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np
from flask import Flask, Response, render_template_string, request
from scipy.optimize import least_squares
from scipy.special import ellipe, ellipk


app = Flask(__name__)


@dataclass
class BallDetail:
    angle_deg: float
    load_q_n: float
    max_stress_mpa: float
    truncation_ratio_pct: float
    film_thickness_um: float
    outer_film_thickness_um: float
    central_film_thickness_um: float
    outer_central_film_thickness_um: float
    contact_angle_deg: float
    ehl_friction_force_n: float
    ehl_friction_torque_nmm: float
    ehl_power_loss_w: float
    traction_coeff_inner: float
    traction_coeff_outer: float
    estimated_slip_ratio_inner: float
    estimated_slip_ratio_outer: float
    lambda_value: float
    outer_lambda_value: float

    def to_dict(self):
        return asdict(self)


@dataclass
class CalculationResult:
    ehl_friction_torque_nmm: float
    ehl_friction_torque_nm: float
    ehl_power_loss_w: float
    radial_displacement_mm: float
    axial_displacement_mm: float
    operating_kinematic_viscosity_cst: float
    reference_kinematic_viscosity_cst: float
    kappa: float
    minimum_film_thickness_um: float
    minimum_outer_film_thickness_um: float
    minimum_lambda: float
    minimum_outer_lambda: float
    solver_converged: bool
    details: list[BallDetail]


@dataclass
class BearingParameters:
    Dw: float = 11.906
    Dm: float = 60.0
    Z: int = 9
    fi: float = 0.505
    fe: float = 0.525
    Pd: float = 0.010
    H_i: float = 2.3812
    E: float = 2.06e5
    nu: float = 0.3
    eta0: float = 0.015
    oil_density_kg_m3: float = 850.0
    composite_roughness_um: float = 0.052
    alpha: float = 1.5e-8
    shear_limit_factor: float = 0.03
    max_shear_stress_mpa: float = 80.0

    @property
    def L(self):
        return self.Dw * (self.fi + self.fe - 1)

    @property
    def reduced_modulus_mpa(self):
        """Two-body Hertz reduced modulus E* for identical steel bodies."""

        return self.E / (2.0 * (1.0 - self.nu**2))

    @property
    def ehl_modulus_mpa(self):
        """Hamrock-Dowson modulus convention, equal to 2 times Hertz E*."""

        return self.E / (1 - self.nu**2)

    @property
    def E_prime(self):
        """Backward-compatible alias for the EHL modulus convention."""

        return self.ehl_modulus_mpa

    def to_dict(self):
        return asdict(self)

    def validate(self):
        numeric_values = tuple(asdict(self).values())
        if any(not math.isfinite(float(value)) for value in numeric_values):
            raise ValueError("All bearing parameters must be finite numbers.")
        positive_fields = {
            "Dw": self.Dw,
            "Dm": self.Dm,
            "fi": self.fi,
            "fe": self.fe,
            "H_i": self.H_i,
            "E": self.E,
            "eta0": self.eta0,
            "oil_density_kg_m3": self.oil_density_kg_m3,
            "composite_roughness_um": self.composite_roughness_um,
            "alpha": self.alpha,
            "max_shear_stress_mpa": self.max_shear_stress_mpa,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0.")

        if isinstance(self.Z, bool) or not isinstance(self.Z, Integral):
            raise ValueError("Steel ball count Z must be an integer.")
        if self.Z < 1:
            raise ValueError("Steel ball count Z must be at least 1.")
        if self.Pd < 0:
            raise ValueError("Diametral clearance Pd cannot be negative.")
        if self.shear_limit_factor <= 0 or self.shear_limit_factor >= 1:
            raise ValueError("shear_limit_factor must be between 0 and 1.")
        if not 0 < self.nu < 0.5:
            raise ValueError("Poisson ratio nu must be between 0 and 0.5.")
        if self.Dm <= self.Dw:
            raise ValueError("Pitch diameter Dm must be greater than ball diameter Dw.")
        if self.fi + self.fe <= 1:
            raise ValueError("Curvature coefficients fi + fe must be greater than 1.")

        r_inner = self.fi * self.Dw
        if self.H_i >= 2 * r_inner:
            raise ValueError("Groove depth H_i exceeds the allowable geometric range.")


def astm_d341_kinematic_viscosity_cst(nu_40_cst, nu_100_cst, temperature_c):
    if not all(math.isfinite(float(value)) for value in (nu_40_cst, nu_100_cst, temperature_c)):
        raise ValueError("ASTM D341 inputs must be finite.")
    if nu_40_cst <= 0 or nu_100_cst <= 0:
        raise ValueError("ASTM D341 requires both nu_40_cst and nu_100_cst > 0.")
    if nu_40_cst <= nu_100_cst:
        raise ValueError("ASTM D341 requires nu_40_cst to be greater than nu_100_cst.")

    temperature_k = temperature_c + 273.15
    if temperature_k <= 0:
        raise ValueError("Oil temperature must be above absolute zero.")

    x_40 = math.log10(40 + 273.15)
    x_100 = math.log10(100 + 273.15)
    y_40 = math.log10(math.log10(nu_40_cst + 0.7))
    y_100 = math.log10(math.log10(nu_100_cst + 0.7))

    slope = (y_40 - y_100) / (x_100 - x_40)
    intercept = y_40 + slope * x_40
    y_temp = intercept - slope * math.log10(temperature_k)
    return (10 ** (10**y_temp)) - 0.7


def dynamic_viscosity_from_kinematic_cst(nu_cst, density_kg_m3):
    if nu_cst <= 0 or density_kg_m3 <= 0:
        raise ValueError("Kinematic viscosity and density must both be > 0.")
    return nu_cst * 1e-6 * density_kg_m3


def kinematic_viscosity_from_dynamic_pa_s(dynamic_viscosity_pa_s, density_kg_m3):
    if dynamic_viscosity_pa_s <= 0 or density_kg_m3 <= 0:
        raise ValueError("Dynamic viscosity and density must both be > 0.")
    return dynamic_viscosity_pa_s / density_kg_m3 * 1e6


def reference_kinematic_viscosity_cst(speed_rpm, pitch_diameter_mm):
    if speed_rpm <= 0 or pitch_diameter_mm <= 0:
        return 0.0
    if speed_rpm < 1000:
        return 45000 * (speed_rpm**-0.83) * (pitch_diameter_mm**-0.5)
    return 4500 * (speed_rpm**-0.5) * (pitch_diameter_mm**-0.5)


class BearingFrictionModel:
    def __init__(self, params=None):
        self.params = params or BearingParameters()
        self.params.validate()

    def _solve_elliptical_param(self, cos_tau):
        cos_tau = np.clip(cos_tau, 1e-6, 0.9999)

        def objective(e):
            if e <= 0 or e >= 1:
                return 1.0
            k_val = ellipk(e**2)
            e_val = ellipe(e**2)
            return (
                ((2 - e**2) * e_val - 2 * (1 - e**2) * k_val) / (e**2 * e_val)
                - cos_tau
            )

        solution = least_squares(
            lambda value: [objective(float(value[0]))],
            [0.85],
            bounds=([1e-5], [0.99999]),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        if not solution.success or abs(objective(float(solution.x[0]))) > 1e-8:
            raise ValueError("Contact ellipse parameter solver did not converge.")
        e_sol = float(solution.x[0])
        k_val = ellipk(e_sol**2)
        e_val = ellipe(e_sol**2)
        k_ratio = 1 / np.sqrt(1 - e_sol**2)
        return k_val, e_val, k_ratio

    def get_contact_stiffness(self, is_inner):
        p = self.params
        rho11 = rho12 = 2 / p.Dw
        if is_inner:
            rho21 = -1 / (p.fi * p.Dw)
            rho22 = -2 / (p.Dm - p.Dw)
        else:
            rho21 = -1 / (p.fe * p.Dw)
            rho22 = 2 / (p.Dm + p.Dw)

        sum_rho = rho11 + rho12 + rho21 + rho22
        if sum_rho <= 0.0:
            raise ValueError("Invalid contact curvature combination.")
        diff_rho = (rho11 - rho12) + (rho21 - rho22)
        cos_tau = abs(diff_rho) / sum_rho

        k_el, e_el, k_hd = self._solve_elliptical_param(cos_tau)

        q_test = 1.0
        term_common_1n = (3 * q_test) / (2 * sum_rho * p.reduced_modulus_mpa)
        a_star = (2 * (k_hd**2) * e_el / np.pi) ** (1 / 3)
        delta_star = (2 * k_el) / (np.pi * a_star)

        delta_1n = delta_star * (sum_rho / 2) * (term_common_1n ** (2 / 3))
        k_calc = 1.0 / (delta_1n**1.5)
        rx_mm = 1.0 / (rho12 + rho22)
        return k_calc, k_hd, sum_rho, e_el, rx_mm

    def _get_hertz_params(self, q, sum_rho, k_ratio, e_val):
        if q <= 1e-5:
            return 0.0, 0.0, 0.0, 0.0

        term_common = (3 * q) / (2 * sum_rho * self.params.reduced_modulus_mpa)
        a_star = (2 * (k_ratio**2) * e_val / np.pi) ** (1 / 3)
        b_star = (2 * e_val / (np.pi * k_ratio)) ** (1 / 3)

        a = a_star * (term_common ** (1 / 3))
        b = b_star * (term_common ** (1 / 3))
        area = np.pi * a * b
        p_max = (1.5 * q) / area
        return area, a, b, p_max

    def _film_thicknesses_mm(self, q, rx_m, u_vel, k_hd):
        if q <= 1e-5 or rx_m <= 0 or u_vel <= 0:
            return 0.0, 0.0

        p = self.params
        ehl_modulus_pa = p.ehl_modulus_mpa * 1e6
        g_param = p.alpha * ehl_modulus_pa
        u_dimless = (p.eta0 * u_vel) / (ehl_modulus_pa * rx_m)
        w_dimless = q / (ehl_modulus_pa * rx_m**2)
        central_effect = 1 - 0.61 * np.exp(-0.73 * k_hd)
        minimum_effect = 1 - np.exp(-0.68 * k_hd)
        central_dimless = (
            2.69
            * (u_dimless**0.67)
            * (g_param**0.53)
            * (w_dimless**-0.067)
            * central_effect
        )
        minimum_dimless = (
            3.63
            * (u_dimless**0.68)
            * (g_param**0.49)
            * (w_dimless**-0.073)
            * minimum_effect
        )
        scale_mm = rx_m * 1000
        return central_dimless * scale_mm, minimum_dimless * scale_mm

    def _effective_viscosity(self, mean_pressure_pa):
        pressure_term = np.clip(self.params.alpha * mean_pressure_pa, 0.0, 25.0)
        return self.params.eta0 * np.exp(np.sqrt(pressure_term))

    def _ehl_shear_stress_pa(self, sliding_speed, film_thickness_m, mean_pressure_pa):
        if sliding_speed <= 0 or film_thickness_m <= 0 or mean_pressure_pa <= 0:
            return 0.0

        eta_eff = self._effective_viscosity(mean_pressure_pa)
        tau_newton = eta_eff * sliding_speed / film_thickness_m
        tau_limit = min(
            self.params.shear_limit_factor * mean_pressure_pa,
            self.params.max_shear_stress_mpa * 1e6,
        )
        if tau_limit <= 0:
            return 0.0
        return tau_limit * np.tanh(tau_newton / tau_limit)

    def _estimate_slip_ratios(self, alpha_contact_rad):
        p = self.params
        geometry_inner = abs(p.fi - 0.5) / (p.fi + 0.5)
        geometry_outer = abs(p.fe - 0.5) / (p.fe + 0.5)
        spin_term = 0.5 * (p.Dw / p.Dm) * np.sin(alpha_contact_rad) ** 2

        slip_inner = np.clip(geometry_inner + spin_term, 0.001, 0.12)
        slip_outer = np.clip(geometry_outer + spin_term, 0.001, 0.12)
        return float(slip_inner), float(slip_outer)

    def calculate(self, fr, fa, speed_rpm):
        if not all(math.isfinite(float(value)) for value in (fr, fa, speed_rpm)):
            raise ValueError("Fr, Fa, and speed_rpm must be finite.")
        if fr < 0 or fa < 0 or speed_rpm < 0:
            raise ValueError("Fr, Fa, and speed_rpm cannot be negative.")

        p = self.params
        operating_kinematic_viscosity_cst = kinematic_viscosity_from_dynamic_pa_s(
            p.eta0, p.oil_density_kg_m3
        )
        reference_kinematic_viscosity = reference_kinematic_viscosity_cst(
            speed_rpm, p.Dm
        )
        kappa = 0.0
        if reference_kinematic_viscosity > 0:
            kappa = operating_kinematic_viscosity_cst / reference_kinematic_viscosity

        ki, ki_hd, sum_rho_i, e_val_i, rx_i_mm = self.get_contact_stiffness(
            is_inner=True
        )
        ke, ke_hd, sum_rho_e, e_val_e, rx_e_mm = self.get_contact_stiffness(
            is_inner=False
        )

        k_tot = 1 / (((1 / ki) ** (2 / 3) + (1 / ke) ** (2 / 3)) ** 1.5)
        angles = np.linspace(0, 2 * np.pi, p.Z, endpoint=False)

        def equilibrium_equations(vars_um):
            dr = vars_um[0] * 1e-3
            da = vars_um[1] * 1e-3
            fx = 0.0
            fz = 0.0
            for psi in angles:
                term_r = p.L + dr * np.cos(psi)
                term_a = da
                l_new = np.sqrt(term_r**2 + term_a**2)
                delta = l_new - p.L - (p.Pd / 2)
                if delta > 0:
                    q = k_tot * delta**1.5
                    fx += q * (term_r / l_new) * np.cos(psi)
                    fz += q * (term_a / l_new)
            return [fx - fr, fz - fa]

        load_scale = max(1.0, math.hypot(fr, fa))
        if fr <= 1e-12 and fa <= 1e-12:
            sol_um = np.zeros(2)
            solver_converged = True
        else:
            best_solution = None
            for guess in (
                np.array([50.0, 100.0]),
                np.array([20.0, 20.0]),
                np.array([100.0, 20.0]),
                np.array([100.0, 100.0]),
                np.array([250.0, 250.0]),
            ):
                solution = least_squares(
                    lambda values: np.asarray(equilibrium_equations(values)) / load_scale,
                    guess,
                    bounds=(np.zeros(2), np.full(2, np.inf)),
                    x_scale=np.array([50.0, 50.0]),
                    xtol=1e-11,
                    ftol=1e-11,
                    gtol=1e-11,
                    max_nfev=3000,
                )
                if best_solution is None or solution.cost < best_solution.cost:
                    best_solution = solution
            sol_um = best_solution.x
            scaled_residual = np.asarray(equilibrium_equations(sol_um)) / load_scale
            solver_converged = bool(
                best_solution.success and np.max(np.abs(scaled_residual)) < 1e-5
            )
        dr_mm = sol_um[0] * 1e-3
        da_mm = sol_um[1] * 1e-3

        total_ehl_torque_nm = 0.0
        total_ehl_power_w = 0.0
        details = []
        u_vel_i = (np.pi * speed_rpm * p.Dm / 120) * (1 - (p.Dw / p.Dm) ** 2) / 1000
        u_vel_e = (np.pi * speed_rpm * p.Dm / 120) * (1 + (p.Dw / p.Dm) ** 2) / 1000
        shaft_angular_speed_rad_s = 2.0 * np.pi * speed_rpm / 60.0

        r_inner = p.fi * p.Dw
        theta_edge_i = np.arccos(1.0 - p.H_i / r_inner)

        for psi in angles:
            term_r = p.L + dr_mm * np.cos(psi)
            term_a = da_mm
            l_new = np.sqrt(term_r**2 + term_a**2)
            delta = l_new - p.L - (p.Pd / 2)

            if delta <= 0:
                details.append(
                    BallDetail(
                        angle_deg=float(np.degrees(psi)),
                        load_q_n=0.0,
                        max_stress_mpa=0.0,
                        truncation_ratio_pct=0.0,
                        film_thickness_um=0.0,
                        outer_film_thickness_um=0.0,
                        central_film_thickness_um=0.0,
                        outer_central_film_thickness_um=0.0,
                        contact_angle_deg=0.0,
                        ehl_friction_force_n=0.0,
                        ehl_friction_torque_nmm=0.0,
                        ehl_power_loss_w=0.0,
                        traction_coeff_inner=0.0,
                        traction_coeff_outer=0.0,
                        estimated_slip_ratio_inner=0.0,
                        estimated_slip_ratio_outer=0.0,
                        lambda_value=0.0,
                        outer_lambda_value=0.0,
                    )
                )
                continue

            q = k_tot * delta**1.5
            alpha_contact = np.arcsin(term_a / l_new)
            slip_ratio_inner, slip_ratio_outer = self._estimate_slip_ratios(alpha_contact)
            delta_u_i = slip_ratio_inner * u_vel_i
            delta_u_e = slip_ratio_outer * u_vel_e

            rx_i = rx_i_mm / 1000
            area_i, a_i, _, pmax_i = self._get_hertz_params(q, sum_rho_i, ki_hd, e_val_i)
            h_c_i, h_min_i = self._film_thicknesses_mm(q, rx_i, u_vel_i, ki_hd)
            inner_film_um = h_min_i * 1000
            inner_central_film_um = h_c_i * 1000
            lambda_i = inner_film_um / p.composite_roughness_um

            s_avail_i = r_inner * (theta_edge_i - alpha_contact)
            if a_i > s_avail_i:
                trunc_ratio_i = ((a_i - s_avail_i) / (2 * a_i)) * 100.0
            else:
                trunc_ratio_i = 0.0

            rx_e = rx_e_mm / 1000
            area_e, _, _, _ = self._get_hertz_params(q, sum_rho_e, ke_hd, e_val_e)
            h_c_e, h_min_e = self._film_thicknesses_mm(q, rx_e, u_vel_e, ke_hd)
            outer_film_um = h_min_e * 1000
            outer_central_film_um = h_c_e * 1000
            lambda_e = outer_film_um / p.composite_roughness_um

            area_i_m2 = area_i * 1e-6
            area_e_m2 = area_e * 1e-6
            mean_pressure_i_pa = q / area_i_m2 if area_i_m2 > 0 else 0.0
            mean_pressure_e_pa = q / area_e_m2 if area_e_m2 > 0 else 0.0
            tau_i = self._ehl_shear_stress_pa(delta_u_i, h_c_i * 1e-3, mean_pressure_i_pa)
            tau_e = self._ehl_shear_stress_pa(delta_u_e, h_c_e * 1e-3, mean_pressure_e_pa)
            friction_force_i = tau_i * area_i_m2
            friction_force_e = tau_e * area_e_m2
            friction_force_ball = friction_force_i + friction_force_e
            ball_power_w = friction_force_i * delta_u_i + friction_force_e * delta_u_e
            ball_torque_nm = (
                ball_power_w / shaft_angular_speed_rad_s
                if shaft_angular_speed_rad_s > 0.0
                else 0.0
            )
            total_ehl_power_w += ball_power_w
            total_ehl_torque_nm += ball_torque_nm

            details.append(
                BallDetail(
                    angle_deg=float(np.degrees(psi)),
                    load_q_n=float(q),
                    max_stress_mpa=float(pmax_i),
                    truncation_ratio_pct=float(trunc_ratio_i),
                    film_thickness_um=float(inner_film_um),
                    outer_film_thickness_um=float(outer_film_um),
                    central_film_thickness_um=float(inner_central_film_um),
                    outer_central_film_thickness_um=float(outer_central_film_um),
                    contact_angle_deg=float(np.degrees(alpha_contact)),
                    ehl_friction_force_n=float(friction_force_ball),
                    ehl_friction_torque_nmm=float(ball_torque_nm * 1000),
                    ehl_power_loss_w=float(ball_power_w),
                    traction_coeff_inner=float(friction_force_i / q),
                    traction_coeff_outer=float(friction_force_e / q),
                    estimated_slip_ratio_inner=float(slip_ratio_inner),
                    estimated_slip_ratio_outer=float(slip_ratio_outer),
                    lambda_value=float(lambda_i),
                    outer_lambda_value=float(lambda_e),
                )
            )

        active_details = [detail for detail in details if detail.load_q_n > 0]
        minimum_film_thickness_um = min(
            (detail.film_thickness_um for detail in active_details),
            default=0.0,
        )
        minimum_outer_film_thickness_um = min(
            (detail.outer_film_thickness_um for detail in active_details),
            default=0.0,
        )
        minimum_lambda = min(
            (detail.lambda_value for detail in active_details),
            default=0.0,
        )
        minimum_outer_lambda = min(
            (detail.outer_lambda_value for detail in active_details),
            default=0.0,
        )

        return CalculationResult(
            ehl_friction_torque_nmm=float(total_ehl_torque_nm * 1000),
            ehl_friction_torque_nm=float(total_ehl_torque_nm),
            ehl_power_loss_w=float(total_ehl_power_w),
            radial_displacement_mm=float(dr_mm),
            axial_displacement_mm=float(da_mm),
            operating_kinematic_viscosity_cst=float(operating_kinematic_viscosity_cst),
            reference_kinematic_viscosity_cst=float(reference_kinematic_viscosity),
            kappa=float(kappa),
            minimum_film_thickness_um=float(minimum_film_thickness_um),
            minimum_outer_film_thickness_um=float(minimum_outer_film_thickness_um),
            minimum_lambda=float(minimum_lambda),
            minimum_outer_lambda=float(minimum_outer_lambda),
            solver_converged=solver_converged,
            details=details,
        )


INPUT_GROUPS = [
    {
        "title": "工况输入",
        "description": "这三项定义当前载荷和转速，是摩擦力矩求解的直接工况边界。",
        "fields": [
            {
                "name": "fr",
                "label": "径向载荷 Fr",
                "unit": "N",
                "type": "float",
                "default": 3000.0,
                "help": "轴承承受的径向外载荷。",
            },
            {
                "name": "fa",
                "label": "轴向载荷 Fa",
                "unit": "N",
                "type": "float",
                "default": 1500.0,
                "help": "轴承承受的轴向外载荷。",
            },
            {
                "name": "speed_rpm",
                "label": "转速 n",
                "unit": "rpm",
                "type": "float",
                "default": 3000.0,
                "help": "主要影响滚动速度、膜厚和摩擦力矩。",
            },
        ],
    },
    {
        "title": "轴承几何",
        "description": "这一组决定载荷分布、接触角、沟道截断和单球摩擦力矩。",
        "fields": [
            {
                "name": "Dw",
                "label": "钢球直径 Dw",
                "unit": "mm",
                "type": "float",
                "default": 11.906,
                "help": "滚动体钢球直径，是接触与膜厚计算的核心尺寸。",
            },
            {
                "name": "Dm",
                "label": "节圆直径 Dm",
                "unit": "mm",
                "type": "float",
                "default": 60.0,
                "help": "钢球中心轨迹所在的节圆直径。",
            },
            {
                "name": "Z",
                "label": "钢球数 Z",
                "unit": "个",
                "type": "int",
                "default": 9,
                "help": "轴承中参与分布的钢球数量。",
            },
            {
                "name": "fi",
                "label": "内圈曲率系数 fi",
                "unit": "-",
                "type": "float",
                "default": 0.505,
                "help": "内圈沟道曲率系数。",
            },
            {
                "name": "fe",
                "label": "外圈曲率系数 fe",
                "unit": "-",
                "type": "float",
                "default": 0.525,
                "help": "外圈沟道曲率系数。",
            },
            {
                "name": "Pd",
                "label": "直径游隙 Pd",
                "unit": "mm",
                "type": "float",
                "default": 0.010,
                "help": "影响钢球何时进入受载区。",
            },
            {
                "name": "H_i",
                "label": "内圈沟道深度 H_i",
                "unit": "mm",
                "type": "float",
                "default": 2.3812,
                "help": "用于判断接触椭圆是否发生截断。",
            },
        ],
    },
    {
        "title": "材料与润滑",
        "description": "这组参数决定黏度、赫兹接触、膜厚、kappa、lambda 与 EHL 剪切力矩。",
        "fields": [
            {
                "name": "E",
                "label": "杨氏模量 E",
                "unit": "MPa",
                "type": "float",
                "default": 2.06e5,
                "help": "钢材料可先用 2.06e5 MPa。",
            },
            {
                "name": "nu",
                "label": "泊松比 nu",
                "unit": "-",
                "type": "float",
                "default": 0.3,
                "help": "钢材料常取 0.30 左右。",
            },
            {
                "name": "nu_40_cst",
                "label": "运动黏度 nu40",
                "unit": "cSt",
                "type": "float",
                "default": 0.0,
                "help": "若与 nu100 同时输入，优先按 ASTM D341 推算当前黏度。",
            },
            {
                "name": "nu_100_cst",
                "label": "运动黏度 nu100",
                "unit": "cSt",
                "type": "float",
                "default": 0.0,
                "help": "需与 nu40 成对输入。",
            },
            {
                "name": "oil_temperature_c",
                "label": "油温",
                "unit": "degC",
                "type": "float",
                "default": 90.0,
                "help": "用于黏温换算。",
            },
            {
                "name": "oil_density_kg_m3",
                "label": "油品密度",
                "unit": "kg/m^3",
                "type": "float",
                "default": 850.0,
                "help": "用于运动黏度与动力黏度互算。",
            },
            {
                "name": "eta0",
                "label": "动力黏度 eta0",
                "unit": "Pa.s",
                "type": "float",
                "default": 0.015,
                "help": "当 nu40 和 nu100 都为 0 时直接采用。",
            },
            {
                "name": "composite_roughness_um",
                "label": "综合粗糙度 sigma",
                "unit": "um",
                "type": "float",
                "default": 0.052,
                "help": "lambda = h / sigma。",
            },
            {
                "name": "alpha",
                "label": "压黏系数 alpha",
                "unit": "1/Pa",
                "type": "float",
                "default": 1.5e-8,
                "help": "压力升高时润滑油黏度增加的敏感系数。",
            },
            {
                "name": "shear_limit_factor",
                "label": "极限剪应力系数",
                "unit": "-",
                "type": "float",
                "default": 0.03,
                "help": "油膜极限牵引剪应力近似取平均接触压强乘以该系数。",
            },
            {
                "name": "max_shear_stress_mpa",
                "label": "最大剪应力上限",
                "unit": "MPa",
                "type": "float",
                "default": 80.0,
                "help": "用于抑制高压下 Barus 压黏模型的过大剪应力。",
            },
        ],
    },
]


def iter_fields():
    for group in INPUT_GROUPS:
        for field in group["fields"]:
            yield field


FIELD_MAP = {field["name"]: field for field in iter_fields()}


def build_default_inputs():
    defaults = {
        "fr": 3000.0,
        "fa": 1500.0,
        "speed_rpm": 3000.0,
        "nu_40_cst": 0.0,
        "nu_100_cst": 0.0,
        "oil_temperature_c": 90.0,
    }
    defaults.update(BearingParameters().to_dict())
    return defaults


DEFAULT_INPUTS = build_default_inputs()


def parameter_notes():
    return [
        "如果同时输入 nu40 和 nu100，程序会先按 ASTM D341 计算当前温度下的运动黏度，再结合密度换算为动力黏度 eta0。",
        "kappa 按参考黏度法计算：kappa = nu / nu1，其中 nu1 由节圆直径 Dm 和转速 n 估算。",
        "lambda 按 lambda = h_min / sigma 计算，h_min 使用 Hamrock-Dowson 最小膜厚；中央膜厚仅用于油膜剪切近似。",
        "内圈和外圈滑滚比不需要手工输入，程序会根据沟道曲率、Dw/Dm 比值和接触角自动估算。",
        "总力矩按能量闭合：每个接触的剪切耗散功率求和后除以轴角速度；滑滚比和牵引模型仍属于需用实测力矩标定的工程代理。",
        "当前网页只保留摩擦力矩分析，不包含电容或 PPS 包塑层结果。",
    ]


def display_error(message):
    replacements = {
        "must be greater than 0.": "必须大于 0。",
        "must be between 0 and 1.": "需要在 0 和 1 之间。",
        "must be between 0 and 0.5.": "需要在 0 和 0.5 之间。",
        "Steel ball count Z must be at least 1.": "钢球数 Z 必须至少为 1。",
        "Diametral clearance Pd cannot be negative.": "直径游隙 Pd 不能为负数。",
        "Pitch diameter Dm must be greater than ball diameter Dw.": "节圆直径 Dm 必须大于钢球直径 Dw。",
        "Curvature coefficients fi + fe must be greater than 1.": "沟道曲率系数 fi + fe 必须大于 1。",
        "Groove depth H_i exceeds the allowable geometric range.": "内圈沟道深度 H_i 超出几何允许范围。",
        "ASTM D341 requires both nu_40_cst and nu_100_cst > 0.": "ASTM D341 黏度换算要求 nu40 和 nu100 都大于 0。",
        "Kinematic viscosity and density must both be > 0.": "运动黏度和密度都必须大于 0。",
        "Dynamic viscosity and density must both be > 0.": "动力黏度和密度都必须大于 0。",
        "Oil temperature must be above absolute zero.": "油温必须高于绝对零度。",
        "Fr, Fa, and speed_rpm cannot be negative.": "Fr、Fa 和转速不能为负数。",
    }
    return replacements.get(message, message)


def resolve_viscosity(inputs):
    density = inputs["oil_density_kg_m3"]
    nu_40_cst = inputs["nu_40_cst"]
    nu_100_cst = inputs["nu_100_cst"]

    if (nu_40_cst > 0) != (nu_100_cst > 0):
        raise ValueError("nu40 和 nu100 需要成对输入；要么都填，要么都留为 0。")

    if nu_40_cst > 0 and nu_100_cst > 0:
        operating_kinematic_viscosity_cst = astm_d341_kinematic_viscosity_cst(
            nu_40_cst,
            nu_100_cst,
            inputs["oil_temperature_c"],
        )
        eta0 = dynamic_viscosity_from_kinematic_cst(
            operating_kinematic_viscosity_cst,
            density,
        )
        viscosity_source = "ASTM D341"
    else:
        eta0 = inputs["eta0"]
        operating_kinematic_viscosity_cst = eta0 / density * 1e6
        viscosity_source = "直接输入 eta0"

    return eta0, operating_kinematic_viscosity_cst, viscosity_source


def parse_inputs(source):
    values = {}
    for name, field in FIELD_MAP.items():
        raw_value = str(source.get(name, DEFAULT_INPUTS[name])).strip()
        if raw_value == "":
            raise ValueError(f"{field['label']} 不能为空。")

        try:
            if field["type"] == "int":
                values[name] = int(raw_value)
            else:
                values[name] = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{field['label']} 需要输入有效数字。") from exc

    return values


def build_parameters(inputs):
    eta0, _, _ = resolve_viscosity(inputs)
    return BearingParameters(
        Dw=inputs["Dw"],
        Dm=inputs["Dm"],
        Z=inputs["Z"],
        fi=inputs["fi"],
        fe=inputs["fe"],
        Pd=inputs["Pd"],
        H_i=inputs["H_i"],
        E=inputs["E"],
        nu=inputs["nu"],
        eta0=eta0,
        oil_density_kg_m3=inputs["oil_density_kg_m3"],
        composite_roughness_um=inputs["composite_roughness_um"],
        alpha=inputs["alpha"],
        shear_limit_factor=inputs["shear_limit_factor"],
        max_shear_stress_mpa=inputs["max_shear_stress_mpa"],
    )


def detail_rows(result):
    rows = []
    for detail in result.details:
        if detail.truncation_ratio_pct == 0.0:
            truncation_status = "0.0% (安全)"
        elif detail.truncation_ratio_pct <= 15.0:
            truncation_status = f"{detail.truncation_ratio_pct:.1f}% (允许)"
        else:
            truncation_status = f"{detail.truncation_ratio_pct:.1f}% (NG/超标)"

        rows.append(
            {
                "angle_deg": detail.angle_deg,
                "load_q_n": detail.load_q_n,
                "max_stress_mpa": detail.max_stress_mpa,
                "truncation_ratio_pct": detail.truncation_ratio_pct,
                "truncation_status": truncation_status,
                "film_thickness_um": detail.film_thickness_um,
                "outer_film_thickness_um": detail.outer_film_thickness_um,
                "central_film_thickness_um": detail.central_film_thickness_um,
                "outer_central_film_thickness_um": detail.outer_central_film_thickness_um,
                "contact_angle_deg": detail.contact_angle_deg,
                "ehl_friction_force_n": detail.ehl_friction_force_n,
                "ehl_friction_torque_nmm": detail.ehl_friction_torque_nmm,
                "ehl_power_loss_w": detail.ehl_power_loss_w,
                "traction_coeff_inner": detail.traction_coeff_inner,
                "traction_coeff_outer": detail.traction_coeff_outer,
                "estimated_slip_ratio_inner": detail.estimated_slip_ratio_inner,
                "estimated_slip_ratio_outer": detail.estimated_slip_ratio_outer,
                "lambda_value": detail.lambda_value,
                "outer_lambda_value": detail.outer_lambda_value,
                "is_active": detail.load_q_n > 0,
            }
        )
    return rows


def build_summary(result, rows, viscosity_source):
    active_rows = [row for row in rows if row["is_active"]]
    peak_load = max((row["load_q_n"] for row in active_rows), default=0.0)
    peak_contact_angle = max((row["contact_angle_deg"] for row in active_rows), default=0.0)
    peak_ball_torque = max((row["ehl_friction_torque_nmm"] for row in active_rows), default=0.0)
    peak_inner_slip_ratio = max(
        (row["estimated_slip_ratio_inner"] for row in active_rows),
        default=0.0,
    )
    peak_outer_slip_ratio = max(
        (row["estimated_slip_ratio_outer"] for row in active_rows),
        default=0.0,
    )
    peak_stress = max((row["max_stress_mpa"] for row in active_rows), default=0.0)

    return {
        "active_ball_count": len(active_rows),
        "peak_load_q_n": peak_load,
        "peak_contact_angle_deg": peak_contact_angle,
        "peak_ball_torque_nmm": peak_ball_torque,
        "peak_inner_slip_ratio": peak_inner_slip_ratio,
        "peak_outer_slip_ratio": peak_outer_slip_ratio,
        "peak_stress_mpa": peak_stress,
        "operating_kinematic_viscosity_cst": result.operating_kinematic_viscosity_cst,
        "reference_kinematic_viscosity_cst": result.reference_kinematic_viscosity_cst,
        "kappa": result.kappa,
        "minimum_film_thickness_um": result.minimum_film_thickness_um,
        "minimum_outer_film_thickness_um": result.minimum_outer_film_thickness_um,
        "minimum_lambda": result.minimum_lambda,
        "minimum_outer_lambda": result.minimum_outer_lambda,
        "total_ehl_torque_nmm": result.ehl_friction_torque_nmm,
        "total_ehl_torque_nm": result.ehl_friction_torque_nm,
        "total_ehl_power_w": result.ehl_power_loss_w,
        "solver_status": "求解收敛" if result.solver_converged else "求解未收敛",
        "viscosity_source": viscosity_source,
    }


def run_calculation(inputs):
    _, _, viscosity_source = resolve_viscosity(inputs)
    model = BearingFrictionModel(build_parameters(inputs))
    result = model.calculate(
        fr=inputs["fr"],
        fa=inputs["fa"],
        speed_rpm=inputs["speed_rpm"],
    )
    rows = detail_rows(result)
    summary = build_summary(result, rows, viscosity_source)
    return result, rows, summary


def build_csv(rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "angle_deg",
            "load_q_n",
            "contact_angle_deg",
            "max_stress_mpa",
            "truncation_ratio_pct",
            "film_thickness_inner_um",
            "film_thickness_outer_um",
            "central_film_thickness_inner_um",
            "central_film_thickness_outer_um",
            "lambda_inner",
            "lambda_outer",
            "estimated_slip_ratio_inner",
            "estimated_slip_ratio_outer",
            "ehl_friction_force_n",
            "ehl_friction_torque_nmm",
            "ehl_power_loss_w",
            "traction_coeff_inner",
            "traction_coeff_outer",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                f"{row['angle_deg']:.0f}",
                f"{row['load_q_n']:.4f}",
                f"{row['contact_angle_deg']:.4f}",
                f"{row['max_stress_mpa']:.4f}",
                f"{row['truncation_ratio_pct']:.4f}",
                f"{row['film_thickness_um']:.6f}",
                f"{row['outer_film_thickness_um']:.6f}",
                f"{row['central_film_thickness_um']:.6f}",
                f"{row['outer_central_film_thickness_um']:.6f}",
                f"{row['lambda_value']:.6f}",
                f"{row['outer_lambda_value']:.6f}",
                f"{row['estimated_slip_ratio_inner']:.6f}",
                f"{row['estimated_slip_ratio_outer']:.6f}",
                f"{row['ehl_friction_force_n']:.6f}",
                f"{row['ehl_friction_torque_nmm']:.6f}",
                f"{row['ehl_power_loss_w']:.9f}",
                f"{row['traction_coeff_inner']:.6f}",
                f"{row['traction_coeff_outer']:.6f}",
            ]
        )
    return buffer.getvalue()


PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>球轴承摩擦力矩计算器</title>
    <style>
      :root {
        --bg: #eef2f5;
        --card: rgba(255, 255, 255, 0.94);
        --card-strong: #ffffff;
        --ink: #142033;
        --muted: #5d6878;
        --line: rgba(20, 32, 51, 0.12);
        --accent: #0d8a6a;
        --accent-2: #ff8b38;
        --shadow: 0 20px 55px rgba(16, 28, 45, 0.12);
        --radius: 22px;
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        font-family: "Segoe UI", "PingFang SC", "Noto Sans SC", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(13, 138, 106, 0.16), transparent 28%),
          radial-gradient(circle at top right, rgba(255, 139, 56, 0.16), transparent 26%),
          linear-gradient(180deg, #f7fafb 0%, var(--bg) 100%);
      }

      .page {
        width: min(1240px, calc(100vw - 32px));
        margin: 0 auto;
        padding: 28px 0 60px;
      }

      .hero {
        display: grid;
        grid-template-columns: minmax(0, 1.7fr) minmax(280px, 0.9fr);
        gap: 20px;
        align-items: stretch;
        margin-bottom: 22px;
      }

      .hero-card,
      .panel,
      .metric-card,
      .summary-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
      }

      .hero-main {
        padding: 30px;
      }

      .eyebrow {
        display: inline-flex;
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.04em;
        color: var(--accent);
        background: rgba(13, 138, 106, 0.1);
      }

      h1, h2, h3, p {
        margin: 0;
      }

      h1 {
        margin-top: 14px;
        font-size: clamp(30px, 4vw, 48px);
        line-height: 1.06;
      }

      .hero-main p {
        margin-top: 14px;
        max-width: 48rem;
        color: var(--muted);
        line-height: 1.72;
        font-size: 16px;
      }

      .hero-side {
        padding: 24px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        background:
          linear-gradient(150deg, rgba(13, 138, 106, 0.9), rgba(16, 43, 58, 0.95));
        color: #f3fbf9;
      }

      .hero-side strong {
        display: block;
        margin-top: 10px;
        font-size: 22px;
      }

      .hero-side small {
        display: block;
        margin-top: 10px;
        color: rgba(243, 251, 249, 0.8);
        line-height: 1.6;
      }

      .hero-tags {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 18px;
      }

      .hero-tags span {
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 999px;
        padding: 8px 12px;
        font-size: 13px;
      }

      .panel {
        padding: 22px;
        margin-bottom: 18px;
      }

      .panel h2 {
        font-size: 24px;
        margin-bottom: 10px;
      }

      .panel .subtext {
        color: var(--muted);
        line-height: 1.7;
      }

      .notes-list {
        margin: 0;
        padding-left: 18px;
        color: var(--muted);
        line-height: 1.8;
      }

      .notes-list li + li {
        margin-top: 6px;
      }

      .input-section + .input-section {
        margin-top: 18px;
      }

      .section-head {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: end;
        margin-bottom: 16px;
      }

      .section-head p {
        color: var(--muted);
        max-width: 38rem;
        line-height: 1.6;
      }

      .input-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 14px;
      }

      .field {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }

      .field span {
        font-size: 14px;
        font-weight: 700;
      }

      .field input {
        width: 100%;
        padding: 14px 16px;
        border-radius: 16px;
        border: 1px solid rgba(20, 32, 51, 0.12);
        background: #fcfdfe;
        color: var(--ink);
        font-size: 15px;
        outline: none;
      }

      .field input:focus {
        border-color: rgba(13, 138, 106, 0.5);
        box-shadow: 0 0 0 4px rgba(13, 138, 106, 0.12);
      }

      .field-help {
        color: var(--muted);
        line-height: 1.5;
      }

      .submit-row {
        display: flex;
        gap: 12px;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
      }

      .button,
      .ghost-button {
        appearance: none;
        border: 0;
        cursor: pointer;
        border-radius: 999px;
        padding: 14px 22px;
        font-size: 15px;
        font-weight: 700;
      }

      .button {
        color: white;
        background: linear-gradient(135deg, var(--accent), #0a6f56);
      }

      .ghost-button {
        color: var(--ink);
        background: rgba(20, 32, 51, 0.06);
      }

      .submit-note {
        color: var(--muted);
        line-height: 1.6;
      }

      .alert {
        border-color: rgba(191, 73, 45, 0.2);
        background: rgba(255, 245, 242, 0.95);
      }

      .alert strong {
        color: #b54825;
        display: block;
        margin-bottom: 6px;
      }

      .metrics,
      .summary-grid {
        display: grid;
        gap: 14px;
      }

      .metrics {
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        margin-bottom: 16px;
      }

      .summary-grid {
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        margin-bottom: 18px;
      }

      .metric-card,
      .summary-card {
        padding: 20px;
      }

      .metric-card span,
      .summary-card span {
        display: block;
        color: var(--muted);
        font-size: 14px;
      }

      .metric-card strong,
      .summary-card strong {
        display: block;
        margin-top: 8px;
        font-size: 28px;
        line-height: 1.15;
      }

      .summary-card strong.good {
        color: var(--accent);
      }

      .summary-card strong.warn {
        color: #bf492d;
      }

      .table-head {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: center;
        margin-bottom: 16px;
        flex-wrap: wrap;
      }

      .table-shell {
        overflow: auto;
        border-radius: 20px;
        border: 1px solid rgba(20, 32, 51, 0.08);
      }

      table {
        width: 100%;
        border-collapse: collapse;
        min-width: 1120px;
        background: var(--card-strong);
      }

      thead th {
        position: sticky;
        top: 0;
        background: #f4f7f9;
        z-index: 1;
        font-size: 13px;
        color: var(--muted);
        text-align: left;
        letter-spacing: 0.02em;
      }

      th, td {
        padding: 14px 16px;
        border-bottom: 1px solid rgba(20, 32, 51, 0.07);
        white-space: nowrap;
      }

      tbody tr:hover {
        background: rgba(13, 138, 106, 0.04);
      }

      tbody tr.inactive {
        color: #8a93a1;
        background: rgba(20, 32, 51, 0.025);
      }

      @media (max-width: 900px) {
        .hero {
          grid-template-columns: 1fr;
        }

        .page {
          width: min(100vw - 20px, 1240px);
          padding-top: 18px;
        }

        .hero-main,
        .hero-side,
        .panel,
        .metric-card,
        .summary-card {
          padding: 18px;
        }

        .section-head {
          flex-direction: column;
          align-items: start;
        }
      }
    </style>
  </head>
  <body>
    <main class="page">
      <section class="hero">
        <article class="hero-card hero-main">
          <span class="eyebrow">独立第二通道</span>
          <h1>球轴承摩擦力矩在线计算器</h1>
          <p>
            这是与旧站完全分开的新网页，只保留球轴承摩擦力矩分析。你可以输入载荷、转速、轴承几何和润滑参数，网页会自动完成载荷分布、赫兹接触、EHL 膜厚、kappa、lambda 和总摩擦力矩计算。
          </p>
        </article>
        <article class="hero-card hero-side">
          <div>
            <span>当前默认案例</span>
            <strong>6208 工况基线</strong>
            <small>默认值延续此前验证过的 6208 参数，方便你直接对比旧模型的摩擦力矩结果。</small>
          </div>
          <div class="hero-tags">
            <span>EHL 膜厚</span>
            <span>kappa / lambda</span>
            <span>逐钢球力矩</span>
          </div>
        </article>
      </section>

      <section class="panel">
        <h2>模型说明</h2>
        <ul class="notes-list">
          {% for note in parameter_notes %}
          <li>{{ note }}</li>
          {% endfor %}
        </ul>
      </section>

      <form method="post">
        {% for group in input_groups %}
        <section class="panel input-section">
          <div class="section-head">
            <div>
              <h2>{{ group.title }}</h2>
              <p>{{ group.description }}</p>
            </div>
          </div>
          <div class="input-grid">
            {% for field in group.fields %}
            <label class="field">
              <span>{{ field.label }}{% if field.unit %} ({{ field.unit }}){% endif %}</span>
              <input
                type="text"
                name="{{ field.name }}"
                value="{{ inputs[field.name] }}"
                inputmode="{{ 'numeric' if field.type == 'int' else 'decimal' }}"
                autocomplete="off"
                autocorrect="off"
                autocapitalize="off"
                spellcheck="false"
                required
              />
              <small class="field-help">{{ field.help }}</small>
            </label>
            {% endfor %}
          </div>
        </section>
        {% endfor %}

        <section class="panel">
          <div class="submit-row">
            <button type="submit" class="button">开始计算摩擦力矩</button>
            <div class="submit-note">输入为空、非数字或几何越界时，页面会直接给出中文提示。</div>
          </div>
        </section>
      </form>

      {% if error_message %}
      <section class="panel alert">
        <strong>输入错误</strong>
        <p>{{ error_message }}</p>
      </section>
      {% endif %}

      {% if result %}
      <section class="metrics">
        <article class="metric-card">
          <span>总 EHL 摩擦力矩</span>
          <strong>{{ "%.3f"|format(result.ehl_friction_torque_nmm) }} N.mm</strong>
        </article>
        <article class="metric-card">
          <span>总 EHL 摩擦力矩</span>
          <strong>{{ "%.6f"|format(result.ehl_friction_torque_nm) }} N.m</strong>
        </article>
        <article class="metric-card">
          <span>EHL 剪切耗散功率</span>
          <strong>{{ "%.3f"|format(result.ehl_power_loss_w) }} W</strong>
        </article>
        <article class="metric-card">
          <span>当前运动黏度 nu(T)</span>
          <strong>{{ "%.3f"|format(result.operating_kinematic_viscosity_cst) }} cSt</strong>
        </article>
        <article class="metric-card">
          <span>参考黏度 nu1</span>
          <strong>{{ "%.3f"|format(result.reference_kinematic_viscosity_cst) }} cSt</strong>
        </article>
        <article class="metric-card">
          <span>kappa</span>
          <strong>{{ "%.3f"|format(result.kappa) }}</strong>
        </article>
        <article class="metric-card">
          <span>最小内圈膜厚</span>
          <strong>{{ "%.3f"|format(result.minimum_film_thickness_um) }} um</strong>
        </article>
        <article class="metric-card">
          <span>最小外圈膜厚</span>
          <strong>{{ "%.3f"|format(result.minimum_outer_film_thickness_um) }} um</strong>
        </article>
        <article class="metric-card">
          <span>最小内圈 lambda</span>
          <strong>{{ "%.2f"|format(result.minimum_lambda) }}</strong>
        </article>
        <article class="metric-card">
          <span>最小外圈 lambda</span>
          <strong>{{ "%.2f"|format(result.minimum_outer_lambda) }}</strong>
        </article>
        <article class="metric-card">
          <span>径向 / 轴向位移</span>
          <strong>{{ "%.1f"|format(result.radial_displacement_mm * 1000) }} / {{ "%.1f"|format(result.axial_displacement_mm * 1000) }} um</strong>
        </article>
      </section>

      <section class="summary-grid">
        <article class="summary-card">
          <span>求解状态</span>
          <strong class="{{ 'good' if result.solver_converged else 'warn' }}">{{ summary.solver_status }}</strong>
        </article>
        <article class="summary-card">
          <span>黏度来源</span>
          <strong>{{ summary.viscosity_source }}</strong>
        </article>
        <article class="summary-card">
          <span>参与接触钢球数</span>
          <strong>{{ summary.active_ball_count }}</strong>
        </article>
        <article class="summary-card">
          <span>最大单球载荷</span>
          <strong>{{ "%.1f"|format(summary.peak_load_q_n) }} N</strong>
        </article>
        <article class="summary-card">
          <span>最大单球 EHL 力矩</span>
          <strong>{{ "%.3f"|format(summary.peak_ball_torque_nmm) }} N.mm</strong>
        </article>
        <article class="summary-card">
          <span>最大接触角</span>
          <strong>{{ "%.2f"|format(summary.peak_contact_angle_deg) }} deg</strong>
        </article>
        <article class="summary-card">
          <span>最大赫兹应力</span>
          <strong>{{ "%.2f"|format(summary.peak_stress_mpa) }} MPa</strong>
        </article>
        <article class="summary-card">
          <span>最大内 / 外滑滚比</span>
          <strong>{{ "%.4f"|format(summary.peak_inner_slip_ratio) }} / {{ "%.4f"|format(summary.peak_outer_slip_ratio) }}</strong>
        </article>
      </section>

      <section class="panel">
        <div class="table-head">
          <div>
            <h2>逐钢球分析明细</h2>
            <p class="subtext">下表展示每颗钢球在当前工况下的受载、最小膜厚、lambda、滑滚比和按耗散功率折算的单球摩擦力矩。</p>
          </div>
          {% if result.solver_converged %}
          <form method="get" action="{{ url_for('download_csv') }}">
            {% for group in input_groups %}
              {% for field in group.fields %}
            <input type="hidden" name="{{ field.name }}" value="{{ inputs[field.name] }}" />
              {% endfor %}
            {% endfor %}
            <button type="submit" class="ghost-button">下载 CSV</button>
          </form>
          {% endif %}
        </div>

        <div class="table-shell">
          <table>
            <thead>
              <tr>
                <th>角度 (deg)</th>
                <th>载荷 Q (N)</th>
                <th>接触角 (deg)</th>
                <th>最大应力 (MPa)</th>
                <th>截断状态</th>
                <th>内 / 外圈膜厚 (um)</th>
                <th>内 / 外圈 lambda</th>
                <th>内 / 外圈滑滚比</th>
                <th>总摩擦力 (N)</th>
                <th>单球 EHL 力矩 (N.mm)</th>
              </tr>
            </thead>
            <tbody>
              {% for row in rows %}
              <tr class="{{ 'inactive' if not row.is_active }}">
                <td>{{ "%.0f"|format(row.angle_deg) }}</td>
                <td>{{ "%.2f"|format(row.load_q_n) if row.is_active else "---" }}</td>
                <td>{{ "%.2f"|format(row.contact_angle_deg) if row.is_active else "---" }}</td>
                <td>{{ "%.2f"|format(row.max_stress_mpa) if row.is_active else "---" }}</td>
                <td>{{ row.truncation_status if row.is_active else "---" }}</td>
                <td>{{ "%.4f"|format(row.film_thickness_um) if row.is_active else "---" }} / {{ "%.4f"|format(row.outer_film_thickness_um) if row.is_active else "---" }}</td>
                <td>{{ "%.2f"|format(row.lambda_value) if row.is_active else "---" }} / {{ "%.2f"|format(row.outer_lambda_value) if row.is_active else "---" }}</td>
                <td>{{ "%.4f"|format(row.estimated_slip_ratio_inner) if row.is_active else "---" }} / {{ "%.4f"|format(row.estimated_slip_ratio_outer) if row.is_active else "---" }}</td>
                <td>{{ "%.6f"|format(row.ehl_friction_force_n) if row.is_active else "---" }}</td>
                <td>{{ "%.4f"|format(row.ehl_friction_torque_nmm) if row.is_active else "---" }}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </section>
      {% endif %}
    </main>
  </body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    inputs = DEFAULT_INPUTS.copy()
    result = None
    rows = []
    summary = None
    error_message = None

    if request.method == "POST":
        try:
            inputs = parse_inputs(request.form)
            result, rows, summary = run_calculation(inputs)
        except ValueError as exc:
            error_message = display_error(str(exc))

    return render_template_string(
        PAGE_TEMPLATE,
        error_message=error_message,
        input_groups=INPUT_GROUPS,
        inputs=inputs,
        parameter_notes=parameter_notes(),
        result=result,
        rows=rows,
        summary=summary,
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok"}, 200


@app.get("/download.csv")
def download_csv():
    try:
        inputs = parse_inputs(request.args)
        result, rows, _ = run_calculation(inputs)
    except ValueError as exc:
        return Response(display_error(str(exc)), status=400, mimetype="text/plain")

    if not result.solver_converged:
        return Response("求解器未收敛，未生成 CSV。", status=400, mimetype="text/plain")

    csv_text = build_csv(rows)
    return Response(
        csv_text,
        content_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": "attachment; filename=bearing_friction_results.csv"
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)
