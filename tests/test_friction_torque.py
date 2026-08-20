from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from friction_torque_app import (
    BearingFrictionModel,
    BearingParameters,
    app,
    astm_d341_kinematic_viscosity_cst,
)


class BearingFrictionModelTests(unittest.TestCase):
    def test_web_health_and_default_calculation_render(self) -> None:
        client = app.test_client()

        self.assertEqual(client.get("/healthz").status_code, 200)
        response = client.post("/", data={})
        self.assertEqual(response.status_code, 200)
        self.assertIn("EHL 剪切耗散功率".encode("utf-8"), response.data)

    def test_hertz_and_ehl_modulus_conventions_are_separate(self) -> None:
        parameters = BearingParameters()

        self.assertAlmostEqual(
            parameters.ehl_modulus_mpa,
            2.0 * parameters.reduced_modulus_mpa,
        )

    def test_minimum_film_is_below_central_film(self) -> None:
        model = BearingFrictionModel()
        _, ellipticity, _, _, radius_mm = model.get_contact_stiffness(is_inner=True)
        central_mm, minimum_mm = model._film_thicknesses_mm(
            500.0,
            radius_mm / 1000.0,
            1.0,
            ellipticity,
        )

        self.assertGreater(central_mm, 0.0)
        self.assertGreater(minimum_mm, 0.0)
        self.assertLess(minimum_mm, central_mm)

    def test_default_load_solution_closes_force_and_power_balance(self) -> None:
        speed_rpm = 3000.0
        result = BearingFrictionModel().calculate(3000.0, 1500.0, speed_rpm)

        self.assertTrue(result.solver_converged)
        radial_force = sum(
            detail.load_q_n
            * math.cos(math.radians(detail.contact_angle_deg))
            * math.cos(math.radians(detail.angle_deg))
            for detail in result.details
        )
        axial_force = sum(
            detail.load_q_n * math.sin(math.radians(detail.contact_angle_deg))
            for detail in result.details
        )
        self.assertAlmostEqual(radial_force, 3000.0, delta=0.05)
        self.assertAlmostEqual(axial_force, 1500.0, delta=0.05)

        shaft_angular_speed = 2.0 * math.pi * speed_rpm / 60.0
        self.assertAlmostEqual(
            result.ehl_friction_torque_nm * shaft_angular_speed,
            result.ehl_power_loss_w,
            places=10,
        )
        self.assertAlmostEqual(
            result.ehl_power_loss_w,
            sum(detail.ehl_power_loss_w for detail in result.details),
            places=10,
        )

    def test_zero_speed_has_no_hydrodynamic_film_or_shear_torque(self) -> None:
        result = BearingFrictionModel().calculate(3000.0, 1500.0, 0.0)

        self.assertTrue(result.solver_converged)
        self.assertEqual(result.minimum_film_thickness_um, 0.0)
        self.assertEqual(result.minimum_outer_film_thickness_um, 0.0)
        self.assertEqual(result.ehl_power_loss_w, 0.0)
        self.assertEqual(result.ehl_friction_torque_nm, 0.0)

    def test_astm_d341_reproduces_anchors_and_rejects_reversed_pair(self) -> None:
        self.assertAlmostEqual(astm_d341_kinematic_viscosity_cst(68.0, 8.8, 40.0), 68.0)
        self.assertAlmostEqual(astm_d341_kinematic_viscosity_cst(68.0, 8.8, 100.0), 8.8)
        with self.assertRaises(ValueError):
            astm_d341_kinematic_viscosity_cst(8.8, 68.0, 60.0)


if __name__ == "__main__":
    unittest.main()
