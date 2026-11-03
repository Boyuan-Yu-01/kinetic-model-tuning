"""Concise Cantera calculations for a specified premixed condition."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Literal, Mapping

import cantera as ct
import numpy as np


Composition = str | Mapping[str, float]


class CombustionCalculator:
    """Calculate flame speed, thicknesses, and ignition delay for one mixture.

    ``P`` is pressure in atm; ``T`` is the initial temperature in K.
    """

    def __init__(
        self,
        mechanism: str | Path,
        fuel: Composition,
        oxidizer: Composition,
        phi: float,
        P: float,
        T: float,
    ) -> None:
        """Set mixture conditions; pressure is supplied in atm and stored in Pa."""
        if not all(np.isfinite(v) and v > 0 for v in (phi, P, T)):
            raise ValueError("phi, P, and T must be positive.")

        self.mechanism = str(mechanism)
        self.fuel = fuel
        self.oxidizer = oxidizer
        self.phi = phi
        self.P = P * ct.one_atm
        self.T = T
        self.final_temperature: float | None = None
        self._flame = None
        self._flame_key = None
        self._flame_options = (0.01, "multicomponent", 3.0, 0.06, 0.12, 0)

    def _solve_flame(
        self,
        width: float | None = None,
        transport_model: str | None = None,
        refine_ratio: float | None = None,
        refine_slope: float | None = None,
        refine_curve: float | None = None,
        loglevel: int | None = None,
    ) -> ct.FreeFlame:
        """Reuse the last flame unless its inputs change."""
        supplied = (width, transport_model, refine_ratio, refine_slope, refine_curve, loglevel)
        options = tuple(old if new is None else new for old, new in zip(self._flame_options, supplied))
        width, transport_model, refine_ratio, refine_slope, refine_curve, loglevel = options
        mechanism = str(self.mechanism)
        paths = [Path(mechanism)] + [Path(d) / mechanism for d in ct.get_data_directories()]
        source = next((p.resolve() for p in paths if p.is_file()), None)
        identity = (str(source), sha256(source.read_bytes()).hexdigest()) if source else (mechanism, None)
        def composition_key(value):
            """Snapshot composition so in-place edits invalidate the flame cache."""
            return value if isinstance(value, str) else tuple(sorted(value.items()))
        key = (identity, composition_key(self.fuel), composition_key(self.oxidizer),
               self.phi, self.P, self.T, options)
        if self._flame is not None and key == self._flame_key:
            return self._flame
        gas = ct.Solution(self.mechanism)
        gas.TP = self.T, self.P
        gas.set_equivalence_ratio(self.phi, self.fuel, self.oxidizer)

        unburned_sound_speed = float(gas.sound_speed)
        flame = ct.FreeFlame(gas, width=width)
        flame.transport_model = transport_model
        flame.set_refine_criteria(
            ratio=refine_ratio,
            slope=refine_slope,
            curve=refine_curve,
        )
        flame.solve(loglevel=loglevel, auto=True)
        # Reject numerical convergence to a nonphysical freely propagating flame.
        speed = float(flame.velocity[0])
        if (not np.all(np.isfinite(flame.T)) or not np.isfinite(speed)
                or not 0 < speed < unburned_sound_speed or flame.T[-1] <= flame.T[0]):
            raise RuntimeError("Flame solution must have positive subsonic speed and a temperature rise.")
        self._flame, self._flame_key, self._flame_options = flame, key, options
        return flame

    def solve_flame(
        self,
        width: float | None = None,
        transport_model: str | None = None,
        refine_ratio: float | None = None,
        refine_slope: float | None = None,
        refine_curve: float | None = None,
        loglevel: int | None = None,
    ) -> tuple[float, float]:
        """Return (flame speed in m/s, burned temperature in K).

        Reuses a matching solved flame.

        Omitted options reuse the last successful solve's settings. Initial
        defaults: width=0.01 m, transport_model="multicomponent",
        refine_ratio=3, refine_slope=0.06, refine_curve=0.12, loglevel=0.
        """
        flame = self._solve_flame(
            width, transport_model, refine_ratio, refine_slope, refine_curve, loglevel
        )
        return float(flame.velocity[0]), float(flame.T[-1])

    def flame_thickness(self, **flame_options) -> float:
        """Return (Tb - Tu) / max(dT/dx) in metres.

        Accepts the same keyword options as solve_flame; reuses the cached flame.
        """
        flame = self._solve_flame(**flame_options)
        gradient = np.diff(flame.T) / np.diff(flame.grid)
        max_gradient = float(np.max(gradient))
        temperature_rise = float(flame.T[-1] - flame.T[0])
        # _solve_flame already validates the temperature rise and flame speed.
        if not np.isfinite(max_gradient) or max_gradient <= 0:
            raise RuntimeError("Flame thickness requires a finite positive temperature gradient.")
        return temperature_rise / max_gradient

    def diffusion_thickness(self, **flame_options) -> float:
        """Return unburned thermal diffusivity / flame speed in metres.

        Accepts the same keyword options as solve_flame; reuses the cached flame.
        Velocity stays in m/s to match thermal diffusivity in m^2/s.
        """
        flame = self._solve_flame(**flame_options)
        alpha = float(flame.thermal_conductivity[0] / (flame.density[0] * flame.cp[0]))
        speed = float(flame.velocity[0])
        if not np.isfinite(alpha) or alpha <= 0:
            raise RuntimeError("Diffusion thickness requires positive finite diffusivity.")
        return alpha / speed

    def ignition_delay(
        self,
        end_time: float = 1.0e-2,
        marker_species: str = "OH",
        reactor_type: Literal["constant-volume", "constant-pressure"] = "constant-volume",
        rtol: float = 1.0e-9,
        atol: float = 1.0e-15,
        max_time_step: float | None = None,
        anchor_species: tuple = (),
    ) -> float:
        """Return IDT in seconds; expose the last reactor temperature in K.

        Set marker_species="temperature" to use maximum dT/dt; otherwise
        ignition is the maximum species-concentration growth rate.
        ``final_temperature`` is sampled at integration termination, without
        a steady-state convergence check. The IDT return type stays unchanged.
        """
        self.final_temperature = None
        self.temperature_anchor = None
        if end_time <= 0:
            raise ValueError("end_time must be positive.")

        gas = ct.Solution(self.mechanism)
        gas.TP = self.T, self.P
        gas.set_equivalence_ratio(self.phi, self.fuel, self.oxidizer)
        if marker_species != "temperature" and marker_species not in gas.species_names:
            raise ValueError(f"Species {marker_species!r} is not in the mechanism.")

        reactor_classes = {
            "constant-volume": ct.IdealGasReactor,
            "constant-pressure": ct.IdealGasConstPressureReactor,
        }
        if reactor_type not in reactor_classes:
            raise ValueError(f"Unsupported reactor_type: {reactor_type!r}")

        reactor = reactor_classes[reactor_type](gas, energy="on", clone=True)
        network = ct.ReactorNet([reactor])
        network.rtol = rtol
        network.atol = atol
        if max_time_step is not None:
            if max_time_step <= 0:
                raise ValueError("max_time_step must be positive.")

        phase = reactor.phase
        species_index = None if marker_species == "temperature" else phase.species_index(marker_species)
        def marker_value():
            """Sample temperature or marker-species concentration for ignition timing."""
            return float(phase.T if species_index is None else phase.concentrations[species_index])
        def temperature_rate():
            capacity = phase.cv_mass if reactor_type == 'constant-volume' else phase.cp_mass
            energy = phase.partial_molar_int_energies if reactor_type == 'constant-volume' else phase.partial_molar_enthalpies
            return -float(np.dot(phase.net_production_rates,energy))/(phase.density*capacity)
        peak_temperature_rate = -np.inf
        def capture_anchor():
            nonlocal peak_temperature_rate
            if anchor_species:
                value=temperature_rate()
                if value>peak_temperature_rate:
                    from .order_tuning import state_anchor
                    self.temperature_anchor=state_anchor(phase,anchor_species)
                    peak_temperature_rate=value
        capture_anchor()
        rates = [temperature_rate()] if species_index is None else []
        times = [0.0]
        concentrations = [marker_value()]

        while network.time < end_time:
            # Retain adaptive samples without integrating beyond the requested endpoint.
            network.max_time_step = min(max_time_step or end_time, end_time - network.time)
            times.append(network.step())
            concentrations.append(marker_value())
            capture_anchor()
            if species_index is None:rates.append(temperature_rate())

        self.final_temperature = float(phase.T)
        if phase.T - self.T < 200.0:
            raise RuntimeError(
                "Ignition was not detected: the final temperature rise is less than 200 K."
            )

        if len(times) < 3:
            raise RuntimeError("Insufficient samples to resolve ignition delay.")
        gradient = np.asarray(rates) if species_index is None else np.gradient(concentrations, times)
        peak = int(np.argmax(gradient))
        if not np.all(np.isfinite(gradient)) or gradient[peak] <= 0 or peak in (0, len(times) - 1):
            raise RuntimeError("No ignition was detected before end_time.")
        # Fit the local maximum between adaptive samples instead of selecting a timestamp.
        scale = times[peak+1]-times[peak-1]
        offsets=(np.asarray(times[peak-1:peak+2])-times[peak])/scale
        a,b,_=np.polyfit(offsets,gradient[peak-1:peak+2],2)
        vertex=-b/(2*a) if a<0 else 0.
        return float(times[peak]+scale*vertex) if offsets[0]<=vertex<=offsets[-1] else float(times[peak])
