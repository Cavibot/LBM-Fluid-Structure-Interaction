# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Differentiable CompositeSimulation: saves all states for each step (for autodiff/optimization)."""

from wanphys.core import CompositeSimulation, DomainState


class DiffCompositeSimulation(CompositeSimulation):

    """
    CompositeSimulation variant for differentiable physics:
    - Each step, saves a deep copy of all domain states (state_in) to self.state_history.
    - No state is overwritten; all time steps are accessible for autodiff, loss, etc.
    """

    def create_out_states(self) -> None:
        """Allocate new out states for each domain to avoid in/out state aliasing."""
        states = {}
        for name, domain in self._domains.items():
            domain.create_state()
            states[name] = getattr(domain, "_state_out", None)
        return states

    def __init__(self):
        super().__init__()
        self._domains: dict[str, object] = {}
        self._states_in: dict[str, DomainState] = {}
        self._states_out: dict[str, DomainState] = {}
        self._external_forces = []
        self._couplings = []
        self.state_history: list[dict[str, DomainState]] = []  # List of {domain_name: state}

    def add_domain(self, domain) -> None:
        """Register a domain and allocate its initial state buffers."""
        self._domains[domain.name] = domain
        self._states_in[domain.name] = domain.create_state()
        self._states_out[domain.name] = getattr(domain, "_state_out", self._states_in[domain.name])

    def get_state(self, domain_name: str) -> DomainState:
        """Return the active state for a registered domain."""
        return self._states_in[domain_name]

    def step(self, dt: float) -> None:
        # Create new out states each step to avoid in/out aliasing.
        self._states_out = self.create_out_states()

        # 1. pre-step hooks
        for name, domain in self._domains.items():
            state = self._states_in[name]
            domain.pre_step(state, dt)

        # 2. Apply external forces (viewer, user input, etc.)
        for external_force_fn in self._external_forces:
            external_force_fn(self._states_in, dt)

        # 3. Apply couplings
        for coupling in self._couplings:
            coupling.apply(self._domains, self._states_in, dt)

        # 4. Step each domain
        for name, domain in self._domains.items():
            state_in = self._states_in[name]
            state_out = self._states_out[name]
            domain.step(state_in, state_out, dt)

        self.state_history.append(self._states_out)
        self._states_in = self._states_out  # Update in-states for next step
        should_reset = False
        for name, domain in self._domains.items():
            state = self._states_out[name]
            domain.post_step(state, dt)
            if hasattr(domain, "consume_reset_request") and domain.consume_reset_request():
                should_reset = True

        if should_reset:
            # Reset clears _states and state_history; also resets domain runtime counters.
            self.reset()
            return

        self._time += dt


    def get_state_at(self, step_idx: int, domain_name: str) -> DomainState:
        """Get the state of a domain at a specific step (0-based)."""
        return self.state_history[step_idx][domain_name]

    def reset(self) -> None:
        super().reset()
        self.state_history.clear()
        for name, domain in self._domains.items():
            if hasattr(domain, "on_simulation_reset"):
                domain.on_simulation_reset()
            self._states_in[name] = getattr(domain, "_state_in", self._states_in[name])
            self._states_out[name] = getattr(domain, "_state_out", self._states_out[name])
            if hasattr(domain, "on_reset_states"):
                domain.on_reset_states(self._states_in[name], self._states_out[name])