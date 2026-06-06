"""Route table for the flat refactor (change-2, Shape B).

``effects`` holds the four effect routes (thin, byte-identical wrappers over the
existing run_loop effect functions); ``selector`` maps a ``DeciderAction`` to a
route id; ``builder`` wires them into a ``select_router.Router``. All dormant
behind ``ENGINE_USE_ROUTER`` (default OFF) until change-3.
"""
