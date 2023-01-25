
def _generate_positions(G, num_vortices: int, threshold: float):
    """Generates and returns a list of positions that are separated by at least
    `threshold`.
    """
    print(f"Attempting to find {num_vortices} positions...")
    max_iter = 10000
    vortex_positions = []

    iterations = 0
    while len(vortex_positions) < num_vortices:
        if iterations > max_iter:
            print(
                f"WARNING: Number of iterations exceeded maximum, "
                f"returning with only {len(vortex_positions)} positions\n"
            )
            return vortex_positions

        position = G.params.xp.random.uniform(
            -G.params.Lx / 2, G.params.Lx / 2
        ), G.params.xp.random.uniform(-G.params.Ly / 2, G.params.Ly / 2)

        if _position_sufficiently_far(position, vortex_positions, threshold):
            vortex_positions.append(position)

        iterations += 1

    print(f"Successfully found {num_vortices} positions in {iterations} iterations!")
    return iter(vortex_positions)


def _position_sufficiently_far(
    position: tuple, accepted_positions: list[tuple], threshold: float
) -> bool:
    """Tests that the given `position` is at least `threshold` away from all the positions
    currently in `accepted_positions`.
    """
    # Special case where accepted_positions is empty
    if not accepted_positions:
        return True

    for accepted_pos in accepted_positions:
        if abs(position[0] - accepted_pos[0]) > threshold:
            if abs(position[1] - accepted_pos[1]) > threshold:
                return True
    return False


def _heaviside(G, array):
    """Computes the heaviside function on a given array and returns the result."""
    return G.params.xp.where(array < 0, G.params.xp.zeros(array.shape), G.params.xp.ones(array.shape))


def vortex_phase_profile(G, num_vortices: int, threshold: float):
    """Constructs a 2D phase profile consisting of 2pi phase windings.
    This phase can be applied to a wavefunction to generate different types of vortices.
    :param grid: The 2D grid of the system.
    :type grid: :class:`Grid`
    :param num_vortices: The total number of vortices to be included in the phase profile.
    :type num_vortices: int
    :param threshold: The minimum distance allowed between any two vortices.
    :type threshold: float
    """
    vortex_positions_iter = _generate_positions(G, num_vortices, threshold)

    phase = G.params.xp.zeros((G.params.Nx, G.params.Ny), dtype="float32")

    for _ in range(num_vortices // 2):
        phase_temp = G.params.xp.zeros((G.params.Nx, G.params.Ny), dtype="float32")
        x_pos_minus, y_pos_minus = next(
            vortex_positions_iter
        )  # Negative circulation vortex
        x_pos_plus, y_pos_plus = next(
            vortex_positions_iter
        )  # Positive circulation vortex

        # Aux variables
        y_minus = 2 * G.params.xp.pi / G.params.Ly * (G.grid.yy - y_pos_minus)
        x_minus = 2 * G.params.xp.pi / G.params.Lx * (G.grid.xx - x_pos_minus)
        y_plus = 2 * G.params.xp.pi / G.params.Ly * (G.grid.yy - y_pos_plus)
        x_plus = 2 * G.params.xp.pi / G.params.Lx * (G.grid.xx - x_pos_plus)

        heaviside_x_plus = _heaviside(G, x_plus)
        heaviside_x_minus = _heaviside(G, x_minus)

        for nn in G.params.xp.arange(-5, 6):
            phase_temp += (
                G.params.xp.arctan(
                    G.params.xp.tanh((y_minus + 2 * G.params.xp.pi * nn) / 2)
                    * G.params.xp.tan((x_minus - G.params.xp.pi) / 2)
                )
                - G.params.xp.arctan(
                    G.params.xp.tanh((y_plus + 2 * G.params.xp.pi * nn) / 2)
                    * G.params.xp.tan((x_plus - G.params.xp.pi) / 2)
                )
                + G.params.xp.pi * (heaviside_x_plus - heaviside_x_minus)
            )
        phase_temp -= (
            2
            * G.params.xp.pi
            * (G.grid.yy - G.grid.yy.min())
            * (x_pos_plus - x_pos_minus)
            / (G.params.Ly * G.params.Lx)
        )
        phase += phase_temp

    return phase