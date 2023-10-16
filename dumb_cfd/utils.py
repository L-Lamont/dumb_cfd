import numpy as np


def update_boundary(
            working_array: np.ndarray,
            boundary_size: int,
            periodic: bool,
            constant_boundary_value: float = 0.0
        ):
    """
    Updates boundary conditions based on parameters

    Parameters
    ----------
    working_array: np.ndarray
        The array whose boundary conditions need to be updated.
    boundary_size: int
        The width of the boundary being updated
    periodic: bool
        A flag indicating whether the boundary update should be periodic or
        constant value.
    constant_boundary_value: float
        If using a constant value the value the boundary will be updated to.
        Defaults to 0.0.
    """

    ndim = working_array.ndim

    if ndim == 1 and not periodic:
        constant_1d(working_array, boundary_size, constant_boundary_value)

    if ndim == 1 and periodic:
        periodic_1d(working_array, boundary_size)

    if ndim == 2 and not periodic:
        constant_2d(working_array, boundary_size, constant_boundary_value)

    if ndim == 2 and periodic:
        periodic_2d(working_array, boundary_size)


def constant_1d(
        working_array: np.ndarray,
        boundary_size: int,
        constant_boundary_value: float):
    """
    Set a constant boundary condition for a 1D array.

    Parameters
    ----------
    working_array : ndarray
        The 1D array for which the boundary condition is applied.
    boundary_size : int
        The size of the boundary on each side of the array.
    constant_boundary_value : float or int
        The constant value to be applied at the boundaries.

    Notes
    -----
    This function modifies `working_array` in place, applying a constant
    value to the 'boundary_size' elements at both ends of the array.
    """

    for i in range(boundary_size):
        working_array[i] = constant_boundary_value
        working_array[-(i+1)] = constant_boundary_value


def constant_2d(
        working_array: np.ndarray,
        boundary_size: int,
        constant_boundary_value: float):
    """
    Set a constant boundary condition for a 2D array.

    Parameters
    ----------
    working_array : ndarray
        The 2D array for which the boundary condition is applied.
    boundary_size : int
        The size of the boundary along each axis of the array.
    constant_boundary_value : float or int
        The constant value to be applied at the boundaries.

    Notes
    -----
    This function modifies `working_array` in place, applying a constant
    value to the 'boundary_size' elements along all edges of the 2D array.
    """

    for i in range(boundary_size):
        working_array[i:, :] = constant_boundary_value
        working_array[:-(i+1), :] = constant_boundary_value

        working_array[:, i:] = constant_boundary_value
        working_array[:, :-(i+1)] = constant_boundary_value


def periodic_1d(
        working_array: np.ndarray,
        boundary_size: int):
    """
    Set a periodic boundary condition for a 1D array.

    Parameters
    ----------
    working_array : ndarray
        The 1D array for which the boundary condition is applied.
    boundary_size : int
        The size of the boundary on each side of the array.

    Notes
    -----
    This function modifies `working_array` in place, wrapping the
    'boundary_size' elements at both ends of the array to create a periodic
    boundary condition.
    """

    for i in range(boundary_size):
        working_array[i] = working_array[i - 2 * boundary_size]
        working_array[-(i + 1)] = working_array[2 * boundary_size - (i + 1)]


def periodic_2d(
        working_array: np.ndarray,
        boundary_size: int):
    """
    Set a periodic boundary condition for a 2D array.

    Parameters
    ----------
    working_array : ndarray
        The 2D array for which the boundary condition is applied.
    boundary_size : int
        The size of the boundary along each axis of the array.

    Notes
    -----
    This function modifies `working_array` in place, wrapping the
    'boundary_size' elements along all edges of the 2D array to create a
    periodic boundary condition.
    """

    shape = working_array.shape

    for xr in [-1, 0, 1]:
        w_x = shape[0] - 2 * boundary_size
        x_0 = boundary_size + max(xr * w_x, -boundary_size)
        x_1 = boundary_size + min((xr + 1) * w_x, w_x + boundary_size)

        for yr in [-1, 0, 1]:
            w_y = shape[1] - 2 * boundary_size
            y_0 = boundary_size + max(yr * w_y, -boundary_size)
            y_1 = boundary_size + min((yr + 1) * w_y, w_y + boundary_size)

            working_array[x_0:x_1, y_0: y_1] = \
                working_array[x_0 - xr * w_x:x_1 - xr * w_x,
                              y_0 - yr * w_y:y_1 - yr * w_y]
