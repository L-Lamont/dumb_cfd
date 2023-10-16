import numpy as np

from utils import update_boundary


def convection(
            initial_state: np.ndarray,
            num_timesteps: int,
            timestep_size: float,
            step_length: tuple,
            periodic: bool = False,
            speed: tuple = None,
            constant_boundary_value: float = 0.0
        ) -> np.ndarray:
    """
    Computes convection equation for domain and given parameters

    Parameters
    ----------
    initial_state : np.ndarray
        An array of floats representing the initial state of the system. This
        array should contain the discretized values of the field variable at
        each spatial step.
    num_timesteps : int
        The number of time steps to simulate.
    timestep_size : float
        The amount of time each time step represents.
    step_length : tuple
        Tuple containing the distance between grid points for each dimension.
    periodic : bool, optional
        A flag indicating whether the simulation uses periodic boundary
        conditions. If False, the boundary conditions are assumed to be fixed.
        Default is False.
    speed : tuple, optional
        Tuple containing the constant convection speed for each dimension.
        Only used for linear convection
    constant_boundary_value: float, optional
        The constant value of the boundary. Used when periodic == False.
        Default is 0.0

    Returns
    -------
    np.ndarray
        A numpy array representing the state of the system at the end of the
        simulation. The array will have the same shape as `initial_state`.
    """
    ndim = initial_state.ndim

    if ndim == 1 and speed:
        return convection_linear_1d(
            initial_state=initial_state,
            num_timesteps=num_timesteps,
            timestep_size=timestep_size,
            step_length=step_length,
            periodic=periodic,
            speed=speed,
            constant_boundary_value=constant_boundary_value
        )

    if ndim == 1 and not speed:
        return convection_nonlinear_1d(
            initial_state=initial_state,
            num_timesteps=num_timesteps,
            timestep_size=timestep_size,
            step_length=step_length,
            periodic=periodic,
            constant_boundary_value=constant_boundary_value
        )

    if ndim == 2 and speed:
        return convection_linear_2d(
            initial_state=initial_state,
            num_timesteps=num_timesteps,
            timestep_size=timestep_size,
            step_length=step_length,
            periodic=periodic,
            speed=speed,
            constant_boundary_value=constant_boundary_value
        )

    if ndim == 2 and not speed:
        return convection_nonlinear_2d(
            initial_state=initial_state,
            num_timesteps=num_timesteps,
            timestep_size=timestep_size,
            step_length=step_length,
            periodic=periodic,
            constant_boundary_value=constant_boundary_value
        )

    raise RuntimeError("Arguments not set correctly please check docs")


def convection_linear_1d(
            initial_state: np.ndarray,
            num_timesteps: int,
            timestep_size: float,
            step_length: tuple,
            periodic: bool,
            speed: tuple,
            constant_boundary_value: float = 0.0
        ) -> np.ndarray:
    """
    Computes linear convection equation for the 1D domain and given parameters

    Parameters
    ----------
    initial_state : np.ndarray
        1D array representing the initial state
    num_timesteps : int
        Number of timesteps to simulate.
    timestep_size : float
        The amount of simulated time each timestep represents.
    step_length : tuple
        Tuple containing the distance between grid points for each dimension.
    speed : tuple
        Tuple containing the constant convection speed for each dimension.
    periodic : bool
        If True, the wave is treated as periodic. If False, the wave is
        treated as bounded with a constant value of 0.
    constant_boundary_value: float, optional
        The constant value of the boundary. Used when periodic == False.
        Default is 0.0

    Returns
    -------
    np.ndarray
        A numpy array representing the state of the system at the end of the
        simulation. The array will have the same shape as `initial_state`.
    """

    step_length_x, = step_length
    speed_x, = speed
    boundary_size = 1

    state = np.pad(initial_state, (boundary_size, boundary_size))

    for _ in range(num_timesteps):
        update_boundary(state,
                        boundary_size=boundary_size,
                        periodic=periodic)

        state[1:-1] -= \
            speed_x * timestep_size / step_length_x * \
            (state[1:-1] - state[:-2])

    return state[boundary_size:-boundary_size]


def convection_nonlinear_1d(
            initial_state: np.ndarray,
            num_timesteps: int,
            timestep_size: float,
            step_length: tuple,
            periodic: bool
        ) -> np.ndarray:
    """
    Computes nonlinear convection equation for a 1D domain with the given
    parameters.

    Parameters
    ----------
    initial_state : np.ndarray
        1D array representing the initial state of the system.
    num_timesteps : int
        Number of timesteps to simulate.
    timestep_size : float
        The amount of simulated time each timestep represents.
    step_length : tuple
        Tuple containing the distance between grid points for each dimension.
    periodic : bool
        If True, the domain is treated as periodic. If False, the domain is
        treated as bounded with a constant value of 0.
    constant_boundary_value: float, optional
        The constant value of the boundary. Used when periodic == False.
        Default is 0.0

    Returns
    -------
    np.ndarray
        A numpy array representing the state of the system at the end of the
        simulation. The array will have the same shape as `initial_state`.
    """
    raise NotImplementedError


def convection_linear_2d(
            initial_state: np.ndarray,
            num_timesteps: int,
            timestep_size: float,
            step_length: tuple,
            periodic: bool,
            speed: tuple,
            constant_boundary_value: float = 0.0
        ) -> np.ndarray:
    """
    Computes linear convection equation for a 2D domain with the given
    parameters.

    Parameters
    ----------
    initial_state : np.ndarray
        2D array representing the initial state of the system.
    num_timesteps : int
        Number of timesteps to simulate.
    timestep_size : float
        The amount of simulated time each timestep represents.
    step_length : tuple
        Tuple containing the distance between grid points for each dimension.
    periodic : bool
        If True, the domain is treated as periodic. If False, the domain is
        treated as bounded with a constant value of 0. Defaults to False.
    speed : tuple
        Tuple containing the constant convection speed for each dimension.
    constant_boundary_value: float, optional
        The constant value of the boundary. Used when periodic == False.
        Default is 0.0

    Returns
    -------
    np.ndarray
        A numpy array representing the state of the system at the end of the
        simulation. The array will have the same shape as `initial_state`.
    """
    raise NotImplementedError


def convection_nonlinear_2d(
            initial_state: np.ndarray,
            num_timesteps: int,
            timestep_size: float,
            step_length: tuple,
            periodic: bool,
            constant_boundary_value: float = 0.0
        ) -> np.ndarray:
    """
    Computes nonlinear convection equation for a 2D domain with the given
    parameters.

    Parameters
    ----------
    initial_state : np.ndarray
        2D array representing the initial state of the system.
    num_timesteps : int
        Number of timesteps to simulate.
    timestep_size : float
        The amount of simulated time each timestep represents.
    step_length : tuple
        Tuple containing the distance between grid points for each dimension.
    periodic : bool
        If True, the domain is treated as periodic. If False, the domain is
        treated as bounded with a constant value of 0.
    constant_boundary_value: float, optional
        The constant value of the boundary. Used when periodic == False.
        Default is 0.0

    Returns
    -------
    np.ndarray
        A numpy array representing the state of the system at the end of the
        simulation. The array will have the same shape as `initial_state`.
    """
    raise NotImplementedError
