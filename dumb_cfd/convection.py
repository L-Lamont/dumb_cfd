import numpy as np


def convection(
            initial_state: np.ndarray,
            num_timesteps: int,
            timestep_size: float,
            step_length: np.ndarray,
            periodic: bool = False,
            speed: np.ndarray = None
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
    step_length : np.ndarray
        The spatial step length for each dimension.
    periodic : bool, optional
        A flag indicating whether the simulation uses periodic boundary
        conditions. If False, the boundary conditions are assumed to be fixed.
        Default is False.
    speed : np.ndarray, optional
        The constant convection speed for each dimension.
        Only used for linear convection

    Returns
    -------
    np.ndarray
        A numpy array representing the state of the system at the end of the
        simulation. The array will have the same shape as `initial_state`.
    """
    ndims = initial_state.ndims

    if ndims == 1 and speed:
        return convection_linear_1d(
            initial_state=initial_state,
            num_timesteps=num_timesteps,
            timestep_size=timestep_size,
            step_length=step_length,
            periodic=periodic,
            speed=speed
        )

    if ndims == 1 and not speed:
        return convection_nonlinear_1d(
            initial_state=initial_state,
            num_timesteps=num_timesteps,
            timestep_size=timestep_size,
            step_length=step_length,
            periodic=periodic
        )

    if ndims == 2 and speed:
        return convection_linear_2d(
            initial_state=initial_state,
            num_timesteps=num_timesteps,
            timestep_size=timestep_size,
            step_length=step_length,
            periodic=periodic,
            speed=speed
        )

    if ndims == 2 and not speed:
        return convection_nonlinear_2d(
            initial_state=initial_state,
            num_timesteps=num_timesteps,
            timestep_size=timestep_size,
            step_length=step_length,
            periodic=periodic
        )

    raise RuntimeError("Arguments not set correctly please check docs")


def convection_linear_1d(
            initial_state: np.ndarray,
            num_timesteps: int,
            timestep_size: float,
            step_length: np.ndarray,
            periodic: bool,
            speed: np.ndarray
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
    step_length : float
        Length 1 array the value representing the distance between grid points
    speed : np.ndarray
        Length 1 array the value representing the constant convection speed
    periodic : bool
        If True, the wave is treated as periodic. If False, the wave is
        treated as bounded with a constant value of 0.

    Returns
    -------
    np.ndarray
        A numpy array representing the state of the system at the end of the
        simulation. The array will have the same shape as `initial_state`.
    """
    raise NotImplementedError


def convection_nonlinear_1d(
            initial_state: np.ndarray,
            num_timesteps: int,
            timestep_size: float,
            step_length: np.ndarray,
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
    step_length : np.ndarray
        Length 1 array the value representing the distance between grid points.
    periodic : bool
        If True, the domain is treated as periodic. If False, the domain is
        treated as bounded with a constant value of 0.

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
            step_length: np.ndarray,
            periodic: bool,
            speed: np.ndarray
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
    step_length : np.ndarray
        Length 2 array the values representing the distance between grid
        points for each dimensions.
    periodic : bool
        If True, the domain is treated as periodic. If False, the domain is
        treated as bounded with a constant value of 0. Defaults to False.
    speed : np.ndarray
        Length 2 array the values representing the constant convection speed
        for each dimensions.

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
            step_length: np.ndarray,
            periodic: bool
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
    step_length : np.ndarray
        Length 2 array representing the distance between grid points for each
        dimension.
    periodic : bool
        If True, the domain is treated as periodic. If False, the domain is
        treated as bounded with a constant value of 0.

    Returns
    -------
    np.ndarray
        A numpy array representing the state of the system at the end of the
        simulation. The array will have the same shape as `initial_state`.
    """
    raise NotImplementedError
