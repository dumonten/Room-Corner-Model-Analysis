import numpy as np
from numpy.typing import NDArray
from typing import Any


def mean_absolute_deviation(y: NDArray[Any], comp: Any) -> float:
    """
    Calculate the mean absolute deviation between two arrays.

    Parameters
    ----------
    y : NDArray[Any]
        The first array of values.
    comp : Any
        The second array of values or value to compare against.

    Returns
    -------
    float
        The mean absolute deviation between the two arrays.
    """
    return np.mean(np.abs(y - comp))


def root_mean_square_deviation(y: NDArray[Any], comp: Any) -> float:
    """
    Calculate the root mean square deviation between two arrays.

    Parameters
    ----------
    y : NDArray[Any]
        The first array of values.
    comp : Any
        The second array of values or value to compare against.

    Returns
    -------
    float
        The root mean square deviation between the two arrays.
    """
    return np.sqrt(np.mean(y - comp) ** 2)


def mean_absolute_percentage_deviation(y: NDArray[Any], comp: NDArray[Any]) -> float:
    """
    Calculate the mean absolute percentage deviation between two arrays.

    Parameters
    ----------
    y : NDArray[Any]
        The first array of values.
    comp : Any
        The second array of values or value to compare against.

    Returns
    -------
    float
        The mean absolute percentage deviation between the two arrays.
    """
    return np.mean(np.abs((y - comp) / y)) * 100


def emax(y: NDArray[Any], comp: NDArray[Any]) -> float:
    """
    Calculate the maximum absolute deviation between two arrays.

    Parameters
    ----------
    y : NDArray[Any]
        The first array of values.
    comp : Any
        The second array of values or value to compare against.

    Returns
    -------
    float
        The maximum absolute deviation between the two arrays.
    """
    return np.max(np.abs(y - comp))
