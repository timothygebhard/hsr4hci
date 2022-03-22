"""
Compute the correlation coefficients between every position (x, y)
with every other position (x', y').
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import time

from hsr4hci.config import get_experiments_dir
from hsr4hci.data import load_dataset
from hsr4hci.fits import save_fits


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def correlation_map(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Auxiliary function to compute the correlation map efficiently.
    """

    epsilon = np.finfo(float).eps

    mu_x = x.mean(axis=1)
    mu_y = y.mean(axis=1)
    n = x.shape[1]

    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)

    cov = np.dot(x, y.T) - n * np.dot(mu_x[:, np.newaxis], mu_y[np.newaxis, :])

    correlation_map = cov / (
        np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :]) + epsilon
    )

    correlation_map = np.clip(correlation_map, -1, 1)

    return np.asarray(correlation_map)


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCOMPUTE CORRELATION COEFFICIENT MAPS\n')

    # -------------------------------------------------------------------------
    # Compute full 4D correlation map for two example data sets
    # -------------------------------------------------------------------------

    # Ensure that the directory for the FITS files exists
    fits_dir = (
        get_experiments_dir()
        / 'appendix'
        / 'A.2_correlation-coefficient-maps'
        / 'fits'
    )
    fits_dir.mkdir(exist_ok=True)

    # Loop over different data sets
    for dataset in ('beta_pictoris__mp', 'hr_8799__lp'):

        start_time = time.time()
        print(f'Running for {dataset}...', end=' ', flush=True)

        # Load data set (and crop to some reasonable size)
        stack, parang, psf_template, obs_con, metadata = load_dataset(
            name_or_path=dataset,
            frame_size=(51, 51),
        )
        n_frames, x_size, y_size = stack.shape

        # Reshape and flatten the stack for use with correlation_map()
        flat_stack = np.transpose(stack, (1, 2, 0)).reshape(-1, n_frames)

        # Compute the correlations (along the temporal axis) between every
        # position (x, y) and every other position (x', y'); the result is
        # a 4D array of shape (x_size, y_size, x_size, y_size).
        correlations = correlation_map(flat_stack, flat_stack)
        correlations = correlations.reshape((x_size, y_size, x_size, y_size))

        # Save correlations to FITS file
        file_path = fits_dir / f'{dataset}.fits'
        save_fits(array=correlations, file_path=file_path)

        print(f'Done! ({time.time() - start_time:.1f} seconds)', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
