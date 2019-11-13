"""
Compute figures of merit (SNR, FPF, ...) for results and write output to CSV.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import time

import pandas as pd

from hsr4hci.utils.fits import read_fits
from hsr4hci.utils.evaluation import compute_figures_of_merit


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCOMPUTE FIGURES OF MERIT\n', flush=True)

    # -------------------------------------------------------------------------
    # Collect FITS files and run SNR computation for all of them
    # -------------------------------------------------------------------------

    # Get a list of all FITS files we want to process
    fits_files = [_ for _ in os.listdir('../03_results')
                  if _.endswith('.fits')]

    # Keep track of results for all FITS files (experiments)
    results = list()

    # Loop over all FITS files
    for fits_file in sorted(fits_files):

        print(f'Processing: {fits_file}', end=' ... ', flush=True)

        # Construct path and read in FITS file
        file_path = os.path.join('../03_results', fits_file)
        frame = read_fits(file_path=file_path)

        # Compute figures of merit for this frame
        result = \
            compute_figures_of_merit(frame=frame,
                                     position=(49.5, 55.5),
                                     aperture_size=(0.096 / 0.0271 / 2),
                                     ignore_neighbors=True,
                                     optimize='snr')

        # Add file name to the results and store it
        result['file_name'] = fits_file
        results.append(result)

        print('Done!', flush=True)

        # Create a pandas DataFrame from the results
        results = pd.DataFrame(results)

        # Save the results as a CSV file in the results folder
        file_path = os.path.join('..', '03_results', f'figures_of_merit.csv')
        results.to_csv()

        print('')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'This took {time.time() - script_start:.1f} seconds!\n')
