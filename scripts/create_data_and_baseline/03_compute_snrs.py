"""
This script takes the PCA-based signal estimates from 02_run_pca.py and
computes the SNR at the planet positions specified in the config.json.
To speed up the computation, process-based multiprocessing is used to
compute the SNR for different numbers of PCs in parallel.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from multiprocessing import Process, cpu_count
from typing import Dict, List, Tuple

import json
import os
import time

from astropy import units
from tqdm import tqdm

import numpy as np
import pandas as pd

from hsr4hci.utils.argparsing import get_base_directory
from hsr4hci.utils.evaluation import compute_optimized_snr
from hsr4hci.utils.fits import read_fits
from hsr4hci.utils.queue import Queue
from hsr4hci.utils.units import (
    convert_to_quantity,
    to_pixel,
    set_units_for_instrument,
)


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCOMPUTE SNR ON PCA-BASED SIGNAL ESTIMATES\n', flush=True)

    # -------------------------------------------------------------------------
    # Parse command line arguments and load config.json
    # -------------------------------------------------------------------------

    # Get base_directory from command line arguments
    base_dir = get_base_directory()

    # Construct (expected) path to config.json
    file_path = os.path.join(base_dir, 'config.json')

    # Read in the config file and parse it
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)

    # Now, apply unit conversions to astropy.units:
    # First, convert pixscale and lambda_over_d to astropy.units.Quantity. This
    # is a bit cumbersome, because in the meta data, we cannot use the usual
    # convention to specify units, as the meta data are also written to the HDF
    # file. Hence, we must hard-code the unit conventions here.
    config['metadata']['PIXSCALE'] = units.Quantity(
        config['metadata']['PIXSCALE'], 'arcsec / pixel'
    )
    config['metadata']['LAMBDA_OVER_D'] = units.Quantity(
        config['metadata']['LAMBDA_OVER_D'], 'arcsec'
    )

    # Use this to set up the instrument-specific conversion factors. We need
    # this here to that we can parse "lambda_over_d" as a unit in the config.
    set_units_for_instrument(
        pixscale=config['metadata']['PIXSCALE'],
        lambda_over_d=config['metadata']['LAMBDA_OVER_D'],
    )

    # Convert the relevant entries of the config to astropy.units.Quantity
    for key_tuple in [
        ('evaluation', 'aperture_radius'),
        ('evaluation', 'max_distance'),
    ]:
        config = convert_to_quantity(config, key_tuple)

    # -------------------------------------------------------------------------
    # Define some shortcuts
    # -------------------------------------------------------------------------

    # Define shortcuts for planet positions
    planet_positions: Dict[str, Tuple[float, float]] = dict()
    ignore_neighbors: Dict[str, int] = dict()
    for planet_key, options in config['evaluation']['planets'].items():
        planet_positions[planet_key] = (
            float(options['position'][0]),
            float(options['position'][1]),
        )
        ignore_neighbors[planet_key] = int(options['ignore_neighbors'])

    # Define shortcuts for SNR options
    snr_options = config['evaluation']['snr_options']
    aperture_radius = to_pixel(snr_options['aperture_radius'])
    max_distance = to_pixel(snr_options['max_distance'])
    method = snr_options['method']
    target = snr_options['target']
    time_limit = snr_options['time_limit']

    # Shortcuts to entries in the configuration
    min_n_components = config['pca']['min_n_components']
    max_n_components = config['pca']['max_n_components']
    n_processes = config['evaluation']['n_processes']

    # In case we do not explicitly limit the number of parallel processes, we
    # use the number of CPU cores in the current machine
    if n_processes is None:
        n_processes = cpu_count()

    # Construct numbers of principal components (and count them)
    pca_numbers = list(range(min_n_components, max_n_components + 1))
    n_signal_estimates = len(pca_numbers)

    # Other shortcuts
    baselines_dir = os.path.join(base_dir, 'pca_baselines')

    # -------------------------------------------------------------------------
    # Read in the residuals for each stacking factor and compute SNR
    # -------------------------------------------------------------------------

    # Run for each stacking factor
    for stacking_factor in config['stacking_factors']:

        print(f'Running for stacking factor {stacking_factor}:', flush=True)
        print(80 * '-', flush=True)

        # Construct path to result dir for this stacking factor
        result_dir = os.path.join(baselines_dir, f'stacked_{stacking_factor}')

        # Construct path to signal estimates and read the FITS files to numpy
        print('Reading in signal estimates...', end=' ', flush=True)
        file_path = os.path.join(result_dir, 'signal_estimates.fits')
        signal_estimates: np.ndarray = read_fits(file_path=file_path)
        print('Done!', flush=True)

        # Initialize dict with lists that will hold results for each planet
        results: Dict[str, list] = {_: list() for _ in planet_positions.keys()}

        # ---------------------------------------------------------------------
        # Loop over all planets and compute figures of merit
        # ---------------------------------------------------------------------

        for planet_key, planet_position in planet_positions.items():

            # Construct full name of the planet (target star + letter)
            planet_name = f'{config["metadata"]["TARGET_STAR"]} {planet_key}'

            print(f'Running SNR computation for planet {planet_name}:')

            # In the following, we use a custom parallelization approach to
            # compute the figures of merits for the different numbers of
            # principal components in parallel. We *cannot* simply use the
            # parallelization based on joblib.Parallel.delayed(), because the
            # compute_figures_of_merit() function internally uses a @timeout
            # decorator that relies on the signal library, which only works in
            # the main thread.

            # -----------------------------------------------------------------
            # Prepare queues and target function
            # -----------------------------------------------------------------

            # Initialize a queue for the inputs, i.e., for each number of PCs
            # one frame with the corresponding signal_estimate (as well as the
            # index, so that we can reconstruct the correct result order)
            input_queue = Queue()
            for index in range(n_signal_estimates):
                input_queue.put((index, signal_estimates[index]))

            # Initialize an output queue and a list for the results
            output_queue = Queue()
            results_list: List[Tuple[int, np.ndarray]] = []

            # Define a partial function application of compute_optimized_snr():
            # Fix all arguments except for the frame, and make sure the output
            # is placed into the shared output queue.
            def get_fom(index: int, frame: np.ndarray,) -> None:
                """
                Partial function application of compute_optimized_snr().

                Args:
                    index: Index of the current frame in the array of
                        signal estimates (essentially the number of PCs
                        that were used to create the `frame`).
                    frame: The frame (containing an estimate for the
                        planet signal) for which to compute the SNR.
                """

                result = compute_optimized_snr(
                    frame=frame,
                    position=planet_position,
                    aperture_radius=aperture_radius,
                    ignore_neighbors=ignore_neighbors[planet_key],
                    target=target,
                    method=method,
                    max_distance=max_distance,
                    time_limit=time_limit,
                )
                output_queue.put((index, result))

            # -----------------------------------------------------------------
            # Process signal_estimate frames in parallel
            # -----------------------------------------------------------------

            # Define a context for the progress bar
            with tqdm(ncols=80, total=n_signal_estimates) as progressbar:

                # Initialize a list for the processes that we start
                processes: List[Process] = []

                # Keep going while we do not yet have results for all frames
                while not len(results_list) == n_signal_estimates:

                    # Remove processes that have terminated already from our
                    # list of (running) processes
                    for process in processes:
                        if not process.is_alive():
                            processes.remove(process)

                    # Start new processes until the input_queue is empty, or
                    # until we have reached the maximum number of processes
                    while (
                        not input_queue.empty()
                        and len(processes) < n_processes
                    ):

                        # Get new frame from queue and define new process
                        index, frame = input_queue.get()
                        process = Process(target=get_fom, args=(index, frame))

                        # Add to list of processes and start process
                        processes.append(process)
                        process.start()

                    # Move results from output_queue to results_list:
                    # Without this part, the output_queue blocks the worker
                    # processes so that they will not terminate.
                    while not output_queue.empty():
                        results_list.append(output_queue.get())

                    # Update the progress bar based on the number of results
                    progressbar.update(len(results_list) - progressbar.n)

            # Sort the results_list by the indices. This is needed because the
            # parallelization does not necessarily preserve the frame order.
            results_list = sorted(results_list, key=lambda x: int(x[0]))

            # Separate indices and figures of merit
            _, figures_of_merit = zip(*results_list)

            # Finally, store the figures_of_merit for this planet
            results[planet_key] = figures_of_merit

        # ---------------------------------------------------------------------
        # Compute FOM into pandas data frame and save them as a CSV file
        # ---------------------------------------------------------------------

        # Get planet and FOM names
        planet_names = planet_positions.keys()
        fom_names = next(iter(results.values()))[0].keys()

        # Create new multi-index from the Cartesian product of the planet
        # names and the names for the figures of merits
        multi_index = pd.MultiIndex.from_product(
            iterables=[planet_names, fom_names],
            names=['planet', 'figures_of_merit'],
        )

        # Create an additional index that associates every row in the data
        # frame with the corresponding number of principal components
        row_index = pd.MultiIndex.from_arrays(
            arrays=[
                range(n_signal_estimates),
                pca_numbers[:n_signal_estimates],
            ],
            names=[None, 'n_principal_components'],
        )

        # Create an empty data frame using these two multi-indices
        dataframe = pd.DataFrame(index=row_index, columns=multi_index)

        # Populate the data frame planet by planet
        for planet_key in planet_names:

            # Convert the results for a given planet from a list of dicts to
            # a dict of lists, which we can use to create a new data frame
            planet_dict = pd.DataFrame(results[planet_key]).to_dict('list')

            # Add values for each figure of merit
            for fom_name in planet_dict.keys():
                dataframe[(planet_key, fom_name)] = planet_dict[fom_name]

        # Save the figures of merit as a CSV file
        print('Saving results as CSV file...', end=' ', flush=True)
        file_path = os.path.join(result_dir, 'figures_of_merit.csv')
        dataframe.to_csv(path_or_buf=file_path, sep='\t', float_format='%.3f')
        print('Done!\n')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
