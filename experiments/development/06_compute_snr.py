"""
Compute signal-to-noise ratios and create result plots.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from multiprocessing import Process, set_start_method
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import os
import time

from astropy.units import Quantity
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hsr4hci.utils.config import load_config, load_dataset_config
from hsr4hci.utils.evaluation import compute_optimized_snr
from hsr4hci.utils.fits import read_fits
from hsr4hci.utils.plotting import adjust_luminosity, plot_frame
from hsr4hci.utils.queue import Queue, get_available_cpu_count
from hsr4hci.utils.units import set_units_for_instrument


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_fom(
    index: int,
    frame: np.ndarray,
    position: Tuple[float, float],
    aperture_radius: float,
    target: str,
    method: str,
    max_distance: float,
    grid_size: int,
    time_limit: int,
    output_queue: 'Queue',
) -> None:
    """
    This is a thin wrapper around the `compute_optimized_snr()` function
    to enable it to be used in parallel with multiple worker processes.
    The point of the function is basically to make sure that the outputs
    of `compute_optimized_snr()` end up in the correct results queue.

    Args:
        index: Index of the current frame in the array of signal
            estimates (essentially the number of PCs that were used to
            create the `frame`).
        frame: The frame (containing an estimate for the planet signal)
            for which to compute the SNR.
        position: (Starting) position for the SNR evaluation
        aperture_radius: Aperture radius (in pixels).
        target: Target for the optimization (usually "signal_flux").
        method: Optimization method (usually "brute" for brute force
            optimization on a grid).
        max_distance: Maximum distance (in pixels) between the optimized
            planet positions and the starting position.
        grid_size: When using method == 'brute', this parameter defines
            the size of the grid.
        time_limit: Time limit (in seconds) for the SNR computation.
        output_queue: The queue (in shared memory!) to which this
            function when it is run by a worker process will deliver
            its results.
    """

    result: Dict[str, Any] = compute_optimized_snr(
        frame=frame,
        position=position,
        aperture_radius=aperture_radius,
        ignore_neighbors=1,  # TODO: Can we really fix this to 1?
        target=target,
        method=method,
        max_distance=max_distance,
        grid_size=grid_size,
        time_limit=time_limit,
    )
    output_queue.put((index, result))


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCOMPUTE SIGNAL-TO-NOISE RATIO (AND CREATE PLOTS)\n', flush=True)

    # -------------------------------------------------------------------------
    # Load experiment configuration and data
    # -------------------------------------------------------------------------

    # Define paths for experiment folder and results folder
    experiment_dir = Path(os.path.realpath(__file__)).parent
    results_dir = experiment_dir / 'results'
    masks_dir = results_dir / 'masks_and_thresholds'

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    file_path = experiment_dir / 'config.json'
    config = load_config(file_path)
    metric_function = config['signal_masking']['metric_function']
    print('Done!', flush=True)

    # Load signal estimates
    print('Loading signal estimates...', end=' ', flush=True)
    file_path = results_dir / f'signal_estimates__{metric_function}.fits'
    signal_estimates = read_fits(file_path)
    print('Done!', flush=True)

    # Load thresholds
    print('Loading thresholds...', end=' ', flush=True)
    file_path = masks_dir / f'thresholds__{metric_function}.csv'
    thresholds = np.genfromtxt(file_path, delimiter=',')
    print('Done!', flush=True)

    # Construct path to original dataset configuration and load it
    target_name = config['dataset']['target_name']
    filter_name = config['dataset']['filter_name']
    dataset_config = load_dataset_config(
        target_name=target_name, filter_name=filter_name
    )

    # -------------------------------------------------------------------------
    # Define various useful shortcuts
    # -------------------------------------------------------------------------

    # Values in experiment configuration
    n_test_positions = config['consistency_checks']['n_test_positions']
    metric_function = config['signal_masking']['metric_function']

    # Values from the original data set configuration
    metadata = dataset_config['metadata']
    pixscale = float(metadata['PIXSCALE'])
    lambda_over_d = float(metadata['LAMBDA_OVER_D'])
    planets = dataset_config['evaluation']['planets']
    snr_options = dataset_config['evaluation']['snr_options']

    # Define shortcuts for planet positions
    planet_positions: Dict[str, Tuple[float, float]] = dict()
    for planet_key, options in dataset_config['evaluation']['planets'].items():
        planet_positions[planet_key] = (
            float(options['position'][0]),
            float(options['position'][1]),
        )
    planet_keys = sorted(list(planet_positions.keys()))

    # Activate unit conversions for this instrument
    set_units_for_instrument(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
        verbose=False,
    )

    # Parse Quantities to pixel values and define shortcuts to SNR options
    for _ in ('aperture_radius', 'max_distance'):
        snr_options[_] = Quantity(*snr_options[_])
        snr_options[_] = snr_options[_].to('pixel').value
    aperture_radius = snr_options['aperture_radius']
    max_distance = snr_options['max_distance']
    method = snr_options['method']
    target = snr_options['target']
    grid_size = snr_options['grid_size']
    time_limit = snr_options['time_limit']

    # Other shortcuts
    n_signal_estimates = len(signal_estimates)

    # -------------------------------------------------------------------------
    # Loop over planets and compute SNR for each signal_estimate
    # -------------------------------------------------------------------------

    # Ensure that the start method for new processes is 'fork'.
    # Python 3.8 changed the default from 'fork' to 'spawn', because the
    # former is considered unsafe. However, when the worker processes are
    # spawned rather than forked, it seems that they can no longer access
    # the (shared) output queue correctly. Forcing the start method to be
    # 'fork' again seems to be the simplest workaround for now.
    # FIXME: Find a less hacky solution to this problem!
    set_start_method('fork')

    # Get the number of available CPU cores which defines the number of
    # concurrent processes we can use for the evaluation
    n_processes = get_available_cpu_count()

    # Initialize dict with lists that will hold results for each planet
    results: Dict[str, list] = {_: list() for _ in planet_positions.keys()}

    # Loop over all planets and compute figures of merit
    for planet_key, planet_position in planet_positions.items():

        # Construct full name of the planet (target star + letter)
        planet_name = f'{metadata["TARGET_STAR"]} {planet_key}'

        print(f'\nRunning SNR computation for planet {planet_name}:')

        # In the following, we use a custom parallelization approach to
        # compute the figures of merits for the different numbers of
        # principal components in parallel. We *cannot* simply use the
        # parallelization based on joblib.Parallel.delayed(), because the
        # compute_figures_of_merit() function internally uses a @timeout
        # decorator that relies on the signal library, which only works in
        # the main thread.

        # ---------------------------------------------------------------------
        # Prepare queues and target function
        # ---------------------------------------------------------------------

        # Initialize a queue for the inputs, i.e., for each number of PCs
        # one frame with the corresponding signal_estimate (as well as the
        # index, so that we can reconstruct the correct result order)
        input_queue = Queue()
        for index, signal_estimate in enumerate(signal_estimates):
            input_queue.put((index, signal_estimate))

        # Initialize an output queue and a list for the results
        output_queue = Queue()
        results_list: List[Tuple[int, Dict[str, Any]]] = []

        # ---------------------------------------------------------------------
        # Process signal_estimate frames in parallel
        # ---------------------------------------------------------------------

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

                # Start new processes until the input_queue is empty, or until
                # we have reached the maximum number of processes
                while input_queue.qsize() > 0 and len(processes) < n_processes:

                    # Get new frame from queue and define new process
                    index, frame = input_queue.get()
                    process = Process(
                        target=get_fom,
                        kwargs=dict(
                            index=index,
                            frame=frame,
                            position=planet_position,
                            aperture_radius=aperture_radius,
                            target=target,
                            method=method,
                            max_distance=max_distance,
                            grid_size=grid_size,
                            time_limit=time_limit,
                            output_queue=output_queue,
                        ),
                    )

                    # Add to list of processes and start process
                    processes.append(process)
                    process.start()

                # Move results from output_queue to results_list:
                # Without this part, the output_queue blocks the worker
                # processes so that they will not terminate.
                while output_queue.qsize() > 0:
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

    # -------------------------------------------------------------------------
    # Compute FOM into pandas data frame and save them as a CSV file
    # -------------------------------------------------------------------------

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
            thresholds[:n_signal_estimates],
        ],
        names=[None, 'threshold'],
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

    # Initialize plot directory
    plots_dir = results_dir / 'plots_and_snrs'
    plots_dir.mkdir(exist_ok=True)

    # Save the figures of merit as a CSV file
    print('\nSaving results as CSV file...', end=' ', flush=True)
    file_path = plots_dir / f'figures_of_merit__{metric_function}.csv'
    dataframe.to_csv(file_path, sep='\t', float_format='%.3f')
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create a joint plot of the SNR as a function of the threshold
    # -------------------------------------------------------------------------

    print('Plotting SNR over thresholds...', end=' ', flush=True)

    # Create a new figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # For each planet, keep track of the signal_estimate that produced the
    # highest SNR, as well as the SNRs of all other planets (for plotting)
    max_snr_data: Dict[str, Dict[str, Union[list, np.ndarray]]] = {}

    # Make plot for each planet individually
    for i, planet_key in enumerate(planet_keys):

        # Construct full name of the planet (target star + letter)
        planet_name = f'{target_name} {planet_key}'

        # Select the SNR values from the dataframe
        snr_values = dataframe[planet_key]['snr'].values

        # Get maximum SNR and draw it separately
        max_snr_idx = int(np.argmax(snr_values))
        plt.plot(
            thresholds[max_snr_idx],
            snr_values[max_snr_idx],
            color=adjust_luminosity(f'C{i}'),
            marker='x',
            ms=10,
        )

        # Store signal_estimate with the highest SNR for plotting later
        max_snr_data[planet_key] = dict(
            frame=signal_estimates[max_snr_idx],
            positions=[
                dataframe[_]['new_position'][max_snr_idx].ravel()[0]
                for _ in planet_keys
            ],
            snrs=[
                float(dataframe[_]['snr'][max_snr_idx]) for _ in planet_keys
            ],
        )

        # Plot the SNR as a step function
        ax.step(
            thresholds,
            snr_values,
            'o-',
            color=f'C{i}',
            where='mid',
            label=planet_name,
        )

        # Add a label for each data point
        for (threshold, snr) in zip(thresholds, snr_values):
            ax.annotate(
                text=f'{snr:.2f}\n{threshold:.2f}',
                xy=(threshold, snr),
                ha='center',
                va='center',
                fontsize=1.5,
            )

    # Determine maximum SNR for plot limits
    snr_idx = dataframe.columns.get_level_values(1) == 'snr'
    max_snr = np.max(np.nan_to_num(dataframe.iloc[:, snr_idx].values))

    # Add plot options
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1 * max_snr)
    ax.set_xlabel('Threshold on fraction of passed consistency tests')
    ax.set_ylabel('Signal-to-noise ratio (SNR)')
    ax.set_title('SNR as a function of threshold')
    ax.legend(loc='best')
    ax.set_xticks(np.linspace(0, 1, n_test_positions + 1))
    ax.grid(which='both', ls='--', color='LightGray', alpha=0.5)

    # Save plot as a PNG
    file_path = plots_dir / f'snr_over_threshold__{metric_function}.pdf'
    fig.savefig(file_path, pad_inches=0)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Plot signal_estimate for highest SNR of each planet
    # -------------------------------------------------------------------------

    for planet_key in planet_keys:

        # Define shortcuts
        file_name = f'{target_name}_{planet_key}__{metric_function}.pdf'
        file_path = plots_dir / file_name

        # Create plot
        print(f'Creating {file_name}...', end=' ', flush=True)
        plot_frame(
            frame=max_snr_data[planet_key]['frame'],
            file_path=file_path,
            aperture_radius=aperture_radius,
            positions=max_snr_data[planet_key]['positions'],
            snrs=max_snr_data[planet_key]['snrs'],
        )
        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Plot baseline signal_estimate
    # -------------------------------------------------------------------------

    # Define shortcuts
    file_name = f'baseline__{metric_function}.pdf'
    file_path = plots_dir / file_name

    # Create plot
    print(f'Creating {file_name}...', end=' ', flush=True)
    plot_frame(
        frame=signal_estimates[-1],
        file_path=file_path,
        aperture_radius=aperture_radius,
        positions=[dataframe[_]['new_position'].iloc[-1] for _ in planet_keys],
        snrs=[float(dataframe[_]['snr'].iloc[-1]) for _ in planet_keys],
    )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
