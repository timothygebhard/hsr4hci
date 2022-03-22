"""
This script can be used to automatically collect all results (logFPF)
of the experiments with and without OC and create the LaTeX code for
the comparison table in the paper.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import json
import time

from hsr4hci.config import get_experiments_dir


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCOLLECT LOGFPF VALUES AND CREATE LATEX TABLE\n')

    # -------------------------------------------------------------------------
    # Define some shortcuts / dictionaries with labels
    # -------------------------------------------------------------------------

    # Get the path to the (main) experiments directory
    experiments_dir = get_experiments_dir() / 'main'

    # Define labels for OC, methods and datasets
    oc_labels = {True: ' + OC', False: ''}
    method_labels = {
        'pca': 'PCA',
        'signal_fitting': 'Signal fitting',
        'signal_masking': 'Signal masking',
    }
    dataset_labels = {
        'beta_pictoris__lp': "Beta Pictoris $L'$",
        'beta_pictoris__mp': "Beta Pictoris $M'$",
        'hr_8799__lp': "HR 8799 $L'$",
        'r_cra__lp': "R Coronae Australis $L'$",
    }

    # List of all methods (= rows in the table)
    methods = [
        'pca',
        'signal_fitting',
        'signal_fitting__oc',
        'signal_masking',
        'signal_masking__oc',
    ]

    # Dictionary of datasets, planets (= columns in the table) plus the
    # corresponding line_ends
    datasets = {
        'beta_pictoris__lp': (('b', '&&'),),
        'beta_pictoris__mp': (('b', '&&'),),
        'hr_8799__lp': (('b', '&'), ('c', '&'), ('d', '&'), ('e', '&&')),
        'r_cra__lp': (('b', '\\\\'),),
    }

    # -------------------------------------------------------------------------
    # Loop over all combinations of method, dataset and planet and get results
    # -------------------------------------------------------------------------

    # Loop over methods (= rows in table)
    for method in methods:

        # Get the subdirectory of the experiment folder (based on OC)
        if method.endswith('oc'):
            folder = '6.4_observing-conditions'
            pure_method = method.split('__')[0]
            oc = True
        else:
            folder = '5.1_first-results'
            pure_method = method
            oc = False

        # Print the (properly formatted) name of the method
        print(method_labels[pure_method] + oc_labels[oc] + ' &&')

        # Loop over datasets and planets (= columns in table)
        for dataset in datasets.keys():
            for planet, line_end in datasets[dataset]:

                # Construct file path to JSON file with results / metrics
                file_path = (
                    experiments_dir
                    / folder
                    / pure_method
                    / dataset
                    / 'results'
                    / f'metrics__{planet}.json'
                )

                # Read in JSON file with results
                with open(file_path, 'r') as json_file:
                    metrics = json.load(json_file)

                # Get median logFPF score and min / max
                av = metrics['log_fpf']['median']
                hi = metrics['log_fpf']['max'] - av
                lo = av - metrics['log_fpf']['min']

                # Construct and print label for the current combination
                label = rf'\logfpfvalue{{{av:4.1f}}}{{-{lo:.1f}}}{{+{hi:.1f}}}'
                print(f'\t{label} {line_end}')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
