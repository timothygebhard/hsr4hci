"""
Utility functions to augment the tqdm library.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import contextlib

import joblib


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into a tqdm progress bar
    given as an argument to this function.

    Source: https://stackoverflow.com/a/58936697/4100721

    Args:
        tqdm_object: An object of type `tqdm._tqdm.tqdm`; for example,
            an iterator that has been wrapped with tqdm().
    """

    class TqdmBatchCompletionCallback:
        """
        New tqdm callback that is invoked upon batch completion.
        """

        def __init__(self, _, index, parallel):
            self.index = index
            self.parallel = parallel

        def __call__(self, index):

            tqdm_object.update()

            # noinspection PyProtectedMember
            # pylint: disable=protected-access
            if self.parallel._original_iterator is not None:
                self.parallel.dispatch_next()

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
