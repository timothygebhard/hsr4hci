"""
Utility functions to augment the tqdm library.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Iterator

import contextlib

import joblib
import tqdm


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm.std.tqdm) -> Iterator[None]:
    """
    Context manager to patch joblib to report into a tqdm progress bar
    given as an argument to this function.

    Source: https://stackoverflow.com/a/58936697/4100721

    Args:
        tqdm_object: An tqdm object ; for example, an iterator that
            has been wrapped with tqdm().
    """

    class TqdmBatchCompletionCallback:
        """
        New tqdm callback that is invoked upon batch completion.
        """

        def __init__(self,
                     _: Any,
                     index: int,
                     parallel: joblib.parallel.Parallel):

            self.index = index
            self.parallel = parallel

        def __call__(self,
                     index: int) -> None:

            tqdm_object.update()

            # noinspection PyProtectedMember
            # pylint: disable=protected-access
            if self.parallel._original_iterator is not None:
                self.parallel.dispatch_next()

    # Replace the default joblib.parallel callback for batch completion with
    # the one we have just defined
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    # Yield the updated tqdm object, and ensure that the original callback
    # for joblib.parallel is restored at the end of the context manager
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
