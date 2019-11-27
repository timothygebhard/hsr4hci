"""
General purpose utilities, e.g., cropping arrays.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from bisect import bisect
from copy import deepcopy
from math import modf
from types import SimpleNamespace
from typing import Callable, List, Sequence, Tuple, Union

from astropy.nddata.utils import add_array
from scipy import ndimage

import numpy as np


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def add_array_with_interpolation(array_large: np.ndarray,
                                 array_small: np.ndarray,
                                 position: Tuple[Union[int, float],
                                                 Union[int, float]]
                                 ) -> np.ndarray:
    """
    An extension of astropy.nddata.utils.add_array to add a smaller
    array at a given position in a larger array. In this version, the
    position may also be a float, in which case bilinear interpolation
    is used when adding array_small into array_large.

    Args:
        array_large: Large array, into which array_small is added into.
        array_small: Small array, which is added into array_large.
        position: The target position of the small array’s center, with
            respect to the large array. Coordinates should be in the
            same order as the array shape, but can also be floats.

    Returns:
        The new array, constructed as the the sum of `array_large`
        and `array_small`.
    """

    # Split the position into its integer and its fractional parts
    fractional_position, integer_position = \
        tuple(zip(*tuple(modf(x) for x in position)))

    # Create an empty with the same size as array_larger and add the
    # small array at the approximately correct position
    dummy = np.zeros_like(array_large)
    dummy = add_array(dummy, array_small, integer_position)

    # Use scipy.ndimage.shift to shift the array to the exact
    # position, using bilinear interpolation
    dummy = ndimage.shift(dummy, fractional_position, order=1)

    return array_large + dummy


def crop_center(array: np.ndarray,
                size: Tuple[int, ...]) -> np.ndarray:
    """
    Crop an n-dimensional array to the given size around its center.

    Args:
        array: The numpy array to be cropped.
        size: A tuple containing the size of the cropped array. To not
            crop along a specific axis, you can specify the size of
            that axis as -1.

    Returns:
        The input array, cropped to the desired size around its center.
    """

    start = tuple(map(lambda x, dx: None if dx == -1 else x//2 - dx//2,
                      array.shape, size))
    end = tuple(map(lambda x, dx: None if dx == -1 else x + dx, start, size))
    slices = tuple(map(slice, start, end))

    return array[slices]


def split_into_n_chunks(sequence: Sequence,
                        n_chunks: int) -> List[Sequence]:
    """
    Split a given `sequence` (list, numpy array, ...) into `n_chunks`
    "chunks" (sub-sequences) of approximately equal length, which are
    returned as a list.

    Source: https://stackoverflow.com/a/2135920/4100721

    Args:
        sequence: The sequence which should be split into chunks.
        n_chunks: The number of chunks into which `sequence` is split.

    Returns:
        A list containing the original sequence split into chunks.
    """

    # Basic sanity checks
    if len(sequence) < n_chunks:
        raise ValueError(f"n_chunks is greater than len(sequence): "
                         f"{n_chunks} > {len(sequence)}")
    if n_chunks <= 0:
        raise ValueError("n must be a positive integer!")

    # Compute number of chunks (k) and number of chunks with extra elements (m)
    k, m = divmod(len(sequence), n_chunks)

    # Split the sequence into n chunks of (approximately) size k
    return [sequence[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
            for i in range(n_chunks)]


def split_positions_into_chunks(list_of_positions: List[Tuple[int, int]],
                                weight_function: Callable,
                                n_chunks: int) -> List[List[Tuple[int, int]]]:
    """
    Takes a `list_of_positions` and a function that assigns an single
    position a weight, and then splits the `list_of_positions` into
    `n_chunks` sub-lists (chunks), such the sum of the weights of the
    positions in each chunk is approximately the same.

    This is, in principle, an equal sum partitioning problem, and
    finding its optimal solution is an NP-hard problem. This function,
    therefore, only implements a greedy heuristic which gives a simple
    approximation that should, however, be sufficient for the purpose
    of this function (i.e., helping us to balance the load of parallel
    jobs on a cluster).

    Args:
        list_of_positions: A list of tuples (x, y) containing the
            positions which we want distribute to multiple chunks.
        weight_function: A function which takes a single position (i.e.,
            a tuple (x, y)) as an input and returns a number that can
            be used as a weight for this position. In practice, this
            function will usually return the area of a collection region
            associated with a given position.
        n_chunks: The number of chunks into which we want to split the
            `list_of_positions`.

    Returns:
        A list containing `n_chunks` lists ("chunks"), which each
        contain a (variable) number of positions. The union of all these
        chunks is the original `list_of_positions`.
    """

    # -------------------------------------------------------------------------
    # Calculate the weight for every given position
    # -------------------------------------------------------------------------

    # Compute the weight for every position and create a list of Namespace
    # objects to conveniently keep together a position and its weight
    weighted_positions = list()
    for position in list_of_positions:
        weighted_position = SimpleNamespace(position=position,
                                            weight=weight_function(position))
        weighted_positions.append(weighted_position)

    # Sort the weighted positions descendingly by their weight
    weighted_positions = \
        sorted(weighted_positions, key=lambda x: x.weight, reverse=True)

    # -------------------------------------------------------------------------
    # Initialize the partitioning and compute the target_sum
    # -------------------------------------------------------------------------

    # Initialize the empty partitioning with target number of chunks
    partitioning = \
        [SimpleNamespace(weight_sum=0, elements=[]) for _ in range(n_chunks)]

    # Compute the target sum. Ideally, the weights of all elements in a chunk
    # should always add up to this value.
    target_sum = sum(_.weight for _ in weighted_positions) / n_chunks

    # -------------------------------------------------------------------------
    # Greedily distribute the positions to the chunks (initialization step)
    # -------------------------------------------------------------------------

    # Create a copy of the weighted_positions from which we can remove elements
    # so we can easily keep track of which positions we have used already
    unused = deepcopy(weighted_positions)

    # While there are still unused positions that need to be added to a chunk,
    # we keep distributing its elements to the chunks in the partitioning
    while unused:

        # Loop over all chunks. Per round, each chunk only gets one position.
        # This helps to ensure that in the end, there are not some chunks with
        # only few positions (with high weights), and some chunks with many
        # positions (with low weights), but that every chunk contains a mix.
        for chunk in partitioning:

            # Compute the difference between the target_sum and the weight_sum
            # of the chunk that we are looking at
            difference = target_sum - chunk.weight_sum

            # If the current chunk already has a weight_sum greater than the
            # target_sum, we skip it (i.e., don't add more positions to it)
            if difference < 0:
                continue

            # Find the unused element that, if we add it to the current chunk,
            # gets the chunk's weight_sum the closest to the target_sum
            idx = bisect([_.weight for _ in unused], difference)
            element = unused[min(len(unused) - 1, idx)]

            # Add the element to the current chunk
            chunk.elements.append(element)
            chunk.weight_sum += element.weight

            # Remove the element from the unused list. In case the unused list
            # is now empty, we can stop the loop..
            unused.remove(element)
            if not unused:
                break

    # Sort the partitioning we have obtained by the weight_sum of the chunks
    partitioning = sorted(partitioning, key=lambda x: x.weight_sum)

    # -------------------------------------------------------------------------
    # Move positions between chunks to improve distribution (optimization step)
    # -------------------------------------------------------------------------

    # We keep going as long as we can decrease the distribution error
    while True:

        # Create a backup of the current partitioning, in case the swaps we
        # do reduce how well the positions are distributed over the chunks
        old_partitioning = deepcopy(partitioning)

        # Compute distribution error: This quantity sums up how far the
        # weight_sum of each chunk is from the ideal value (target_sum)
        old_error = sum(abs(_.weight_sum - target_sum) for _ in partitioning)

        # Loop over the potential source chunks, that is, chunks from which we
        # want to remove an element in order to add it to another chunk
        for source_chunk in partitioning:

            # Compute how far the current chunk is from the ideal value
            difference = target_sum - source_chunk.weight_sum

            # We skip all chunks that have less weight than they should
            if difference > 0:
                continue

            # Find that element of the source chunk that we need to remove to
            # get us as close as possible to the target_sum
            idx = int(np.argmin([abs(_.weight - difference)
                                 for _ in source_chunk.elements]))
            element = source_chunk.elements[idx]

            # Find the subset with the lowest weight_sum. This is where we will
            # move the element that we remove from the source_chunk.
            idx = int(np.argmin([_.weight_sum for _ in partitioning]))
            target_chunk = partitioning[idx]

            # Move the element from the source to the target subset
            target_chunk.elements.append(element)
            target_chunk.weight_sum += element.weight
            source_chunk.elements.remove(element)
            source_chunk.weight_sum -= element.weight

        # Compute the new distribution error after the latest round of swaps
        new_error = sum(abs(_.weight_sum - target_sum) for _ in partitioning)

        # If we did not improve the distribution error by the latest round of
        # swaps, we restore the backup of the last partitioning and stop
        if new_error >= old_error:
            partitioning = old_partitioning
            break

    # Again, sort of the (final) partitioning by the weight_sum of the chunks
    partitioning = sorted(partitioning, key=lambda x: x.weight_sum)

    # -------------------------------------------------------------------------
    # Drop redundant information and return result
    # -------------------------------------------------------------------------

    # For the output, we can drop all weights and weight_sums again
    result = [[__.position for __ in _.elements] for _ in partitioning]

    # Final sanity check: Make sure that we have not lost any positions
    if not sorted(sum(result, [])) == sorted(list_of_positions):
        raise RuntimeError('Something went wrong with the partitioning!')

    return result
