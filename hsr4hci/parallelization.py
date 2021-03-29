"""
Utility methods for queues used for parallelization.

The ThreadSafeCounter class is strongly inspired by this blog post:
    http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/
which released the code under a CC license (https://unlicense.org/).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from multiprocessing.queues import Queue as BaseQueue
from typing import Any

import multiprocessing
import re


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class ThreadSafeCounter:
    """
    A threadsafe version of a counter; that is, a counter that can be
    shared between different processes. Thread-safety is achieved by
    using the built-in `get_lock()` method of `multiprocessing.Value`
    variables.

    Args:
        n: The initial value of the counter (usually 0).
    """

    def __init__(self, n: int = 0) -> None:

        self.count = multiprocessing.Value('i', n)

    def increment(self, n: int = 1) -> None:
        """
        Increment the counter by n (default = 1)
        """

        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self) -> int:
        """
        Return the value of the counter.
        """
        with self.count.get_lock():
            return int(self.count.value)


class Queue(BaseQueue):
    """
    A thin wrapper around the `multiprocessing.Queue` class, which is
    mean to solve the problem that some methods (e.g., `Queue.qsize()`)
    raise a `NotImplementedError` on platforms like macOS.

    The basic idea is that we add a ThreadsafeCounter to the class
    members, which we manually increment or decrement whenever something
    is added to or removed from the queue.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:

        # Set up a new multiprocessing context: ctx will contain a context
        # object which has the same attributes as the multiprocessing module.
        ctx = multiprocessing.get_context()

        # Call the constructor of the parent class with this context
        super(Queue, self).__init__(ctx=ctx, *args, **kwargs)

        # Add a variable to keep track of the queue size
        self.queue_size = ThreadSafeCounter(0)

    def put(self, *args: Any, **kwargs: Any) -> None:
        """
        Overload the `put()` method, which is used to add elements to
        the queue. This just calls the `put()` method of the parent
        class (`multiprocessing.Queue`) and increases the `queue_size`
        variable accordingly.

        Args:
            *args: Arguments that are passed to the parent class method
                `multiprocessing.Queue.put()`.
            **kwargs: Keyword arguments that are passed to the parent
                class method `multiprocessing.Queue.put()`.
        """

        # Put a new element into the queue
        super(Queue, self).put(*args, **kwargs)

        # Increase the current size of the queue by 1
        self.queue_size.increment(1)

    def get(self, *args: Any, **kwargs: Any) -> Any:
        """
        Overload the `get()` method, which is used to get elements from
        the queue. This just calls the `get()` method of the parent
        class (`multiprocessing.Queue`) and decreases the `queue_size`
        variable accordingly.

        Args:
            *args: Arguments that are passed to the parent class method
                `multiprocessing.Queue.get()`.
            **kwargs: Keyword arguments that are passed to the parent
                class method `multiprocessing.Queue.get()`.

        Returns:
            The next element in the queue.
        """

        # Get the next element from the queue
        element = super(Queue, self).get(*args, **kwargs)

        # Increase the current size of the queue by 1
        self.queue_size.increment(-1)

        # Return the element
        return element

    def qsize(self) -> int:
        """
        Get the current size (i.e., number of elements) in the queue.

        This function overloads the existing `qsize()` method of
        `multiprocessing.Queue`, which raises a `NotImplementedError`
        on macOS.

        Returns:
            An integer containing the number of elements in the queue.
        """

        return self.queue_size.value

    def empty(self) -> bool:
        """
        Check if the queue is empty.

        This function overloads the existing `empty()` method of
        `multiprocessing.Queue`, which raises a `NotImplementedError`
        on macOS.

        Returns:
            Returns True if there are 0 elements in the queue, and False
            in every other case.
        """

        return self.queue_size.value == 0


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_available_cpu_count() -> int:
    """
    Get the number of (virtual or physical) CPUs on the system by taking
    into account that cpuset may restrict the number of processors that
    are actually *available* to the current process.

    This function tries to check `/proc/self/status` first for any
    limitations on the number of CPUs. If there is a limit, this limit
    is returned; otherwise the function simply returns the number of
    CPUs in the system as given by `multiprocessing.cpu_count()`.

    This is a simplified (and annotated) version of the function
    proposed here:
        https://stackoverflow.com/a/1006301/4100721

    Returns:
        The number of (available) CPUs on the system.
    """

    # Check `/proc/self/status` file for "Cpus_allowed:" keyword
    try:

        # Read in the `/proc/self/status` file
        with open('/proc/self/status', 'r') as status_file:
            file_contents = status_file.read()

        # Search for the `Cpus_allowed:` field
        search_result = re.search(r'(?m)^Cpus_allowed:\s*(.*)$', file_contents)

        if search_result:

            # Get the actual value of the Cpus_allowed field
            cpus_allowed = search_result.group(1)

            # The "CPUs_allowed" field contains a rather cryptic encoding of
            # the CPUs that are available to this process. By parsing its value
            # as done below, cpu_flags becomes, for example:
            #     '0b1011010000000010111000000000000'
            # which is a series of binary flags (that needs to be read right to
            # left) where position N indicates whether or not the N-th CPU is
            # available to us (indexing starts at 0).
            # In the example above, the CPUs that are actually available to us
            # are the following:
            #     Cpus_allowed_list: 12-14,16,25,27-28,30
            # To actually get the number of CPUs available, we simply need to
            # count the number of 1s in the `cpu_flags`.
            cpu_flags = bin(int(cpus_allowed.replace(',', ''), 16))
            cpu_count = cpu_flags.count('1')

            # If the value is not at least 1, something must have gone wrong
            if cpu_count > 0:
                return cpu_count

    # If the `/proc/self/status` file does not exist, ignore the error
    except IOError:
        pass

    # If we have not returned the value for the `/proc/self/status` file, we
    # default to the (unconstrained) number of CPUs on the system
    return multiprocessing.cpu_count()
