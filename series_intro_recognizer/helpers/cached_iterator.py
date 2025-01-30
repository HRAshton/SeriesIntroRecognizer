from typing import TypeVar, Iterator

T = TypeVar('T')


def iterate_with_cache(orig_iter: Iterator[T], n: int) -> Iterator[tuple[tuple[int, T], tuple[int, T]]]:
    """
    Iterates over an iterator, yielding pairs of N elements with a cache of N elements.

    For example, if N=2, the iterator will yield pairs of elements (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), ...

    1st element will be stored in the cache from 1st to Nth iteration, then it will be unloaded, and the N+1st element
    will be stored in the cache.
    :param orig_iter: Original iterator
    :param n: Number of subsequent elements to cache
    :return: Iterator of pairs of elements
    """
    # Buffer to store elements from the iterator, allowing for caching up to N future elements
    buffer = []
    index = 0  # To keep track of the index of each item

    # Populate the initial buffer
    try:
        for _ in range(n + 1):
            buffer.append((index, next(orig_iter)))
            index += 1
    except StopIteration:
        # In case there are fewer than N+1 elements in the iterator
        pass

    # Main loop to yield pairs
    while len(buffer) > 1:  # Need at least two items to make a pair
        current_index, current_content = buffer[0]

        # Create pairs with the next N elements, if available
        for i in range(1, min(n + 1, len(buffer))):
            next_index, next_content = buffer[i]
            yield (current_index, current_content), (next_index, next_content)

        # Remove the processed element from the buffer
        buffer.pop(0)

        # Try to add a new element to the buffer if possible
        try:
            buffer.append((index, next(orig_iter)))
            index += 1
        except StopIteration:
            # No more elements to fetch from the original iterator
            pass
