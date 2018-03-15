# Helper functions for Search classes
import random
import guerilla.data_handler as dh


def k_bot(arr, k, key=None):
    """
    Selects the k-lowest valued elements from the input list. i.e. k = 1 would yield the lowest valued element.
    Uses quickselect.
    Input:
        arr [List]
            List of items from which the k-highest valued items should be selected.
        k [Int]
            The number of highest valued items to be returned.
        key [Function] (Optional)
            Used to convert items in arr to values. Useful when dealing with objects. Default is identity function.
    Output:
        output [List]
            k-highest valued items.
    """
    if k <= 0:
        raise ValueError("K-Bot error: %d is an invalid value for k, k must be > 0." % k)

    if len(arr) < k:
        return arr

    # Identity
    if key is None:
        key = lambda x: x

    # quickselect -> also partitions
    quickselect(arr, 0, len(arr) - 1, k - 1, key=key)

    return arr[:k]


def quickselect(arr, left, right, k, key=None):
    """
    Return the array with the k-th smallest valued element at the k-th index.
    Note that this requires left <= k <= right, as the k-th element can't be placed outside of the considered arr slice.
    Based on: https://en.wikipedia.org/wiki/Quickselect
    Note: left and right input are currently unused.
    Input:
        arr [List]
            List of items from which the k-highest valued items should be selected.
        left [Int]
            Starting index of the array to be considered.
        right [Int]
            Ending index of the array to be considered. (Inclusive)
        k [Int]
            The number of low elements to place.
        key [Function] (Optional)
            Used to convert items in arr to values. Useful when dealing with objects. Default is identity function.
    Output:
        output [List]
            k-highest valued items.
    """

    assert (left <= k <= right)

    if key is None:
        key = lambda x: x

    while True:
        # Base case -> If one element then return that element
        if left == right:
            return arr[right]
        pivot_idx = random.randint(left, right)
        pivot_idx = partition(arr, left, right, pivot_idx, key=key)
        if k == pivot_idx:
            return arr[k]
        elif k < pivot_idx:
            right = pivot_idx - 1
        else:
            left = pivot_idx + 1


def partition(arr, left, right, pivot_idx, key=None):
    """
    Partitions array inplace into two parts, those smaller than pivot, and those larger.
    Based on partition pseudo-code from: https://en.wikipedia.org/wiki/Quickselect
    Input:
        arr [List]
            List of items to be partitioned.
        left [Int]
            Starting index of the array to be partitioned.
        right [Int]
            Ending index of the array to be partitioned. (Inclusive)
        pivot_idx [Int]
            Index of the pivot.
        key [Function] (Optional)
            Used to convert items in arr to values. Useful when dealing with objects. Default is identity function.
    Output:
        store_idx [Int]
            Pivot location.
    """

    if key is None:
        key = lambda x: x

    pivot_val = key(arr[pivot_idx])

    # move pivot to end
    arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]

    store_idx = left
    for i in range(left, right):
        if key(arr[i]) < pivot_val:
            arr[store_idx], arr[i] = arr[i], arr[store_idx]
            store_idx += 1
    # Move pivot into right place
    arr[right], arr[store_idx] = arr[store_idx], arr[right]

    # Return pivot location
    return store_idx


def material_balance(fen, normalize=False):
    """
    Returns the material advantage of a given FEN.
    Material Imbalance = Score[side to move] - Score[not side to move].
    Input:
        fen [String]
            FEN to evaluate.
        normalize [Boolean]
            If True then output is normalized between 0 and 1.
                1 corresponds to the highest possible imbalance of 39 (If > due to promotions, then reduced to 39).
                0.5 corresponds to no imbalance (i.e. score[white] == score[black])
                0 corresponds to the lower possible imbalance of -39.
    Output:
        imbalance [Int]
            Material score advantage (positive) or disadvantage (negative) for the side to move.
    """

    NORM_UPPER = 39
    NORM_LOWER = -NORM_UPPER

    scores = dh.material_score(fen)

    if dh.white_is_next(fen):
        output = scores['w'] - scores['b']
    else:
        output = scores['b'] - scores['w']

    if normalize:
        # set score between -39 and 39
        output = min(NORM_UPPER, max(NORM_LOWER, output))

        # map
        output = output / (NORM_UPPER * 2.0) + 0.5

    return output
