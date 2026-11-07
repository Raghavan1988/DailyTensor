def percent_change(series):
    """
    Compute the fractional change between consecutive values.

        change[i] = (series[i+1] - series[i]) / series[i]

    Returns a list of floats of length len(series) - 1 (empty for fewer than
    two values). When the previous value is 0 the change is undefined, so 0.0
    is reported instead of raising ZeroDivisionError.

    Example:
        >>> percent_change([100, 110, 99, 0, 5])
        [0.1, -0.1, -1.0, 0.0]
    """
    values = list(series)
    return [
        0.0 if prev == 0 else float((curr - prev) / prev)
        for prev, curr in zip(values, values[1:])
    ]
