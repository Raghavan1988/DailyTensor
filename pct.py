def percent_change(series):
    """
    Compute the fractional change between consecutive values.
    """
    pct = []
    for i in range(1, len(series)):
        xi = series[i]
        xim1 = series[i-1]
        val = xi - xim1
        if xim1 == 0:
            val = 0.0
        else:
            val /= (xim1 * 1.0)
        pct.append(val)

    return pct
        
