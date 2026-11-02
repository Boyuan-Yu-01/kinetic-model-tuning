"""Display scales are independent of objective transformations."""
def yscale(settings, quantity, default='linear'):
    """Resolve an editable linear/log scale for one plotted quantity."""
    value=(settings or {}).get(quantity,default)
    if value not in ('linear','log'):
        raise ValueError(f'{quantity}: plot scale must be linear or log.')
    return value
