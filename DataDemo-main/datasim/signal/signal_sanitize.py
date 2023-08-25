# -*- coding: utf-8 -*-
import pandas as pd


def signal_sanitize(signal):
    """Reset indexing for Pandas Series

    Parameters
    ----------
    signal : Series
        The indexed input signal (pandas set_index())

    Returns
    -------
    Series
        The default indexed signal

    Examples
    --------
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=10, sampling_rate=1000, frequency=1)
    >>> df = pd.DataFrame({'signal': signal, 'id': [x*2 for x in range(len(signal))]})
    >>>
    >>> df = df.set_index('id')
    >>> default_index_signal = nk.signal_sanitize(df.signal)
    >>>

    """

    # Series check for non-default index
    if type(signal) is pd.Series and type(signal.index) != pd.RangeIndex:
        return signal.reset_index(drop=True)

    return signal
