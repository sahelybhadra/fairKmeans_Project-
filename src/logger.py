#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for logging.

Routine Listings
----------------
get()
    Get logger.
shutdown()
    Shut down logger.

"""


# %% Libraries
import logging
import os
import sys
from logging import Logger


# %% Get logger
def get(*, name: str | None = None, log_dir: str | None = None) -> Logger:
    r"""
    Get logger.

    Exactly one of `name` or `log_dir` must be specified.

    Parameters
    ----------
    name : str | None, optional
        Logger name. The default is None.
    log_dir : str | None, optional
        Log directory. The default is None.

    Returns
    -------
    Logger
        Logger.

    """
    assert (name is None) ^ (log_dir is None)
    if name is not None:
        lgr = logging.getLogger(name)
    elif log_dir is not None:
        lgr = logging.getLogger(log_dir)
    lgr.setLevel(logging.DEBUG)
    if log_dir is not None:
        fh = logging.FileHandler(os.path.join(log_dir, 'log'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s  %(levelname)s  %(filename)s  %(funcName)s'
            '  %(message)s'
            ))
        lgr.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    lgr.addHandler(ch)
    return lgr


# %% Shut down logger
def shutdown(lgr: Logger):
    r"""
    Shut down logger.

    Parameters
    ----------
    lgr : Logger
        Logger to shut down.

    Returns
    -------
    None.

    """
    for hdlr in lgr.handlers:
        lgr.removeHandler(hdlr)
        if isinstance(hdlr, logging.StreamHandler):
            hdlr.flush()
        elif isinstance(hdlr, logging.FileHandler):
            hdlr.close()
        else:
            print(f"Unknown handler '{type(hdlr)}' in logger; may not"
                  f" close correctly")
            hdlr.close()


# %% END OF FILE
