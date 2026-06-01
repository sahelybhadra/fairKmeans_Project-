#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Class for measuring execution time.

Classes
-------
Timer
    Class for measuring execution time.

"""


# %% Libraries
import time
from typing_extensions import Self


# %% Class for measuring execution time
class Timer:
    r"""
    Class for measuring execution time.

    Attributes
    ----------
    elapsed : int
        Elapsed time in nanoseconds.
    is_running : bool
        Whether the timer is running.

    Methods
    -------
    resume()
        Resume the timer.
    pause()
        Pause the timer.
    add()
        Add time to the timer.
    reset()
        Reset the timer.

    Static Methods
    --------------
    resume_all()
        Resume the passed timers.
    pause_all()
        Pause the passed timers.
    add_to_all()
        Add time to the passed timers.
    reset_all()
        Reset the passed timers.

    """

    def __init__(self):
        self._elapsed: int = 0        # nanoseconds

    def resume(self):
        r"""
        Resume the timer.

        Raises
        ------
        RuntimeError
            If the timer is already running.

        Returns
        -------
        None.

        """
        if self.is_running:
            raise RuntimeError(
                f"'{self.__class__.__name__}' object is already running"
                )
        self._start = time.perf_counter_ns()

    def pause(self):
        r"""
        Pause the timer.

        Raises
        ------
        RuntimeError
            If the timer is not running.

        Returns
        -------
        None.

        """
        if not self.is_running:
            raise RuntimeError(
                f"'{self.__class__.__name__}' object is not running"
                )
        self._elapsed += time.perf_counter_ns() - self._start
        delattr(self, '_start')

    def add(self, ns: int):
        r"""
        Add time to the timer.

        Parameters
        ----------
        ns : int
            Time (in nanoseconds) to be added.

        Raises
        ------
        RuntimeError
            If the timer is running.

        Returns
        -------
        None.

        """
        if self.is_running:
            raise RuntimeError(
                f"'{self.__class__.__name__}' object is running"
                )
        self._elapsed += ns

    def reset(self):
        r"""
        Reset the timer.

        Raises
        ------
        RuntimeError
            If the timer is running.

        Returns
        -------
        None.

        """
        if self.is_running:
            raise RuntimeError(
                f"'{self.__class__.__name__}' object is running"
                )
        self._elapsed = 0

    @property
    def elapsed(self) -> int:
        r"""Elapsed time in nanoseconds."""
        if self.is_running:
            raise RuntimeError(
                f"'{self.__class__.__name__}' object is running"
                )
        return self._elapsed

    @property
    def is_running(self) -> bool:
        r"""Whether the timer is running."""
        return hasattr(self, '_start')

    @staticmethod
    def resume_all(*timers: Self):
        r"""
        Resume the passed timers.

        Parameters
        ----------
        *timers : Self
            Timers.

        Returns
        -------
        None.

        """
        for t in timers:
            t.resume()

    @staticmethod
    def pause_all(*timers: Self):
        r"""
        Pause the passed timers.

        Parameters
        ----------
        *timers : Self
            Timers.

        Returns
        -------
        None.

        """
        for t in timers:
            t.pause()

    @staticmethod
    def add_to_all(*timers: Self, ns: int):
        r"""
        Add time to the passed timers.

        Parameters
        ----------
        *timers : Self
            Timers.
        ns : int
            Time (in nanoseconds) to be added to the timers.

        Returns
        -------
        None.

        """
        for t in timers:
            t.add(ns=ns)

    @staticmethod
    def reset_all(*timers: Self):
        r"""
        Reset the passed timers.

        Parameters
        ----------
        *timers : Self
            Timers.

        Returns
        -------
        None.

        """
        for t in timers:
            t.reset()


# %% END OF FILE
