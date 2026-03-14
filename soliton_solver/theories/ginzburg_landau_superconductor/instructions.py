"""
Print terminal control instructions for the Ginzburg-Landau superconductor simulation.

Examples
--------
>>> print_instructions()
"""

def print_instructions():
    """
    Print the keyboard controls used by the interactive simulation.

    Returns
    -------
    None
        The control instructions are printed to the terminal.

    Examples
    --------
    >>> print_instructions()
    """
    print(
        "\nControls:\n"
        "   Energy density display:                 F1\n"
        "   Order parameter density display:        F2\n"
        "   Magnetic flux density display:          F3\n"
        "   Toggle arrested Newton flow:            n\n"
        "   Toggle arresting criteria:              k\n"
        "   Enter vortex placement mode:            v\n"
        "   Switch vortex type:                     a\n"
        "   Choose vortex number:                   1,...,9\n"
        "   Place vortex:                           Left-click\n"
        "   Leave vortex placement mode:            q\n"
        "   Output results for plotting:            o\n"
        "   Exit simulation:                        Esc\n"
    )