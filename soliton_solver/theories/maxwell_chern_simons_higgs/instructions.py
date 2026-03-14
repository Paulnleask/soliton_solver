"""
Print terminal instructions for the Maxwell-Chern-Simons-Higgs simulation controls.

Examples
--------
>>> print_instructions()
"""

def print_instructions():
    """
    Print the interactive controls for the simulation.

    Examples
    --------
    >>> print_instructions()
    """
    print(
        "\nControls:\n"
        "   Energy density display:                 F1\n"
        "   Order parameter density display:        F2\n"
        "   Magnetic flux density display:          F3\n"
        "   Electric charge density display:        F4\n"
        "   Noether charge density display:         F5\n"
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