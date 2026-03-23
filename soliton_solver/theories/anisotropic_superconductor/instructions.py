"""
Print terminal instructions for the anisotropic s+id superconductor simulation.

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
        "   Order parameter 1 density display:      F2\n"
        "   Order parameter 2 density display:      F3\n"
        "   Magnetic flux density display:          F4\n"
        "   Phase difference density display:       F5\n"
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