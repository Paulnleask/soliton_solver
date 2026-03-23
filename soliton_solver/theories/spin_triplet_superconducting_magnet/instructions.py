"""
Print terminal instructions for the spin triplet superconducting ferromagnet simulation.

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
        "   Magnetization field display:            F2\n"
        "   Order parameter 1 density display:      F3\n"
        "   Order parameter 2 density display:      F4\n"
        "   Magnetic flux density display:          F5\n"
        "   Toggle arrested Newton flow:            n\n"
        "   Toggle arresting criteria:              k\n"
        "   Enter skyrmion placement mode:          s\n"
        "   Enter vortex placement mode:            v\n"
        "   Switch vortex type:                     a\n"
        "   Choose skyrmion/vortex number:          1,...,9\n"
        "   Choose skyrmionium:                     0\n"
        "   Place skyrmion/vortex:                  Left-click\n"
        "   Isorotate skyrmion:                     Drag & release left-click\n"
        "   Leave skyrmion/vortex placement mode:   q\n"
        "   Output results for plotting:            o\n"
        "   Exit simulation:                        Esc\n"
    )