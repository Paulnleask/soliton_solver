"""
Print terminal instructions for the liquid crystal simulation controls.

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
        "   Electric charge density display:        F3\n"
        "   Toggle arrested Newton flow:            n\n"
        "   Toggle arresting criteria:              k\n"
        "   Enter skyrmion placement mode:          s\n"
        "   Choose skyrmion number:                 1,...,9\n"
        "   Choose skyrmionium:                     0\n"
        "   Place skyrmion:                         Left-click\n"
        "   Isorotate skyrmion:                     Drag & release left-click\n"
        "   Leave skyrmion placement mode:          q\n"
        "   Output results for plotting:            o\n"
        "   Exit simulation:                        Esc\n"
    )