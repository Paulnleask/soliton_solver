"""
Instructions for the Baby Skyrme interactive controls.

Examples
--------
Use ``print_instructions()`` to display the interactive control summary.
"""

def print_instructions():
    """
    Print the interactive control summary for the Baby Skyrme simulation.

    Returns
    -------
    None
        The control summary is printed to the terminal.

    Examples
    --------
    Use ``print_instructions()`` to display the available keyboard and mouse controls.
    """
    print(
        "\nControls:\n"
        "   Energy density display:                 F1\n"
        "   Magnetization field display:            F2\n"
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