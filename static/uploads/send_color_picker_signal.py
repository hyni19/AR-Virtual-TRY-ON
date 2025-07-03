import sys
import os

def send_color_picker_signal():
    """
    Send a signal to the running try-on process to open color picker
    This could be implemented via various methods like:
    - File-based signaling
    - Socket communication
    - Inter-process communication
    """
    # Create a signal file that test-2.py will check
    with open('color_picker_signal.txt', 'w') as f:
        f.write('1')

if __name__ == '__main__':
    send_color_picker_signal()