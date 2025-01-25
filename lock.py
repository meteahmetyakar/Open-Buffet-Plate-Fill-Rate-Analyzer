import ctypes
from pynput.mouse import Listener as MouseListener
from pynput.keyboard import Listener as KeyboardListener
import keyboard

# Global flag for lock state
locked = False

# Functions to enable/disable input devices
def block_input(state):
    # Block input devices
    ctypes.windll.user32.BlockInput(state)

def toggle_lock():
    global locked
    locked = not locked
    if locked:
        print("Devices locked")
        block_input(True)  # Disable input devices
    else:
        print("Devices unlocked")
        block_input(False)  # Enable input devices

# Listener functions to ensure control remains responsive
def on_key_press(key):
    if not locked:
        return True

# Dummy mouse and keyboard listeners to block events
def on_mouse_move(x, y):
    if locked:
        return False

def on_mouse_click(x, y, button, pressed):
    if locked:
        return False

def on_scroll(x, y, dx, dy):
    if locked:
        return False

def start_listeners():
    mouse_listener = MouseListener(on_move=on_mouse_move, on_click=on_mouse_click, on_scroll=on_scroll)
    keyboard_listener = KeyboardListener(on_press=on_key_press)

    mouse_listener.start()
    keyboard_listener.start()

# Main function
def main():
    print("Press Ctrl + B to toggle lock/unlock.")
    keyboard.add_hotkey('ctrl+b', toggle_lock)

    # Start dummy listeners to ensure state remains responsive
    start_listeners()

    # Keep the program running
    keyboard.wait('ctrl+q')  # Exit on Ctrl + Q

if __name__ == "__main__":
    main()