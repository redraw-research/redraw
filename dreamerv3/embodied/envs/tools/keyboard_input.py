from pynput import keyboard

key_map = {
    keyboard.Key.up: 'up',
    keyboard.Key.down: 'down',
    keyboard.Key.left: 'left',
    keyboard.Key.right: 'right',
}

current_key = None


def on_press(key):
    global current_key
    if key in key_map:
        current_key = key_map[key]


def on_release(key):
    global current_key
    if key in key_map:
        current_key = None


def get_keyboard_input():
    global current_key
    return current_key


def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
