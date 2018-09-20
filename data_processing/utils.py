import sys


def progress(name, value, endvalue, bar_length=50, width=5):
    """ Displays a progressBar in the console to show current task status"""
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\r{0: <{1}} : [{2}]{3}%".format(
        name, width, arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()
    if value == endvalue:
        sys.stdout.write('\n\n')
