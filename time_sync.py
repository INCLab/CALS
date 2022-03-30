import os


def time_sync():
    start_tsync = os.popen('sudo ntpdate')
    output = start_tsync.read()

    return output