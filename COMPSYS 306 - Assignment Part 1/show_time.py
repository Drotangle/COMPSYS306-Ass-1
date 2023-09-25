import datetime


def print_time(started=True, seconds=False):
    status = "Started"
    if not started:
        status = "Finished"

    current_time = datetime.datetime.now().time()
    if seconds:
        print(f'{status} at: {current_time.hour % 12}:{current_time.minute}:{current_time.second}')
    else:
        print(f'{status} at: {current_time.hour % 12}:{current_time.minute}')