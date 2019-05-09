LEVEL = "DEBUG"


def log(log_str):
    if LEVEL == "DEBUG":
        print(log_str)


def setLevel(level):
    global LEVEL
    LEVEL = level
