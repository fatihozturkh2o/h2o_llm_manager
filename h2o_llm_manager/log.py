# logs
enable_logs: bool = False


def print_log(message: str):
    if enable_logs:
        print(message)
