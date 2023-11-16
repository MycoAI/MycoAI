
import logging


class LoggingWrapper:

    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M')
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    @staticmethod
    def colors(color):
        if color == "blue":
            return LoggingWrapper.BLUE
        elif color == "green":
            return LoggingWrapper.GREEN
        elif color == "yellow":
            return LoggingWrapper.YELLOW
        elif color == "red":
            return LoggingWrapper.RED
        else:
            return ""

    @staticmethod
    def info(text, color = "",  bold = False):
        if bold:
            LoggingWrapper.logger.info(LoggingWrapper.colors(color) + LoggingWrapper.BOLD + text + "\n" + LoggingWrapper.ENDC)
        else:
            LoggingWrapper.logger.info(LoggingWrapper.colors(color) + text + "\n" + LoggingWrapper.ENDC)

    @staticmethod
    def error(text, color ="red", bold = False):
        if bold:
            LoggingWrapper.logger.error(LoggingWrapper.colors(color)+ LoggingWrapper.BOLD + text + "\n" + LoggingWrapper.ENDC)
        else:
            LoggingWrapper.logger.error(LoggingWrapper.colors(color) + text + "\n" + LoggingWrapper.ENDC)

    @staticmethod
    def warning(text, color = "yellow", bold = False):
        if bold:
            LoggingWrapper.logger.warning(LoggingWrapper.colors(color) + LoggingWrapper.BOLD + text + "\n" + LoggingWrapper.ENDC)
        else:
            LoggingWrapper.logger.warning(LoggingWrapper.colors(color) + text + "\n" +  LoggingWrapper.ENDC)