import logging
import datetime

# Define custom log levels
logging.WATERMARK = 25
logging.CLIENT = 35
logging.addLevelName(logging.WATERMARK, "WATERMARK")
logging.addLevelName(logging.CLIENT, "CLIENT")

class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.INFO: self.red + self.fmt + self.reset,
            logging.CLIENT : self.blue + self.fmt + self.reset,
            logging.WATERMARK: self.yellow + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

fmt = '%(asctime)s | %(levelname)8s | %(message)s'

# Create a logger
logger = logging.getLogger('server_logger')
logger.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()

# Create a formatter and set it to the handler
formatter = CustomFormatter(fmt)
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)
