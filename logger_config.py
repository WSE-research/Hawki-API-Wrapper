import colorlog
import logging

# Configure colorful logger
logger = colorlog.getLogger(__name__)
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)s:%(name)s:%(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))
logger.addHandler(handler)
logger.setLevel(logging.INFO) 