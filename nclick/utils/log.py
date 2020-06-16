import time
import datetime
from contextlib import contextmanager
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG


def get_current_time():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M')

class Log:
    @classmethod
    def create_logger(cls, logger_name: str, logfile_path: str):
        # logger
        logger_ = getLogger(logger_name)
        logger_.setLevel(DEBUG)

        # formatter
        fmr = Formatter("[%(asctime)s, %(levelname)s] %(message)s")

        # file handler
        fh = FileHandler(logfile_path)
        fh.setLevel(DEBUG)
        fh.setFormatter(fmr)

        # stream handler
        ch = StreamHandler()
        ch.setLevel(INFO)
        ch.setFormatter(fmr)

        logger_.addHandler(fh)
        logger_.addHandler(ch)

    @classmethod
    def write(cls, message: str, logger_name: str, level: str):
        if level == 'info':
            getLogger(logger_name).info(message)
        if level == 'debug':
            getLogger(logger_name).debug(message)

    @classmethod
    @contextmanager
    def timer(cls,
              process_name: str,
              logger_name: str = None,
              level: str = 'info'):

        t0 = time.time()
        yield
        message = f"{process_name} ({time.time()-t0:.0f}s)"
        if logger_name:
            Log.write(message, logger_name, level)
        else:
            print(message)

    @classmethod
    def emphasis(cls,
                 message: str,
                 logger_name: str = None,
                 level: str = 'info',
                 line_length: int = 50,
                 ):

        if logger_name:
            Log.write('='*line_length, logger_name, level)
            Log.write(message, logger_name, level)
            Log.write('='*line_length, logger_name, level)
        else:
            print('='*line_length)
            print(message)
            print('='*line_length)