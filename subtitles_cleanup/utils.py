import logging
from typing import Optional

class SingletonLogger:
    _instance: Optional[logging.Logger] = None

    @classmethod
    def get_logger(cls) -> logging.Logger:
        if cls._instance is None:
            # Create and configure the logger
            logger = logging.getLogger("AppLogger")
            logger.setLevel(logging.INFO)

            # Define handlers
            file_handler = logging.FileHandler('app.log')
            file_handler.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)

            # Define a common format
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s:%(funcName)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add handlers to the logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

            # Store the logger instance
            cls._instance = logger
        return cls._instance
