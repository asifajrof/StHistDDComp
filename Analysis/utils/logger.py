
import logging

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

# Example usage:
if __name__ == "__main__":
    logger = get_logger("example_logger")
    logger.info("This is an info message.")
    logger.error("This is an error message.")