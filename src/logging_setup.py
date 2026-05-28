import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format='%(asctime)s  %(levelname)-8s  %(name)s — %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout,
        force=True,
    )
