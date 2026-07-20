import logging
import logging.config
from pathlib import Path

log_dir = Path(__file__).resolve.parent

formatter = logging.Formatter()
logging.config.dictconfig(
    {
        "version": 1,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(name)s - %(levelname)s | %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "INFO",  # setting to INFO / DEBUG causes conflicts with displaying tqdm progress bars
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": f"{log_dir}/windy_gridworld.log",
                "formatter": "standard",
                "level": "DEBUG",
                "mode": "w",  # write or replace log file
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "file"],  # logs to both console and file
        }
    }
)

logger = logging.getLogger(__name__)

# Loggers for the future
logger.info("Environment succesfully resetted to agent start location: [3.0] & target location: [3,7]")

logger.debug("Agent location: (%d, %d)", self._agent_location[1], self._agent_location[0])