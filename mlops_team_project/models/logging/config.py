import logging
import sys
from pathlib import Path
from logging.config import dictConfig
import config

__file__ = "config.py"
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = Path(BASE_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging_config = {
	"version": 1,
	"formatters": {
		"minimal": {"format": "%(asctime)s -- %(message)s"},
		"detailed": {
			"format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
		},
	},
	"handlers": {
		"console": {
		"class": "logging.StreamHandler",
		"stream": sys.stdout,
		"formatter": "minimal",
		"level": logging.DEBUG,
		},
	"info": {
		"class": "logging.handlers.RotatingFileHandler",
		"filename": Path(LOGS_DIR, "info.log"),
		"maxBytes": 10485760,  # 1 MB
		"backupCount": 10,
		"formatter": "detailed",
		"level": logging.INFO,
		},
	"error": {
		"class": "logging.handlers.RotatingFileHandler",
		"filename": Path(LOGS_DIR, "error.log"),
		"maxBytes": 10485760,  # 1 MB
		"backupCount": 10,
		"formatter": "detailed",
		"level": logging.ERROR,
		},
},
	"root": {
		"handlers": ["console", "info", "error"],
		"level": logging.INFO,
		"propagate": True,
	},
}

dictConfig(config.logging_config)
logger = logging.getLogger(__name__)

logging.debug("Debug")
logging.info("Information")
logging.warning("Warning")
logging.error("Error")
logging.critical("Critical Error")