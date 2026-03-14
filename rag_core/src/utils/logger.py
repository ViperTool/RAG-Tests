import logging.config
import sys
import os
import config


def init_logging():
    """Инициализация логирования. Вызывается один раз при старте."""

    if not os.path.exists(config.LOGS_DIR):
        os.makedirs(config.LOGS_DIR)

    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'stream': sys.stdout
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'standard',
                'filename': os.path.join(config.LOGS_DIR, 'app.log'),
                'maxBytes': 10 * 1024 * 1024,
                'backupCount': 5,
                'encoding': 'utf8'
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console', 'file']
        }
    }

    logging.config.dictConfig(logging_config)
