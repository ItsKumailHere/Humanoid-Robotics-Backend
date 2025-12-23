import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler


class LoggingSetup:
    """
    Setup logging configuration for the RAG Chatbot Backend
    """
    
    def __init__(self, log_level: str = "INFO", log_file: Optional[str] = None):
        self.log_level = getattr(logging, log_level.upper())
        self.log_file = log_file or "logs/app.log"
        
        # Create logs directory if it doesn't exist
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
    def setup_logging(self):
        """
        Configure the logging for the application
        """
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        
        # Create file handler with rotation
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Add handlers
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        # Configure specific loggers for different modules
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("qdrant_client").setLevel(self.log_level)
        logging.getLogger("asyncpg").setLevel(self.log_level)
        logging.getLogger("httpx").setLevel(logging.WARNING)


# Initialize logging when this module is imported
logging_setup = LoggingSetup(log_level="INFO")
logging_setup.setup_logging()

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.info("Logging configuration initialized")