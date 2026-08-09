import logging
import sys
import json
from datetime import datetime, timezone

class JsonFormatter(logging.Formatter):
    def format(self, record):
        msg = record.getMessage()
        if len(msg) > 300:
            msg = msg[:300] + "... [TRUNCATED]"
            
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": msg,
            "logger": record.name
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            clean_extra = {}
            for k, v in record.extra.items():
                v_str = str(v)
                if len(v_str) > 150:
                    clean_extra[k] = v_str[:150] + "... [TRUNCATED]"
                else:
                    clean_extra[k] = v
            log_obj.update(clean_extra)
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

def get_logger(name: str = "baymax") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JsonFormatter())
        logger.addHandler(console_handler)
        
        # Persistent Shortened File Handler
        try:
            import os
            log_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "baymax.log"), encoding="utf-8")
            file_handler.setFormatter(JsonFormatter())
            logger.addHandler(file_handler)
        except Exception:
            pass
            
    return logger
