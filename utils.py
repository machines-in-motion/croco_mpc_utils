import logging
import yaml

# Global setting used by default in the whole package
GLOBAL_LOG_LEVEL  = 'DEBUG'  # 'DEBUG'
GLOBAL_LOG_FORMAT = 'SHORT' # 'LONG'


grey = "\x1b[38;21m" #"\x1b[38;20m"
blue = "\x1b[1;34m"
yellow = "\x1b[33;20m"
red = "\x1b[31;20m"
bold_red = "\x1b[31;1m"
reset = "\x1b[0m"

FORMAT_LONG   = '[%(levelname)s] %(name)s:%(lineno)s -> %(funcName)s() : %(message)s'
FORMAT_SHORT  = '[%(levelname)s] %(name)s : %(message)s'


class CustomFormatterLong(logging.Formatter):
    
    FORMATS = {
        logging.DEBUG:    blue     + FORMAT_LONG + reset,
        logging.INFO:     grey     + FORMAT_LONG + reset,
        logging.WARNING:  yellow   + FORMAT_LONG + reset,
        logging.ERROR:    red      + FORMAT_LONG + reset,
        logging.CRITICAL: bold_red + FORMAT_LONG + reset
    }


    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class CustomFormatterShort(logging.Formatter):

    FORMATS = {
        logging.DEBUG:    blue     + FORMAT_SHORT + reset,
        logging.INFO:     grey     + FORMAT_SHORT + reset,
        logging.WARNING:  yellow   + FORMAT_SHORT + reset,
        logging.ERROR:    red      + FORMAT_SHORT + reset,
        logging.CRITICAL: bold_red + FORMAT_SHORT + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class CustomLogger:

    def __init__(self, module_name, log_level_name, log_format):

        # Create logger woth desired logging level               
        self.logger = logging.getLogger(module_name)
        
        if(log_level_name=='INFO'):
            self.log_level = logging.INFO
        elif(log_level_name == 'DEBUG'):
            self.log_level = logging.DEBUG
        else:
            print("Unknown logging level. Please choose 'INFO' or 'DEBUG'. ")
        
        self.logger.setLevel(self.log_level)

        # Add custom formatter (long or short)
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)
        if(log_format == 'LONG'):
            self.ch.setFormatter(CustomFormatterLong())
        elif(log_format == 'SHORT'):
            self.ch.setFormatter(CustomFormatterShort())
        else:
            print("Unknown logging format. Please choose 'LONG' or 'SHORT'. ")
        self.logger.addHandler(self.ch)


# Load a yaml file (e.g. simu config file)
def load_yaml_file(yaml_file):
    '''
    Load config file (yaml)
    '''
    with open(yaml_file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data 
