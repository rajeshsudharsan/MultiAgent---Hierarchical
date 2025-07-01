import logging

class Logger:
    def __init__(self) : 
        print('-----------Constructor-----------')
        self.file_handler = logging.FileHandler('my_application.log')
        self.file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    def getLogger(self,name) :
        print('-----------getLogger-----------')
        print(name)
        self.file_handler.setFormatter(self.file_formatter)
        logger = logging.getLogger(name)
        logger.addHandler(self.file_handler)
        logger.setLevel(logging.DEBUG)
        return logger
        
