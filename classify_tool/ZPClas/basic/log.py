import os
import sys

class Logger(object):
    def __init__(self, log_path="default.log"):
        self.terminal = sys.stdout
        self.log = open(log_path, "wb", buffering=0)
 
    def print(self, *message):
        message = ",".join([str(it) for it in message])
        self.terminal.write(str(message) + "\n")
        self.log.write(str(message).encode('utf-8') + b"\n")
 
    def flush(self):
        self.terminal.flush()
        self.log.flush()
 
    def close(self):
        self.log.close()

class LoggerInfer(object):
    def __init__(self, log_path="default.log"):
        # self.terminal = sys.stdout
        self.log = open(log_path, "wb", buffering=0)
 
    def print(self, *message):
        message = ",".join([str(it) for it in message])
        # self.terminal.write(str(message) + "\n")
        self.log.write(str(message).encode('utf-8') + b"\n")
 
    def flush(self):
        # self.terminal.flush()
        self.log.flush()
 
    def close(self):
        self.log.close()
# log=LoggerQ('test.log')

# log.print('牛啊')
# time.sleep(8)
# log.print('5')