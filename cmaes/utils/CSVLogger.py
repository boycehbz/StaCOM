'''
 @FileName    : CSVLogger.py
 @EditTime    : 2022-04-01 19:27:15
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import csv

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None): 
        self.file = None
        self.title = '' if title == None else title
        if fpath is not None:
            self.file = open(fpath, 'w', newline="")
            self.writer = csv.writer(self.file)

    def set_names(self, names):
        # initialize numbers as empty list
        self.names = names
        self.writer.writerow(self.names)
        self.file.flush()

    def append(self, data):
        assert len(self.names) == len(data)
        self.writer.writerow(data)
        self.file.flush()
