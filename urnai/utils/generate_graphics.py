import os,sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from urnai.utils.logger import Logger
from urnai.trainers.trainer import Trainer
from tkinter.filedialog import askdirectory
import tkinter as tk
from urnai.utils.reporter import Reporter as rp   


def generate_graphics(path=None,with_interface=False,rolling_avg_window_size=20):
    
    if with_interface:
        root = tk.Tk()
        root.withdraw()
        path = askdirectory()
               
              
    
    if path:
        
        trainer = Trainer.__new__(Trainer)
        trainer.full_save_path = path
        trainer.full_save_play_path= path + os.path.sep +"play_files"
        trainer.make_persistance_dirs(True)
        logger = Logger.__new__(Logger)
        logger.rolling_avg_window_size=rolling_avg_window_size
        try:
            logger.load_pickle(path)
            logger.save(path)
            rp.report("successfully generated graphs")
        except:
            rp.report("the graphics could not be generated")
    else:
        rp.report("you need to pass a path")


