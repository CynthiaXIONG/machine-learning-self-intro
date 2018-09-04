import os
import sys
import time
import pandas as pd
import random

class ProfileTimer():

    def __init__(self):
        self.scopes = pd.DataFrame(columns=["Average", "Max", "Min", "Last", "Count"])
        self.current_timer = 0
        self.current_scope = None

    def start_scope(self, scope_name):
        if (self.current_scope is not None):
            self.end_scope()

        if (scope_name not in self.scopes.index):
            self.scopes.loc[scope_name] = 0
            self.scopes.loc[scope_name]["Min"] = sys.float_info.max

        self.current_scope = scope_name
        self.current_timer = time.time()


    def end_scope(self):
        elapsed_time = time.time() - self.current_timer
        scope_name = self.current_scope

        #log stats
        self.scopes.loc[scope_name]["Count"] += 1
        self.scopes.loc[scope_name]["Last"] = elapsed_time
        if (self.scopes.loc[scope_name]["Min"] > elapsed_time):
            self.scopes.loc[scope_name]["Min"] = elapsed_time
        if (self.scopes.loc[scope_name]["Max"] < elapsed_time):
            self.scopes.loc[scope_name]["Max"] = elapsed_time
        #average
        if (self.scopes.loc[scope_name]["Count"] > 1):
            frac = (self.scopes.loc[scope_name]["Count"] - 1) / self.scopes.loc[scope_name]["Count"]
            self.scopes.loc[scope_name]["Average"] = self.scopes.loc[scope_name]["Average"] * frac + elapsed_time * (1-frac)
        else:
            self.scopes.loc[scope_name]["Average"] = elapsed_time

        #reset
        self.current_scope = None


def main():
    timer = ProfileTimer()
   
    for i in range(10):
        timer.start_scope("test_0")
        time.sleep(random.random())

   
    for i in range(100):
        timer.start_scope("test_1")
        time.sleep(random.random()*0.1)

    timer.end_scope()

    print(timer.scopes)

if __name__ == "__main__":

    main()

