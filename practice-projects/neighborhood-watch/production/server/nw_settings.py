# -*- coding: utf-8 -*-

import os

SLACK_TOKEN = os.getenv('SLACK_TOKEN', "") # Should be put in private.py, or set as an environment variable..
SLACK_CHANNEL = "neighborhood_watch"
SLEEP_INTERVAL = 10 #in seconds


# Any private settings are imported here.
try:
    from private import *
except Exception:
    pass

# Any external private settings are imported from here.
try:
    from config.private import *
except Exception:
    pass