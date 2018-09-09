# -*- coding: utf-8 -*-

import os

SLACK_TOKEN = os.getenv('SLACK_TOKEN', "") # should be put in private.py, or set as an environment variable..
SLACK_CHANNEL = "neighborhood_watch"
SLEEP_INTERVAL = 1 # in seconds


OG_IMAGES_INPUT_PATH = os.path.join(os.path.dirname(__file__), './og_input_test') # original input path of the images
CACHED_IMAGES_INPUT_PATH = os.path.join(os.path.dirname(__file__), './cached_input') # path where the new input images should be copied to so they can be accessed locally
IMAGES_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), './output') # output path after image detection (good to check if it is working properly)

# any private settings are imported here.
try:
    from private import *
except Exception:
    pass