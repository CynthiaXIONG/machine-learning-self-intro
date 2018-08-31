import os
import sys
import time
import datetime
import traceback

import json

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, DateTime, JSON
from sqlalchemy.orm import sessionmaker

from slackclient import SlackClient

sys.path.append(os.path.dirname(__file__))
import nw_settings

sys.path.append(os.path.join(os.path.dirname(__file__), '../../models/yolo-dnn'))
from yolo_v3_tf_mystic123 import YoloV3Mystic123

#create db entry
Base = declarative_base()
class ItemEntry(Base):
    """
    A table to store data of the items.
    """
    __tablename__ = 'neighbohood_watch_processed_items'

    id = Column(String, primary_key=True)
    date = Column(DateTime)
    data = Column(String)

def load_db():
    #start db session
    db_path = "sqlite:///" + os.path.join(os.path.dirname(__file__), "processed_items.db")
    engine = create_engine(db_path, echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

def fetch_new_images(db):
    pass

def add_entry_to_db(new_image, result, db):
    #id is the image name
    id = new_image
    #check if already exists
    entry = db.query(ItemEntry).filter_by(id=id).first()
    if entry is None:
        #get current timestamp
        current_time = datetime.datetime.now()
        
        entry = ItemEntry(
            id=id,
            date=current_time,
            data=json.dumps(result)
        )

        db.add(entry)
        db.commit()

        return True
    else:
        print("ERROR, img:{0} was already in the database".format(new_image))
        return False


def post_result_to_slack(new_image, result, slack_client):
    found_msg = ""
    for item in result:
        found_msg += ("{0}x{1}, ".format(item["count"], item["name"]))
    msg = "img:{0} | found: {1}".format(new_image, found_msg)

    slack_client.api_call("chat.postMessage", channel=nw_settings.SLACK_CHANNEL, text=msg, username='nw-bot', icon_emoji=':owl:')


if __name__ == "__main__":
    # Change dir to this script location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    #load model and db
    model = YoloV3Mystic123()
    db = load_db()

    #create slack client
    slack_client = SlackClient(nw_settings.SLACK_TOKEN)

    while True:
        print("{}: Starting new server cycle".format(time.ctime()))
        #fetch new untested images
        new_images = ["dog", "person", "pi1", "test"] #fetch_new_images(db)

        #infere new image
        for new_image in new_images:
            result = model.predict(new_image)
            #add to db
            entry_added = add_entry_to_db(new_image, result, db)
            #post to slack
            if (entry_added):
                post_result_to_slack(new_image, result, slack_client)

        print("{}: Successfully finished server cycle".format(time.ctime()))
        time.sleep(nw_settings.SLEEP_INTERVAL)