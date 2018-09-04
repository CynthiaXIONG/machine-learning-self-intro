import os
import sys
import time
import datetime
import traceback
import shutil

import json

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, DateTime, JSON
from sqlalchemy.orm import sessionmaker

from slackclient import SlackClient

import urllib
from smb.SMBConnection import SMBConnection #pysmb lib

sys.path.append(os.path.dirname(__file__))
import nw_settings

sys.path.append(os.path.join(os.path.dirname(__file__), '../../models/yolo-dnn'))
from yolo_v3_tf_mystic123 import YoloV3Mystic123

sys.path.append(os.path.join(os.path.dirname(__file__), '../../utils'))
from profile_timer import ProfileTimer

profile_timer = ProfileTimer()

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

def check_for_new_images(db):
    profile_timer.start_scope("check_for_new_images")

    image_extension = "jpg"
    new_images = []
    
    # check the share samba directory
    # get new images, copy to local cache and delete them
    smb_conn = SMBConnection(nw_settings.SAMBA_USER, nw_settings.SAMBA_PALAVRA_CHAVE, "nw_server", "diogo")
    connected = smb_conn.connect(nw_settings.SAMBA_SERVER_ADDRESS)

    shares_devices = smb_conn.listShares()
    for share in shares_devices:
        if (share.name == nw_settings.SAMBA_SHARED_FILE_NAME):
            shared_files = smb_conn.listPath(share.name, nw_settings.SAMBA_CAMERA_PHOTO_PATH)
            for shared_file in shared_files:
                if (not shared_file.isDirectory):
                    file_name_parts = shared_file.filename.split('.')
                    if (len(file_name_parts) == 2 and file_name_parts[1] == image_extension): #check if image file
                        image_name = file_name_parts[0] 
                        db_entry = db.query(ItemEntry).filter_by(id=image_name).first() #check if already processed
                        if db_entry is None:
                            new_images.append({"name":image_name, "create_time":shared_file.create_time})
                            fo = open(os.path.join(nw_settings.CACHED_IMAGES_INPUT_PATH, shared_file.filename), "wb")
                            smb_conn.retrieveFile(share.name, "{0}/{1}".format(nw_settings.SAMBA_CAMERA_PHOTO_PATH, shared_file.filename), fo)
                            fo.close()
                            smb_conn.deleteFiles(share.name, "{0}/{1}".format(nw_settings.SAMBA_CAMERA_PHOTO_PATH, shared_file.filename))

    smb_conn.close()

    #Old version without samba shared folder
    '''
    all_files = os.scandir(og_img_path)
    new_images = []

    for file_entry in all_files:
        if (file_entry.is_file()):
            file_name_parts = file_entry.name.split('.')
            if (len(file_name_parts) == 2 and file_name_parts[1] == image_extension): #check if image file
                image_name = file_name_parts[0] 

                db_entry = db.query(ItemEntry).filter_by(id=image_name).first() #check if already processed
                if db_entry is None:
                    new_images.append(image_name)

    #copy to cached image path
    for new_image in new_images:
        src_path = os.path.join(og_img_path, "{0}.{1}".format(new_image, image_extension))
        dst_path = os.path.join(cached_img_path, "{0}.{1}".format(new_image, image_extension))
        shutil.copy2(src_path, dst_path)
    '''
    # return sorted by create time (to be processed by the correct order)
    new_images.sort(key=lambda x: x["create_time"])

    return [x["name"] for x in new_images]


def add_entry_to_db(new_image, result, db):
    profile_timer.start_scope("add_entry_to_db")

    #check if already exists
    entry = db.query(ItemEntry).filter_by(id=new_image).first()
    if entry is None:
        #get current timestamp
        current_time = datetime.datetime.now()
        
        entry = ItemEntry(
            id=new_image,
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
    profile_timer.start_scope("post_result_to_slack")
    found_msg = ""
    for item in result:
        found_msg += ("{0}x{1}, ".format(item["count"], item["name"]))
    msg = "img:{0} | found: {1}".format(new_image, found_msg)

    slack_client.api_call("chat.postMessage", channel=nw_settings.SLACK_CHANNEL, text=msg, username='nw-bot', icon_emoji=':owl:')


def main():
    #load model and db
    model = YoloV3Mystic123()
    model.set_image_paths(nw_settings.CACHED_IMAGES_INPUT_PATH, nw_settings.IMAGES_OUTPUT_PATH)

    db = load_db()

    #create slack client
    slack_client = SlackClient(nw_settings.SLACK_TOKEN)

    loop_count = 0
    while True:
        #print("{}: Starting new server cycle".format(time.ctime()))
        #check OG image path and copy new images to cached input image path
        new_images = check_for_new_images(db)

        #infere new image
        for new_image in new_images:
            profile_timer.start_scope("model.predict")
            result = model.predict(new_image)
            #add to db
            entry_added = add_entry_to_db(new_image, result, db)
            #post to slack
            if (entry_added):
                post_result_to_slack(new_image, result, slack_client)

        profile_timer.end_scope()

        #print("{}: Successfully finished server cycle".format(time.ctime()))
        loop_count += 1
        if (loop_count % 10):
            print(profile_timer.scopes)

        time.sleep(nw_settings.SLEEP_INTERVAL)

def test():
    pass


if __name__ == "__main__":
    # Change dir to this script location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    main()

