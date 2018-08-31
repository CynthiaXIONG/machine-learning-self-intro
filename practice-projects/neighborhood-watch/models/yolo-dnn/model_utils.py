import os

class ObjectDetectionModel():
    name = ""

    def __init__(self):
        pass

    def predict(self, image_file_name):
        pass

    def _get_input_filename(self, image_file_name):
        return "input/{0}.jpg".format(image_file_name)

    def _get_output_filename(self, image_file_name):
        return "out/{0}-{1}.jpg".format(image_file_name, self.name)

if __name__ == "__main__":
    #test
    pass
