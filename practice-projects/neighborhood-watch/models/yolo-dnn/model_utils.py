import os

class ObjectDetectionModel():
    name = ""
    input_path = "input"
    output_path = "out"

    def __init__(self):
        pass

    def predict(self, image_file_name):
        pass

    def set_image_paths(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def _get_input_filename(self, image_file_name):
        return os.path.join(self.input_path, "{0}.jpg".format(image_file_name))

    def _get_output_filename(self, image_file_name):
        return os.path.join(self.output_path, "{0}-{1}.jpg".format(image_file_name, self.name))

if __name__ == "__main__":
    #test
    pass
