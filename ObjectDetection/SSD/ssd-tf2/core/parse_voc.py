import xml.dom.minidom as xdom
from configuration import PASCAL_VOC_DIR, OBJECT_CLASSES
import os

from utils.tools import str_to_int


class ParsePascalVOC(object):
    def __init__(self):
        self.all_xml_dir = PASCAL_VOC_DIR + "Annotations"
        self.all_image_dir = PASCAL_VOC_DIR + "JPEGImages"

    def __process_coord(self, x_min, y_min, x_max, y_max):
        x_min = str_to_int(x_min)
        y_min = str_to_int(y_min)
        x_max = str_to_int(x_max)
        y_max = str_to_int(y_max)
        return int(x_min), int(y_min), int(x_max), int(y_max)

    # parse one xml file
    def __parse_xml(self, xml):
        obj_and_box_list = []
        DOMTree = xdom.parse(os.path.join(self.all_xml_dir, xml))
        annotation = DOMTree.documentElement
        image_name = annotation.getElementsByTagName("filename")[0].childNodes[0].data
        size = annotation.getElementsByTagName("size")
        image_height = 0
        image_width = 0
        for s in size:
            image_height = s.getElementsByTagName("height")[0].childNodes[0].data
            image_width = s.getElementsByTagName("width")[0].childNodes[0].data
        obj = annotation.getElementsByTagName("object")
        for o in obj:
            o_list = []
            obj_name = o.getElementsByTagName("name")[0].childNodes[0].data
            bndbox = o.getElementsByTagName("bndbox")
            for box in bndbox:
                xmin = box.getElementsByTagName("xmin")[0].childNodes[0].data
                ymin = box.getElementsByTagName("ymin")[0].childNodes[0].data
                xmax = box.getElementsByTagName("xmax")[0].childNodes[0].data
                ymax = box.getElementsByTagName("ymax")[0].childNodes[0].data
                xmin, ymin, xmax, ymax = self.__process_coord(xmin, ymin, xmax, ymax)
                o_list.append(xmin)
                o_list.append(ymin)
                o_list.append(xmax)
                o_list.append(ymax)
                break
            o_list.append(OBJECT_CLASSES[obj_name])
            obj_and_box_list.append(o_list)
        return image_name, image_height, image_width, obj_and_box_list

    # xxx.xml image_height image_width xmin ymin xmax ymax class_type xmin ymin xmax ymax class_type ...
    def __combine_info(self, image_name, image_height, image_width, box_list):
        image_dir = self.all_image_dir + "/" + image_name
        line_str = image_dir + " " + image_height + " " + image_width + " "
        for box in box_list:
            for item in box:
                item_str = str(item)
                line_str += item_str
                line_str += " "
        line_str = line_str.strip()
        return line_str

    def write_data_to_txt(self, txt_dir):
        for item in os.listdir(self.all_xml_dir):
            image_name, image_height, image_width, box_list = self.__parse_xml(xml=item)
            print("Writing information of picture {} to {}".format(image_name, txt_dir))
            # Combine the information into one line.
            line_info = self.__combine_info(image_name, image_height, image_width, box_list)
            line_info += "\n"
            with open(txt_dir, mode="a+") as f:
                f.write(line_info)
