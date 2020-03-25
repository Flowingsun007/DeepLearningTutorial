from data_process.parse_coco import ParseCOCO
from configuration import TXT_DIR


if __name__ == '__main__':
    coco = ParseCOCO()
    coco.write_data_to_txt(txt_dir=TXT_DIR)