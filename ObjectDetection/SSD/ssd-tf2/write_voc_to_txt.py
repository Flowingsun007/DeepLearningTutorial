from core.parse_voc import ParsePascalVOC
from configuration import TXT_DIR


if __name__ == '__main__':
    voc = ParsePascalVOC()
    voc.write_data_to_txt(txt_dir=TXT_DIR)