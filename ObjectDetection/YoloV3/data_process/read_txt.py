from configuration import MAX_TRUE_BOX_NUM_PER_IMG


class ReadTxt():
    def __init__(self, line_bytes):
        super(ReadTxt, self).__init__()
        # bytes -> string
        self.line_str = bytes.decode(line_bytes, encoding="utf-8")

    def parse_line(self):
        line_info = self.line_str.strip('\n')
        split_line = line_info.split(" ")
        box_num = (len(split_line) - 1) / 5
        image_name = split_line[0]
        # print("Reading {}".format(image_name))
        split_line = split_line[1:]
        boxes = []
        for i in range(MAX_TRUE_BOX_NUM_PER_IMG):
            if i < box_num:
                box_xmin = int(float(split_line[i * 5]))
                box_ymin = int(float(split_line[i * 5 + 1]))
                box_xmax = int(float(split_line[i * 5 + 2]))
                box_ymax = int(float(split_line[i * 5 + 3]))
                class_id = int(split_line[i * 5 + 4])
                boxes.append([box_xmin, box_ymin, box_xmax, box_ymax, class_id])
            else:
                box_xmin = 0
                box_ymin = 0
                box_xmax = 0
                box_ymax = 0
                class_id = 0
                boxes.append([box_xmin, box_ymin, box_xmax, box_ymax, class_id])

        return image_name, boxes

