import glob
import xml.etree.ElementTree as ET
import os
import csv

train_xml_path = "custom_dataset/xml/train"
# test_xml_path = "custom_dataset/xml/test"
train_yolo_path = "custom_dataset/yolo/train"
# test_yolo_path = "custom_dataset/yolo/test"
class_names_path = "custom_dataset/yolo/classes.txt"
if os.path.exists(train_yolo_path) is False:
    os.makedirs(train_yolo_path)
"""    
if os.path.exists(test_yolo_path) is False:
    os.make_dir(test_yolo_path)
"""    
class_names = []

for xml_file in glob.glob(train_xml_path + "/*.xml"):
    with open("custom_dataset/yolo/train.csv", mode="a", newline="") as train_csv:
        image_file = os.path.basename(xml_file).replace(".xml", ".jpg")
        text_file = os.path.basename(xml_file).replace(".xml", ".txt")
        data = [image_file, text_file]
        writer = csv.writer(train_csv)
        writer.writerow(data)
        train_csv.close()
    tree = ET.parse(xml_file)
    root = tree.getroot()
    save_path = os.path.join(train_yolo_path, os.path.basename(xml_file).replace(".xml", ".txt"))
    im_width = int(root.find('size').find('width').text)
    im_height = int(root.find('size').find('height').text)
    for obj in root.iter('object'):
        obj_name = obj.find('name').text
        if obj_name not in class_names:
            class_names.append(obj_name)
        obj_bndbox = obj.find('bndbox')
        xmin = int(obj_bndbox.find('xmin').text)
        ymin = int(obj_bndbox.find('ymin').text)
        xmax = int(obj_bndbox.find('xmax').text)
        ymax = int(obj_bndbox.find('ymax').text)
        
        class_id = class_names.index(obj_name)
        x_center = ((xmin + xmax) / 2) / im_width
        y_center = ((ymin + ymax) / 2) / im_height
        width = (xmax - xmin)/im_width
        height = (ymax - ymin)/im_height
        
        with open(save_path, "a") as f:
            string = "{} {} {} {} {}\n".format(class_id, x_center, y_center, width, height)
            f.write(string)
            f.close()

with open(class_names_path, "w") as f:
    for class_name in class_names:
        f.write(class_name + "\n")
    f.close()

    


print("Conversion complete!")

