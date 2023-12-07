import os
import xml.etree.ElementTree as ET


def convert_xml_to_yolo(xml_file, classes, output_folder):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_filename = root.find('filename').text
    image_width = int(root.find('size').find('width').text)
    image_height = int(root.find('size').find('height').text)

    yolo_labels = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in classes:
            continue

        class_id = classes.index(class_name)
        bbox = obj.find('bndbox')

        x_center = (int(bbox.find('xmin').text) + int(bbox.find('xmax').text)) / 2 / image_width
        y_center = (int(bbox.find('ymin').text) + int(bbox.find('ymax').text)) / 2 / image_height
        w = (int(bbox.find('xmax').text) - int(bbox.find('xmin').text)) / image_width
        h = (int(bbox.find('ymax').text) - int(bbox.find('ymin').text)) / image_height

        yolo_labels.append(f"{class_id} {x_center} {y_center} {w} {h}")

    output_path = os.path.join(output_folder, os.path.splitext(image_filename)[0] + ".txt")

    with open(output_path, 'w') as out_file:
        for line in yolo_labels:
            out_file.write(line + '\n')

# Define your classes
classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']  # Add your own class names

# Path to the directory containing XML files
xml_dir = 'D:/pycharm code/ultralytics-main/data/NEU-DET/train/annatation'
output_dir = 'D:/pycharm code/ultralytics-main/data/NEU-DET/train/labels'  # Output directory for YOLO format files

test_xml_dir = 'D:/pycharm code/ultralytics-main/data/NEU-DET/test/annatation'
test_output_dir = 'D:/pycharm code/ultralytics-main/data/NEU-DET/test/labels'  # Output directory for YOLO format files

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

# Convert all XML files to YOLO format and save to the output directory
for xml_file in os.listdir(xml_dir):
    if xml_file.endswith('.xml'):
        xml_path = os.path.join(xml_dir, xml_file)
        convert_xml_to_yolo(xml_path, classes, output_dir)
# 测试集
for xml_file in os.listdir(test_xml_dir):
    if xml_file.endswith('.xml'):
        xml_path = os.path.join(test_xml_dir, xml_file)
        convert_xml_to_yolo(xml_path, classes, test_output_dir)
