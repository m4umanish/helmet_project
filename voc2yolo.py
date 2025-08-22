import os
import xml.etree.ElementTree as ET

# Classes for YOLO
classes = ["helmet", "no_helmet"]

# Input/Output paths
input_dir = r"C:\Users\hp\Desktop\Helmet_project\annotations"
output_dir = r"C:\Users\hp\Desktop\Helmet_project\labels"

os.makedirs(output_dir, exist_ok=True)

def convert_annotation(xml_file, output_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    img_w = int(root.find("size").find("width").text)
    img_h = int(root.find("size").find("height").text)

    label_file = os.path.join(output_dir, os.path.basename(xml_file).replace(".xml", ".txt"))
    with open(label_file, "w") as f:
        for obj in root.findall("object"):
            cls_name = obj.find("name").text.lower()

            if cls_name == "helmet":
                cls_id = 0  # Helmet
            elif cls_name == "head":
                cls_id = 1  # No Helmet
            else:
                continue  # Ignore "person" or other classes

            xml_box = obj.find("bndbox")
            x1 = float(xml_box.find("xmin").text)
            y1 = float(xml_box.find("ymin").text)
            x2 = float(xml_box.find("xmax").text)
            y2 = float(xml_box.find("ymax").text)

            # YOLO format
            x = (x1 + x2) / 2.0 / img_w
            y = (y1 + y2) / 2.0 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# Convert all XML files
for xml_file in os.listdir(input_dir):
    if xml_file.endswith(".xml"):
        convert_annotation(os.path.join(input_dir, xml_file), output_dir)

print("✅ Conversion complete. YOLO labels saved in 'labels' folder.")
