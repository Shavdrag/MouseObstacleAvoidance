
import xml.etree.ElementTree as ET


def generate_whiskers_xml(side, pos, num_links, stiffness, damping = 0):
    root = [0.00, -0.05, -0.01]


    name = side + " " + pos

    fang = 30
    hang = 60

    xoff_h = 0.012
    yoff = 0.01

    xoff_f = 0.005

    if side == "right":

        #root[0] -= 0.001

        if pos == "hind":
            ang = -hang
            root[0] -= xoff_h
            root[1] += yoff
        else:
            root[0] -= xoff_f
            ang = -fang

    if side == "left":

        #root[0] += 0.003
        if pos == "hind":
            root[0] += xoff_h
            root[1] += yoff
            ang = hang
        else:
            root[0] += xoff_f
            ang = fang




    xml = f'<body name="whisker {name} 1" pos="{root[0]} {root[1]} {root[2]}" euler="0 0 {ang}">\n'

    # CHANGE AXIS IF ANGLE DIFFERENT???
    for i in range(1, num_links + 1):
        x = 0
        y = -0.005*(i-1)
        x2=- 0
        y2=- 0.005*(i)
        xml += "\n"+"    "*(i) + f'<joint name="whisker {name} {i}1" pos="{x} {y} -0" type="hinge" axis = "1 0 0" stiffness = "{stiffness*(num_links-i)}" armature="0.00065" damping = "{damping}" ref = "0"/>'
        xml += "\n"+"    " * (i) + f'<joint name="whisker {name} {i}2" pos="{x} {y} -0" type="hinge" axis = "0 0 1"  stiffness = "{stiffness*(num_links-i)}" armature="0.00065" damping = "{damping}" ref = "0"/>'

        xml += "\n"+"    " * (i) + f'<geom name="whisk_{name} {i}" type="capsule" fromto="{x} {y} -0 {x2} {y2} -0" size="0.001" mass = "0.0000001" rgba="0.5 0.5 0.5 1"/>'

        if i < num_links:
            xml += "\n"+"    "*(i) + f'<body name="whisker {name} {i+1}" pos="0 0 0" euler="0 0 0">'
        else:
            xml += '\n'

    for i in range(1, num_links + 1):
        xml += "\n" + "    "* (num_links-i)+"</body>"

    return xml


def replace_xml_content(xml_file_path, new_content, name):
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Convert the new content to an XML element
    new_element = ET.fromstring(new_content)

    # Find the "mouse" body element
    mouse_body = root.find(".//body[@name='mouse_head']")

    # Find the "whisker left hind 1" body element within the "mouse" body
    old_element = mouse_body.find(f".//body[@name='whisker {name} 1']")

    # Replace the old element with the new
    # Get the index of the old element

    index = list(mouse_body).index(old_element)
    # HIER PROBLEM, FINDET INDEX NICHT



    # Remove the old element
    mouse_body.remove(old_element)

    # Insert the new element at the position of the old one
    mouse_body.insert(index, new_element)

    # Write the modified XML back to the file
    tree.write(xml_file_path)


def modify(stif,damp):
    print("modify")
    for i in range(20):
        model = f"dynamic_4l_maze_env{i}.xml"
        # Load the XML file
        tree = ET.parse('./models/'+model)
        root = tree.getroot()

        for size in root.iter('size'):
            size.set('njmax', '2000')
            size.set('nconmax', '2000')

        if 1:
        # Iterate over all joints in the XML
            for joint in root.iter('joint'):
                # Check if the joint is a whisker joint
                if 'whisker' in joint.attrib['name']:
                    # Set the stiffness to the desired value
                    joint.attrib['stiffness'] = str(stif)
                    joint.attrib['damping'] = str(damp)


        if 0:
            for geom in root.iter('geom'):
                # Check if the 'name' attribute exists
                if 'name' in geom.attrib:
                    # Check if the geom is a whisker geom
                    if 'whisk_left' in geom.attrib['name'] or 'whisk_right' in geom.attrib['name']:
                        # Set the mass to the desired value
                        geom.attrib['mass'] = '0.0001'

        # Write the modified XML back to the file
        tree.write('./models/'+model)


def remove_mass_attributes(element):
    # Remove attributes with "mass" from <geom> elements
    if element.tag == "geom":
        attributes = element.attrib.copy()
        for attr in attributes:
            if "mass" in attr:
                del element.attrib[attr]

    # Recursively remove attributes from child elements
    for child in element:
        remove_mass_attributes(child)

if 0:
    tree = ET.parse("/home/joris/PycharmProjects/RatWhisker/models/head.xml")
    root = tree.getroot()

    # Remove attributes with "mass" from <geom> elements
    remove_mass_attributes(root)

    # Save the modified XML file
    tree.write("/home/joris/PycharmProjects/RatWhisker/models/head.xml")

#modify(0.005,0.00)
if 1:
    num_links = 10
    stiffness = 0.005
    damping = 0.001

    side = "left"
    pos = "hind"
    sides = ["left", "right"]
    pos = ["hind", "front"]

    for i in sides:
        for j in pos:
            if j=="hind": num_links=10
            else: num_links=10

            generated_xml = generate_whiskers_xml(i, j, num_links, stiffness, damping)
            #print(generated_xml)
            replace_xml_content("./models/dynamic_4l_maze_env4.xml", generated_xml, i +" "+j)



# Generate the new XML content
import os

# Read the template from the file
if 0:
    with open("./models/dynamic_4l.xml", "r") as f:
        template = f.read()

    # Replace the maze include line for each maze
    for i in range(21):  # for mazes 0-20
        new_xml = template.replace('<include file="test_environment/maze0.xml"/>',
                                   f'<include file="test_environment/maze{i}.xml"/>')

        # Save the new XML file
        with open(f"./models/dynamic_4l_maze_env{i}.xml", "w") as f:
            f.write(new_xml)