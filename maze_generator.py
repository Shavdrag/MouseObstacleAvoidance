import random
import xml.etree.ElementTree as ET
from xml.dom import minidom


def generate_maze_xml(num_cubes, maze_area, maze_size, obstacle_size=0.3):
    # Create the root XML element

    a = random.uniform(0.85, 1.1)
    maze_area = [-a, a, -a, a ]

    body = ET.Element("body")
    body.set("name", "maze")
    body.set("pos", "0.2 -0.4 0.0")

    # Add the base plate
    base_plate = ET.SubElement(body, "geom")
    base_plate.set("name", "base_plate")
    base_plate.set("type", "box")
    base_plate.set("pos", "0 -0.747 0")
    base_plate.set("size", "2 2.0 0.005")
    base_plate.set("rgba", "0.8 0.9 0.9 1")

    min = 0.10
    max = 0.35 # obstacle size  but

    dist = 0.4
    free_square = [-dist, dist, -dist, dist]

    #print(" maze :   ",maze_area[0], "   ",maze_area[1])
    #print("free:   ", free_square[0], "    ", free_square[1])
    for count in range(num_cubes):
        while True:
            x_size = random.uniform(min, max)
            y_size = random.uniform(min, max)

            x_pos = random.uniform(maze_area[0], maze_area[1])
            y_pos = random.uniform(maze_area[2], maze_area[3])

            # Check if the block is outside the free area
            if not (free_square[0] <= x_pos <= free_square[1] and free_square[2] <= y_pos <= free_square[3]):
                cube = ET.SubElement(body, "geom")
                cube.set("name", f"block{count + 1}")
                cube.set("type", "box")
                cube.set("euler", f"0 0 {random.uniform(0, 45)}")
                cube.set("pos", f"{x_pos} {y_pos} 0.05")
                cube.set("size", f"{x_size/2} {y_size/2} 0.1")
                cube.set("rgba", "0.95 0.95 0.95 1")
                break
    border = False
    if maze_size>0:
        cube = ET.SubElement(body, "geom")
        cube.set("name", f"blockX1")
        cube.set("type", "box")
        cube.set("euler", f"0 0 0")
        cube.set("pos", f"{0} {maze_size} 0.05")
        cube.set("size", f"{maze_size} {0.1} 0.1")
        cube.set("rgba", "0.95 0.95 0.95 1")
        cube = ET.SubElement(body, "geom")
        cube.set("name", f"blockX2")
        cube.set("type", "box")
        cube.set("euler", f"0 0 0")
        cube.set("pos", f"{0} {-maze_size} 0.05")
        cube.set("size", f"{maze_size} {0.1} 0.1")
        cube.set("rgba", "0.95 0.95 0.95 1")

        cube = ET.SubElement(body, "geom")
        cube.set("name", f"blockX3")
        cube.set("type", "box")
        cube.set("euler", f"0 0 0")
        cube.set("pos", f"{maze_size} {0} 0.05")
        cube.set("size", f"{0.1} {maze_size} 0.1")
        cube.set("rgba", "0.95 0.95 0.95 1")
        cube = ET.SubElement(body, "geom")
        cube.set("name", f"blockX4")
        cube.set("type", "box")
        cube.set("euler", f"0 0 0")
        cube.set("pos", f"{-maze_size} {0} 0.05")
        cube.set("size", f"{0.1} {maze_size} 0.1")
        cube.set("rgba", "0.95 0.95 0.95 1")
    # Generate XML string
    xml_str = ET.tostring(body, encoding='unicode')

    return xml_str

def change_mazes(num_cubes, maze_area, size):
# Use the function
# [x_min, x_max, y_min, y_max]


    for i in range(20):

        maze_xml = generate_maze_xml(num_cubes, maze_area,size)
        file_path = f"./models/test_environment/maze{i}.xml"

        with open(file_path, 'w') as file:
            file.write(maze_xml)




def change_orientation(orientation):
    # Create the root XML element
    body = ET.Element("body")
    body.set("name", "flat_obstacle")
    body.set("pos", "0.2 -0.4 0.0")

    # Add the base plate
    base_plate = ET.SubElement(body, "geom")
    base_plate.set("name", "base_plate")
    base_plate.set("type", "box")
    base_plate.set("pos", "0 -0.747 0")
    base_plate.set("size", "2 2.0 0.005")
    base_plate.set("rgba", "0.8 0.9 0.9 1")

    min = 0.005
    max = 0.25

    x_pos = -0.05
    y_pos = -0.6

    cube = ET.SubElement(body, "geom")
    cube.set("name", f"block1")
    cube.set("type", "box")
    cube.set("euler", f"0 0 {orientation}")
    cube.set("pos", f"{x_pos} {y_pos} 0.05")
    cube.set("size", f"{min} {max} 0.1")
    cube.set("rgba", "0.95 0.95 0.95 1")

    xml_str = ET.tostring(body, encoding='unicode')

    return xml_str

def create_flat_obstacle_with_hinge():
    # Create the root XML element
    body = ET.Element("body")
    body.set("name", "flat_obstacle")
    body.set("pos", "0.2 -0.4 0.0")

    # Add the base plate
    base_plate = ET.SubElement(body, "geom")
    base_plate.set("name", "base_plate")
    base_plate.set("type", "box")
    base_plate.set("pos", "0 -0.747 0")
    base_plate.set("size", "2 2.0 0.005")
    base_plate.set("rgba", "0.8 0.9 0.9 1")

    # Define the size parameters for the cube
    min_size = 0.005
    max_size = 0.25

    # Define the position for the cube
    x_pos = -0.05
    y_pos = -0.6

    # Add a hinge joint for rotation around the z-axis
    hinge_joint = ET.SubElement(body, "joint")
    hinge_joint.set("name", "hinge_for_rotation")
    hinge_joint.set("type", "hinge")
    hinge_joint.set("axis", "0 0 1")  # z-axis
    hinge_joint.set("pos", f"{x_pos} {y_pos} 0.05")  # Position of the hinge joint
    hinge_joint.set("range", "-180 180")  # Allowable range of rotation in degrees

    rotating_part = ET.SubElement(body, "body")
    rotating_part.set("name", "rotating_part")
    rotating_part.set("pos", "0 0 0")  # Position relative to the parent body

    # Add the cube geom
    cube = ET.SubElement(rotating_part, "geom")
    cube.set("name", "block1")
    cube.set("type", "box")
    cube.set("pos", f"{x_pos} {y_pos} 0.05")
    cube.set("size", f"{min_size} {max_size} 0.1")
    cube.set("rgba", "0.95 0.95 0.95 1")


    # Convert the XML structure to a string with pretty print format
    xml_str = ET.tostring(body, encoding='unicode', method='xml')
    xml_pretty = minidom.parseString(xml_str).toprettyxml(indent="   ")

    return xml_pretty

def generate_obstacles(start):


    for i in range(0,10):
        orient = i*10 - start
        orient_num = i*10
        file_path = f"./models/test_environment/orientation{orient_num}.xml"
        xmlfile = change_orientation(orient)
        with open(file_path, 'w') as file:
            file.write(xmlfile)


if 0:
    file_path = f"./models/test_environment/hingeorientation.xml"
    xmlfile = create_flat_obstacle_with_hinge()
    with open(file_path, 'w') as file:
        file.write(xmlfile)


