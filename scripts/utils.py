import functools
import xml.etree.ElementTree as et
import openslide


@functools.cache
def parse_xml(xml_path):
    with open(xml_path, "rt") as f:
        tree = et.parse(f)
        root = tree.getroot()
    # Get the number of coordinates lists inside the xml file
    count = 0
    for _ in root.findall(".//Annotation"):
        count += 1
    # Make a list of tuples containing the coordinates
    temp = []
    for Coordinate in root.findall(".//Coordinate"):
        order = float(Coordinate.get("Order"))
        x = float(Coordinate.get("X"))
        y = float(Coordinate.get("Y"))
        temp.append((order, x, y))
    # Separate list of tuples into lists depending on how many segments are annotated
    coordinates = [[] for i in range(count)]
    i = -1
    for j in range(len(temp)):
        if temp[j][0] == 0:
            i += 1
        x = temp[j][1]
        y = temp[j][2]
        coordinates[i].append((x, y))
    return coordinates


@functools.cache
def get_mpp(slide_id, wsi_dir=None):
    if wsi_dir is None:
        wsi_dir = os.environ.get("WSI_DIR", "./data/wsis")
    with openslide.OpenSlide(
        os.path.join(wsi_dir, f"{slide_id}.tif")
    ) as o:
        mppx = float(o.properties[openslide.PROPERTY_NAME_MPP_X])
        mppy = float(o.properties[openslide.PROPERTY_NAME_MPP_Y])
        return (mppx + mppy) / 2
