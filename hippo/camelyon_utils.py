from __future__ import annotations

import xml.etree.ElementTree as ET


def load_cam16_annot(xml_path) -> list[list[tuple[float, float]]]:
    """Load a tumor annotation from CAMELYON16.

    Parameters
    ----------
    xml_path : str or Path
        Path to XML annotation file.

    Returns
    -------
    A nested list of tuples containing XY coordinates.
    """
    with open(xml_path) as f:
        tree = ET.parse(f)
        root = tree.getroot()
    # Get the number of coordinates lists inside the xml file
    count = 0
    for _ in root.findall(".//Annotation"):
        count += 1
    # Make a list of tuples containing the coordinates
    temp = []
    for Coordinate in root.findall(".//Coordinate"):
        order = float(Coordinate.get("Order"))  # type: ignore
        x = float(Coordinate.get("X"))  # type: ignore
        y = float(Coordinate.get("Y"))  # type: ignore
        temp.append((order, x, y))
    # Separate list of tuples into lists depending on how many segments are annotated
    coordinates: list[list[tuple[float, float]]] = [[] for i in range(count)]
    i = -1
    for j in range(len(temp)):
        if temp[j][0] == 0:
            i += 1
        x = temp[j][1]
        y = temp[j][2]
        coordinates[i].append((x, y))
    return coordinates
