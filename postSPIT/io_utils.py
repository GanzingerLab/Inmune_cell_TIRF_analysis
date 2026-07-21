from glob import glob
from spit import tools
import yaml
def get_nm2px(folder):
    """
    Get nanometers-per-pixel scaling factor from result.txt.

    Returns
    -------
    float
        Nanometers per pixel.

    Raises
    ------
    ValueError
        If the microscope source is unknown.
    """
    result_files = glob(folder + '/**/*result.txt', recursive=True)
    if not result_files:
        raise FileNotFoundError("No result.txt file found in the folder.")
    result_txt = tools.read_result_file(result_files[0])
    if result_txt['Computer'] == 'ANNAPURNA': 
        return 90.16
    elif result_txt['Computer'] == 'K2-BIVOUAC':
        return 108
    else:
        raise ValueError(f"Unknown microscope source: {result_txt['Computer']}")

def get_time_interval(folder):
    """
    Retrieve the time interval (dt) between frames from result.txt.

    Returns
    -------
    float
        Time interval in seconds.

    Raises
    ------
    FileNotFoundError
        If no result.txt file is found in the folder.
    """
    # Extract dt (frame interval) from result.txt
    result_files = glob(folder + '/**/**result.txt', recursive=True)
    if not result_files:
        raise FileNotFoundError("No result.txt file found in the folder.")
    with open(result_files[0], 'r') as f:
        resultLines = f.readlines()

    if tools.find_string(resultLines, 'Interval'): 
        interval = tools.find_string(resultLines, 'Interval').split(":")[-1].strip()
        if interval.split(" ")[-1] == 'sec':
            dt = 1.0 * float(interval.split(" ")[0])
        elif interval.split(" ")[-1] == 'ms':
            dt = 0.001 * float(interval.split(" ")[0])
    else:
        dtStr = tools.find_string(resultLines, 'Camera Exposure')[17:-1]
        dt = 0.001 * float((''.join(c for c in dtStr if (c.isdigit() or c == '.'))))
    return dt

def openyaml(name):
    """
    Open a YAML file or list of YAML files and merge all YAML documents.

    Parameters
    ----------
    name : list[str] or str
        Path to a YAML file, or list of matching YAML files.

    Returns
    -------
    dict or bool
        Parsed YAML data. Returns False if no file is provided.
    """
    if not name:
        return False

    if isinstance(name, (list, tuple)):
        yaml_path = name[0]
    else:
        yaml_path = name

    with open(yaml_path, "r") as file:
        data_parts = list(yaml.safe_load_all(file))

    data = {}
    for part in data_parts:
        if part:
            data.update(part)

    return data