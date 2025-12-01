"Functions to work with file and directories"
# copied from real-time-neuropixels (implemented by Mostafa Safaie)

from pathlib import Path
import datetime
from typing import Union

def find_file(path: str | Path, extension: tuple[str] = ('.raw.kwd',)) -> list[Path]:
    """
    Recursively finds files with the specified extensions within the given path.
    `path` (str or Path): The directory in which to search for files.
    `extension`: A tuple of file extensions e.g., ('.dat', '.prm').
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    # Convert extension to list if it is a string.
    if isinstance(extension, str):
        extension = extension.split()

    # Normalize extensions to ensure they start with a dot.
    normalized_exts = [ext if ext.startswith('.') else '.' + ext for ext in extension]

    found_files = []
    for ext in normalized_exts:
        for file in p.rglob(f"*{ext}"):
            if file.is_file():
                found_files.append(file)
    return found_files


def list_dirs(main_path: Union[str, Path]) -> list[str]:
    "List the names of directories under a path"
    # Create a Path object from the provided path string
    if isinstance(main_path, str):
        p = Path(main_path)
    else:
        p = main_path
    # List all directories in the given path
    directories = [d.name for d in p.iterdir() if d.is_dir()]
    return directories

def list_animals(data_path: Union[str, Path]) -> list[Path]:
    """ List all the animals in the local path
    data_path: where all the animal directories are: /data/raw/
    Return: ['/data/raw/M0123', '/data/raw/M0124']
    """
    return [animal_dir for animal_dir in data_path.glob('M???') if animal_dir.is_dir()]

def list_session_datetime(animal_path: Union[str, Path]) -> tuple[list[datetime.date], list[str]]:
    """List and sort the datetimes of the sessions in the given path
    animal_path: path to the directory containing the session directories: /data/raw/M034/
    Return: - list of datetime objects sorted in ascending order, 
            - list of session names in the format M034_2024_07_12_10_00
    """
    # List all directories in the given path
    session_list = list_dirs(Path(animal_path))
    session_datetime_list = [datetime.datetime.strptime(s[5:],'%Y_%m_%d_%H_%M')
                             for s in session_list]
    session_datetime_list.sort()
    sort_session_list = [f"{Path(animal_path).name}_{s.strftime('%Y_%m_%d_%H_%M')}"
                         for s in session_datetime_list]

    return session_datetime_list, sort_session_list

def get_last_session(animal_path: Union[str, Path]) -> Path:
    """Get the name of the last session for a given animal: M034_2024_07_12_10_00
    animal_path: path to the directory containing the session directories : /data/raw/M034/
    """

    last_session = list_session_datetime(Path(animal_path))[1][-1]

    assert (Path(animal_path) / last_session).is_dir(), f"Session {last_session} not found in {animal_path}"

    return last_session

def get_last_experiment_path(data_path: Union[str, Path], animal: str) -> Path:
    """Get the path to the experiment directory with the latest date
    data_path: path to the directory containing the animal directories -> /data/raw/
    animal: name of the animal -> M034
    Return: path to the experiment directory -> /data/raw/M034/M034_2024_07_12_10_00
    """
    last_session = get_last_session(Path(data_path) / animal)
    last_session_path = Path(data_path) / animal / last_session

    assert last_session_path.is_dir(), f"Session {last_session} not found in {animal}"
    assert datetime.datetime.strptime(last_session[5:],'%Y_%m_%d_%H_%M').date() == datetime.datetime.today().date(), \
        f"Last session found ({last_session}) is not today's date."

    return last_session_path

def find_lastest_experiment_path(data_path: Union[str, Path]) -> Path:
    """Get the path to the experiment directory with the latest date across all the animals
    data_path: path to the directory containing the animal directories -> /data/raw/
    Return: path to the experiment directory -> /data/raw/M034/M034_2024_07_12_10_00
    """
    animal_list = list_animals(data_path)
    latest_session_date = datetime.datetime.min
    out = Path('')
    for animal_path in animal_list:
        last_session = get_last_session(animal_path)
        last_session_date = datetime.datetime.strptime(last_session[5:],'%Y_%m_%d_%H_%M')
        if last_session_date > latest_session_date:
            out = animal_path / last_session
            latest_session_date = last_session_date
    return out
