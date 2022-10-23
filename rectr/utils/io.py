import datetime
import os
import random
import re
import shutil
import string

import yaml


def printv(string, verbose, **kwargs):
    """ Calls the print() function only if the verbose parameter is True.
    
    Args:
        string: The text to be printed
        verbose: False to disable printing, or True to print
        **kwargs: An arbitrary number of arguments to be interpreted in the
                  same way as the print() function. """
    if verbose:
        print(string, **kwargs)


def ls(path, exts=None, ignore_dot_underscore=True):
    """ Lists the directory and returns it sorted.

    Only the files with extensions in `exts` are kept. The x should match
    the x of Linux command "ls". It wraps os.listdir() which is not
    guaranteed to produce alphanumerically sorted items.

    Args:
        path (str): Absolute or relative path to list.
        exts (str or list of str or None): Extension(s). If None, files with
            any extension are listed. Each e within `exts` can (but does
            not have to) start with a '.' character. E.g. both
            '.tiff' and 'tiff' are allowed.
        ignore_dot_underscore (bool): Whether to ignore files starting with
            '._' (usually spurious files appearing after manipulating the
            linux file system using sshfs)

    Returns:
        list of str: Alphanumerically sorted list of files contained in
        directory `path` and having extension `ext`.
    """
    if isinstance(exts, str):
        exts = [exts]

    files = [f for f in sorted(os.listdir(path))]

    if exts is not None:
        # Include patterns.
        extsstr = ''
        for e in exts:
            extsstr += ('.', '')[e.startswith('.')] + '{}|'.format(e)
        patt_ext = r'.*({})$'.format(extsstr[:-1])
        re_ext = re.compile(patt_ext)

        # Exclude pattern.
        patt_excl = '^/'
        if ignore_dot_underscore:
            patt_excl = '^\._'
        re_excl = re.compile(patt_excl)

        files = [f for f in files if re_ext.match(f) and not re_excl.match(f)]

    return files


def load_conf(conf_file, copy_to=None):
    """ Load the training config.

    The config is loaded from the file specified in the `conf_file` argument.
    If specified, it is also copied to new directory created using an item
    'output_path' from the config and the `copy_to` argument.

    Args:
        conf_file (str): Path to the configuration file.
        copy_to (str|None): Path to directory to copy the configuration file to.
                            If None, the file is not copied.

    Returns:
        dict: Loaded configuration.
        str|None: Path to the output directory for this training run.
    """
    with open(conf_file, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    output_path = None
    if copy_to is not None:
        output_path = os.path.join(conf["checkpoint_path"], copy_to)

        # Create train run dir.
        if os.path.exists(output_path):
            output_path = unique_dir_name(output_path)

        os.makedirs(output_path, exist_ok=True)

        # Save config.
        shutil.copy(conf_file, os.path.join(output_path, os.path.basename(conf_file)))

    return conf, output_path


def unique_dir_name(d):
    """ Checks if the `dir` already exists and if so, generates a new name
     by adding current system time as its suffix. If it is still duplicate,
     it adds a random string as a suffix and makes sure it is unique. If
     `dir` is unique already, it is not changed.

    Args:
        d (str): Absolute path to `dir`.

    Returns:
        str: Unique directory name.
    """
    unique_dir = d

    if os.path.exists(d):
        # Add time suffix.
        dir_name = add_time_suffix(d, keep_extension=False)

        # Add random string suffix until the file is unique in the folder.
        unique_dir = dir_name
        while os.path.exists(unique_dir):
            unique_dir += add_random_suffix(unique_dir, keep_extension=False)

    return unique_dir


def add_time_suffix(name, keep_extension=True):
    """ Adds the current system time suffix to the file name.
    If `keep_extension`, then the suffix is added before the extension
    (including the ".") if there is any.

    Args:
        name (str): File name.
        keep_extension (bool): Add the suffix before the extension?

    Returns:
        str: New file name.
    """

    # Get time suffix.
    time_suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    # Generate new name.
    if keep_extension:
        n, e = split_name_ext(name)
        new_name = n + '_' + time_suffix + ('', '.{}'.format(e))[len(e) > 0]
    else:
        new_name = name + '_' + time_suffix

    return new_name


def add_random_suffix(name, length=5, keep_extension=True):
    """ Adds the random string suffix of the form '_****' to a file name,
    where * stands for an upper case ASCII letter or a digit.
    If `keep_extension`, then the suffix is added before the extension
    (including the ".") if there is any.

    Args:
        name (str): File name.
        length (int32): Length of the suffix: 1 letter for underscore,
            the rest for alphanumeric characters.
        keep_extension (bool): Add the suffix before the extension?

    Returns:
        str: New name.
    """
    # Check suffix length.
    if length < 2:
        print('[WARNING] Suffix must be at least of length 2, '
              'using "length = 2"')

    # Get random string suffix.
    s = ''.join(random.choice(string.ascii_uppercase + string.digits)
                for _ in range(length - 1))

    # Generate new name.
    if keep_extension:
        n, e = split_name_ext(name)
        new_name = n + '_' + s + ('', '.{}'.format(e))[len(e) > 0]
    else:
        new_name = name + '_' + s

    return new_name


def split_name_ext(fname):
    """ Splits the file name to its name and extension.

    Args:
        fname (str): File name without suffix (and without '.').

    Returns:
        str: Name without the extension.
        str: Extension.
    """
    parts = fname.rsplit('.', 1)
    name = parts[0]

    if len(parts) > 1:
        ext = parts[1]
    else:
        ext = ''

    return name, ext
