# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00_data.ipynb (unless otherwise specified).

__all__ = ['html_vid', 'html_titled_vid', 'html_vids', 'get_annots']

# Cell
from fastai.vision import *
import pandas as pd
from IPython.display import display, HTML

# Cell
def html_vid(fname):
    "Return HTML for video."
    return f'''
    <video width="300" height="250" controls>
    <source src="{fname}" type="video/mp4">
    </video>
    '''

# Cell
def html_titled_vid(fname, title):
    "Return HTML for titled video."
    return f'<div><p>{title}</p><br>{html_vid(fname)}</div>'

# Cell
def html_vids(fnames, titles=None, ncols=3):
    "Return HTML for table of (titled) videos."
    n = len(fnames)
    if titles is None: titles = n * ['']
    assert len(titles) == n
    rs = []
    for i in range(0, n, ncols):
        fs, ts = fnames[i:i+ncols], titles[i:i+ncols]
        xs = (html_titled_vid(f, t) for f,t in zip(fs, ts))
        xs = (f'<td>{x}</td>' for x in xs)
        r = f"<tr>{''.join(xs)}</tr>"
        rs.append(r)
    return f"<table>{''.join(rs)}</table>"

# Cell
def get_annots(SOURCE):
    """
    extract the metadata from all the folders contained in SOURCE.
    """

    files = []
    annots = []

    for i in SOURCE.iterdir(): # iterate over the files in SOURCE
        if i.is_dir() and (i/'metadata.json').is_file(): # Get only the directories
            print(f'Extracting data from the {i.name} folder')
            f = get_files(i, extensions=['.json']) # Extract the metadata
            files.append(f)

            a = pd.read_json(f[0]).T
            a.reset_index(inplace=True)
            a.rename({'index':'fname'}, axis=1, inplace=True)
            a.fname = i.name + '/' + a.fname.astype(str)
            a.loc[a.label=='FAKE', 'original'] = i.name + '/' + a.original[a.label=='FAKE']

            annots.append(a)
    return pd.concat(annots).reset_index(drop=True)