import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def constant_vars(data, groups):
    groups = data[groups]
    constant_cols = data.columns[data.apply(lambda col: len(col.unique()) == len(groups.unique()))]
    return data.drop_duplicates(subset=groups)[constant_cols]

def inverse_gnomonic(x, y, center_x, center_y, distance, unit_size=1):
    x = (x - center_x) * unit_size / distance
    y = (y - center_y) * unit_size / distance

    rho = np.sqrt(x**2 + y**2)
    c = np.arctan(rho)

    sin_c = np.sin(c)
    lat = np.arcsin(y * sin_c / rho) * 180/np.pi
    lon = np.arctan2(x * sin_c, rho * np.cos(c)) * 180/np.pi
    return pd.DataFrame({'lat': lat, 'lon': lon})



def extract_alignment(d, path):
    path = np.uint(path)
    sop = []
    top = []
    cost = []
    i = d.shape[0] - 1
    j = d.shape[1] - 1
    
    while i > 0 or j > 0:
        path = np.int32(path)
        if path[i, j] == 2:
            sop.insert(0, i - 1)
            top.insert(0, j - 1)
            cost.insert(0, d[i, j] - d[i - 1, j - 1])
            i -= 1
            j -= 1
        elif path[i, j] == 1:
            sop.insert(0, np.nan)
            top.insert(0, j - 1)
            cost.insert(0, d[i, j] - d[i, j - 1])
            j -= 1
        elif path[i, j] == 0:
            sop.insert(0, i - 1)
            top.insert(0, np.nan)
            cost.insert(0, d[i, j] - d[i - 1, j])
            i -= 1
    alignment_df = pd.DataFrame({"s": sop, "t": top, "cost": cost})
    return alignment_df


def rscasim(s, t, center_x, center_y, viewing_distance, unit_size, modulator=0.83):
    s = s[['fixation_duration', 'fixation_position_x', 'fixation_position_y']]
    t = t[['fixation_duration', 'fixation_position_x', 'fixation_position_y']]
    s = pd.concat([s, inverse_gnomonic(s['fixation_position_x'], s['fixation_position_y'], center_x, center_y, viewing_distance, unit_size)], axis=1)
    t = pd.concat([t, inverse_gnomonic(t['fixation_position_x'], t['fixation_position_y'], center_x, center_y, viewing_distance, unit_size)], axis=1)

    d = np.zeros((len(s) + 1, len(t) + 1))
    d[:, 0] = np.cumsum(list(s['fixation_duration']) + [0])
    d[0, :] = np.cumsum(list(t['fixation_duration'])+ [0])


    path = np.zeros((len(s) + 1, len(t) + 1))
    path[1:, 0] = 1
    path[0, 1:] = 2
    path[0, 0] = 0


    for i in range(1, len(s) + 1):
        for j in range(1, len(t) + 1):
            sa = s['lon'][i-1] / (180/np.pi)
            ta = t['lon'][j-1] / (180/np.pi)
            sb = s['lat'][i-1] / (180/np.pi)
            tb = t['lat'][j-1] / (180/np.pi)
            angle = np.arccos(np.sin(sb) * np.sin(tb) + np.cos(sb) * np.cos(tb) * np.cos(sa - ta)) * (180/np.pi)
            mixer = modulator**angle
            cost = abs(t['fixation_duration'][j-1] - s['fixation_duration'][i-1]) * mixer + (t['fixation_duration'][j-1] + s['fixation_duration'][i-1]) * (1 - mixer)
            operations = [d[i-1, j] + s['fixation_duration'][i-1], d[i, j-1] + t['fixation_duration'][j-1], d[i-1, j-1] + cost]
            path[i, j] = np.argmin(operations)
            d[i, j] = min(operations)

    a = extract_alignment(d, path)
    # assert np.sum(a['cost']) == d[-1, -1]
    return(d, path, a)