import numpy as np


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    m = 6367 * c * 1000
    return m

def get_perp( X1, Y1, X2, Y2, X3, Y3):
    """************************************************************************************************ 
    Purpose - X1,Y1,X2,Y2 = Two points representing the ends of the line segment
              X3,Y3 = The offset point 
    'Returns - X4,Y4 = Returns the Point on the line perpendicular to the offset or None if no such
                        point exists
    '************************************************************************************************ """
    XX = X2 - X1 
    YY = Y2 - Y1 
    ShortestLength = ((XX * (X3 - X1)) + (YY * (Y3 - Y1))) / ((XX * XX) + (YY * YY))
    print(ShortestLength)
    X4 = X1 + XX * ShortestLength 
    Y4 = Y1 + YY * ShortestLength
    if X4 < X2 and X4 > X1 and Y4 < Y2 and Y4 > Y1:
        return X4,Y4
    return None
