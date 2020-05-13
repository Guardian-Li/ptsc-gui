import numpy as np


class Point(object):
    """Point are two-dimension"""

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Segment(object):
    """the 2 points p1 and p2 are unordered"""

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2


class Line(object):
    """p1 and p2 are 2 points in straight line"""

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2


class Vector(object):
    """start and end are two points"""

    def __init__(self, start, end):
        self.x = end.x - start.x
        self.y = end.y - start.y


def pointDistance(p1, p2):
    """calculate the distance between point p1 and p2"""

    # v: a Vector object
    v = Vector(p1, p2)

    # translate v to a ndarray object
    t = np.array([v.x, v.y])

    # calculate the inner product of ndarray t
    return float(np.sqrt(t @ t))


def pointToLine(C, AB):
    """calculate the shortest distance between point C and straight line AB, return: a float value"""

    # two Vector object
    vector_AB = Vector(AB.p1, AB.p2)
    vector_AC = Vector(AB.p1, C)

    # two ndarray object
    tAB = np.array([vector_AB.x, vector_AB.y])
    tAC = np.array([vector_AC.x, vector_AC.y])

    # vector AD, type: ndarray
    tAD = ((tAB @ tAC) / (tAB @ tAB)) * tAB

    # get point D
    Dx, Dy = tAD[0] + AB.p1.x, tAD[1] + AB.p1.y
    D = Point(Dx, Dy)

    return pointDistance(D, C)


def pointInLine(C, AB):
    """determine whether a point is in a straight line"""

    return pointToLine(C, AB) < 1e-9


def pointInSegment(C, AB):
    """determine whether a point is in a segment"""

    # if C in segment AB, it first in straight line AB
    if pointInLine(C, Line(AB.p1, AB.p2)):
        return min(AB.p1.x, AB.p2.x) <= C.x <= max(AB.p1.x, AB.p2.x)
    return False


def linesAreParallel(l1, l2):
    """determine whether 2 straight lines l1, l2 are parallel"""

    v1 = Vector(l1.p1, l1.p2)
    v2 = Vector(l2.p1, l2.p2)

    return abs((v1.y / v1.x) - (v2.y / v2.x)) < 1e-9


def crossProduct(v1, v2):
    """calculate the cross product of 2 vectors"""

    # v1, v2 are two Vector object
    return v1.x * v2.y - v1.y * v2.x


def segmentsIntersect(s1, s2):
    """determine whether 2 segments s1, s2 intersect with each other"""

    v1 = Vector(s1.p1, s1.p2)
    v2 = Vector(s2.p1, s2.p2)

    t1 = Vector(s1.p1, s2.p1)
    t2 = Vector(s1.p1, s2.p2)

    d1 = crossProduct(t1, v1)
    d2 = crossProduct(t2, v1)

    t3 = Vector(s2.p1, s1.p1)
    t4 = Vector(s2.p1, s1.p2)

    d3 = crossProduct(t3, v2)
    d4 = crossProduct(t4, v2)

    if d1 * d2 < 0 and d3 * d4 < 0:
        return True

    if d1 == 0:
        return pointInSegment(s2.p1, s1)
    elif d2 == 0:
        return pointInSegment(s2.p2, s1)
    elif d3 == 0:
        return pointInSegment(s1.p1, s2)
    elif d4 == 0:
        return pointInSegment(s1.p2, s2)

    return False

if __name__ == "__main__":
    p1 = Point(0, 0)
    p2 = Point(2, 2)

    # 计算点p1, p2之间的距离
    print(pointDistance(p1, p2))  # >>> 2.82...

    # 通过p1, p2分别建立一个线段和一个直线
    l1 = Line(p1, p2)
    s1 = Segment(p1, p2)

    # 设点p3，显然p3在l1上，却不在l2上
    p3 = Point(3, 3)

    print(pointInLine(p3, l1))  # >>> True
    print(pointInSegment(p3, s1))  # >>> False

    # 设点p4, p5得到一条与l1平行的直线l2
    p4 = Point(0, 1)
    p5 = Point(2, 3)

    l2 = Line(p4, p5)

    print(linesAreParallel(l1, l2))  # >>> True

    # 计算p4到l1的距离
    print(pointToLine(p4, l1))  # >>> 0.7071067...

    # 设两条线段s2, s3
    s2 = Segment(Point(0, 2), Point(5, -1))
    s3 = Segment(Point(1, 0.7), Point(5, -1))

    # s2与s1相交；s3与s1不相交
    print(segmentsIntersect(s2, s1))  # >>> True
    print(segmentsIntersect(s3, s1))  # >>> False
