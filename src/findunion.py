from numba import njit
import numpy as np
import cv2 as cv


import numpy as np

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size
        self.mode = [''] * size  # Store concatenation mode for each element
    
    def find(self, p):
        if self.parent[p] != p:
            self.parent[p], self.mode[p] = self.find(self.parent[p])
        return self.parent[p], self.mode[p]
    
    def union(self, p, q, mode):
        rootP, modeP = self.find(p)
        rootQ, modeQ = self.find(q)
        if rootP == rootQ: return
        
        # Union by rank
        if self.rank[rootP] > self.rank[rootQ]:
            self.parent[rootQ] = rootP
            self.mode[rootQ] = mode
        elif self.rank[rootP] < self.rank[rootQ]:
            self.parent[rootP] = rootQ
            self.mode[rootP] = mode
        else:
            self.parent[rootQ] = rootP
            self.mode[rootQ] = mode
            self.rank[rootP] += 1

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def concatenate_lines(lines, threshold):
    n = len(lines)
    uf = UnionFind(n)

    # Convert each line's list of points into a NumPy array for efficient computation
    for i in range(n):
        lines[i] = np.array(lines[i])

    # Check all lines against each other
    for i in range(n):
        for j in range(i + 1, n):
            # Check all combinations of endpoints
            combinations = [(lines[i][0], lines[j][0], 'ss'), (lines[i][0], lines[j][-1], 'se'),
                            (lines[i][-1], lines[j][0], 'es'), (lines[i][-1], lines[j][-1], 'ee')]
            for start, end, mode in combinations:
                if calculate_distance(start, end) <= threshold:
                    uf.union(i, j, mode)
                    break  # Union made, no need to check further combinations

    # Construct concatenated lines from sets
    sets = {}
    for i in range(n):
        root, _ = uf.find(i)
        if root not in sets:
            sets[root] = []
        sets[root].append(i)

    # Concatenate all lines in the same set based on mode
    result = []
    for indices in sets.values():
        set_lines = [lines[i] for i in indices]
        concatenated = set_lines[0]
        set_lines.remove(concatenated)
        finished = False
        end_concat = True
        IDX_DISTANCE = 0
        IDX_PROPER_LINE = 1
        IDX_LINE_POSITION = 2
        while not finished:
            best_match = (float('+inf'), None, -1)
            for i, line in enumerate(set_lines):
                anchor = concatenated[-1] if end_concat else concatenated[0]
                normal = (calculate_distance(anchor, line[0]), line, i)

                if normal[0] < best_match[0]:
                    best_match = normal

                revrs = (calculate_distance(anchor, line[-1]), line[::-1], i)

                if revrs[0] < best_match[0]:
                    best_match = revrs

            if best_match[IDX_DISTANCE] < threshold:
                proper_line = np.array(best_match[IDX_PROPER_LINE])
                if end_concat:
                    concatenated = np.concatenate([concatenated, proper_line])
                else:
                    concatenated = np.concatenate([proper_line, concatenated])

                set_lines.pop(best_match[IDX_LINE_POSITION])
            elif end_concat:
                end_concat = False
            else:
                finished = True
        result.append(concatenated)

    return result

if __name__ == '__main__':
    # Example usage
    lines = [
        np.random.randint(0,256,size=(10,2)),
        np.random.randint(0,256,size=(10,2)),
        np.random.randint(0,256,size=(10,2)),
        np.random.randint(0,256,size=(10,2)),
        np.random.randint(0,256,size=(10,2)),
    ]

    img1 = cv.cvtColor(np.zeros((256,256,3), dtype=np.uint8), cv.COLOR_BGR2HSV)
    img2 = cv.cvtColor(np.zeros((256,256,3), dtype=np.uint8), cv.COLOR_BGR2HSV)

    def get_random_color():
        # Generate a random hue value between 0 and 180 degrees
        h = np.random.randint(0, 180)

        # Generate a random saturation and value values between 0 and 255
        s = np.random.randint(100, 200)
        v = np.random.randint(100, 200)

        # Convert the HSV color to BGR using cv2.cvtColor
        return [h,s,v]


    threshold = 10
    concatenated_lines = concatenate_lines(lines, threshold)
    for line in concatenated_lines:
        print(line)

    for line in lines:
        color = get_random_color()
        for i in range(len(line) - 1):
            cv.line(img1, line[i], line[i+1], color=color, thickness=1)

    for line in concatenated_lines:
        color = get_random_color()
        for i in range(len(line) - 1):
            cv.line(img2, line[i], line[i+1], color=color, thickness=1)

    while True:
        cv.imshow("first", cv.resize(img1, (1024,1024), interpolation=cv.INTER_NEAREST))
        cv.imshow("second", cv.resize(img2, (1024,1024), interpolation=cv.INTER_NEAREST))
        k = cv.waitKey(0)
        if k == ord('q'):
            break
