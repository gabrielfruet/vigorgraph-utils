import scipy
import numpy as np
import cv2 as cv

def draw_line(img, line, color):
    for i in range(len(line) - 1):
        img = cv.line(img, line[i], line[i+1], color=color)
    return img

class Seedling:
    COLOR_RAIZ = (0,255,0)
    COLOR_HIPOCOTILO = (0,0,255)
    COLOR_COTILEDONE = (0,255,255)
    def __init__(self, raiz_prim, hipocotilo, cotiledone):
        self.raiz_prims = [raiz_prim] if raiz_prim is not None else []
        self.hipocotilos = [hipocotilo] if hipocotilo is not None else []
        self.cotiledone = cotiledone

    def draw(self, img):
        cv.circle(img, self.cotiledone, radius=5, color=Seedling.COLOR_COTILEDONE)
        for hip in self.hipocotilos:
            img = draw_line(img,hip,Seedling.COLOR_HIPOCOTILO)
            img = draw_line(img, [self.cotiledone, hip[0]], color=(255,127,0))

        for raiz in self.raiz_prims:
            img = draw_line(img,raiz,Seedling.COLOR_RAIZ)
            img = draw_line(img, [self.cotiledone, raiz[0]], color=(255,127,0))

        return img


class SeedlingBuilder:
    def __init__(self):
        self.hipocotilo = None
        self.raiz_prim = None
        self.cotiledone = None

    def set_hipocotilo(self, hipocotilo):
        """
        Position 0 of the hipocotilo should be connected to the cotiledone
        """
        self.hipocotilo = hipocotilo
        return self

    def set_raiz_prim(self, raiz_prim):
        """
        Position 0 of the raiz_prim should be connected to the -1 position of the hipocotilo
        """
        self.raiz_prim = raiz_prim
        return self

    def set_cotiledone(self, cotiledone):
        self.cotiledone = cotiledone
        return self

    def get_hipocotilo(self):
        return self.hipocotilo

    def build(self):
        return Seedling(self.raiz_prim, self.hipocotilo, self.cotiledone)




class SeedlingSolver:
    def __init__(self, raiz_prim_links, hipocotilo_links, cotiledone):
        self.raiz_prim_links = raiz_prim_links
        self.hipocotilo_links = hipocotilo_links
        self.cotiledone = cotiledone

    def match(self):
        MAX_COST = 70
        hipocotilo_bounds = []
        for hip in self.hipocotilo_links:
            hipocotilo_bounds.append(hip[[0,-1]])

        hipocotilo_bounds = np.array(hipocotilo_bounds)
        expnd_hip = hipocotilo_bounds[None, ...]
        expnd_cot = self.cotiledone[:, None, None]
        # rows are cotiledone
        # columns are hipocotilo
        costs = np.linalg.norm(expnd_hip - expnd_cot, axis=-1)
        choosen_bound = np.eye(2)[np.argmin(costs,axis=-1)]
        min_costs = costs[choosen_bound.astype(bool)].reshape(costs.shape[:2])
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(min_costs)
        seedlings_builders = []

        for i,j in zip(row_ind, col_ind):
            if min_costs[i,j] >= MAX_COST: continue
            sdl_builder = SeedlingBuilder().set_cotiledone(self.cotiledone[i])

            if np.argmax(choosen_bound[i,j]) == 0:
                sdl_builder.set_hipocotilo(self.hipocotilo_links[j])
            else:
                sdl_builder.set_hipocotilo(self.hipocotilo_links[j][::-1])

            seedlings_builders.append(sdl_builder)

        choosen_hipocotilos = []

        for sdl in seedlings_builders:
            choosen_hipocotilos.append(sdl.get_hipocotilo()[-1])

        choosen_hipocotilos = np.array(choosen_hipocotilos)

        raiz_prim_bounds = []
        for rp in self.raiz_prim_links:
            raiz_prim_bounds.append(rp[[0,-1]])

        raiz_prim_bounds = np.array(raiz_prim_bounds)
        expnd_rp = raiz_prim_bounds[None, ...]
        expnd_cot = choosen_hipocotilos[:, None, None]
        # rows are choosen hipocotilos
        # columns are raiz prim
        costs = np.linalg.norm(expnd_rp - expnd_cot, axis=-1)
        choosen_bound = np.eye(2)[np.argmin(costs,axis=-1)]
        min_costs = costs[choosen_bound.astype(bool)].reshape(costs.shape[:2])
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(min_costs)

        for i,j in zip(row_ind, col_ind):
            if min_costs[i,j] >= MAX_COST: continue
            sdl_builder = seedlings_builders[i]
            if np.argmax(choosen_bound[i,j]) == 0:
                sdl_builder.set_raiz_prim(self.raiz_prim_links[j])
            else:
                sdl_builder.set_raiz_prim(self.raiz_prim_links[j][::-1])

        return [builder.build() for builder in seedlings_builders]

