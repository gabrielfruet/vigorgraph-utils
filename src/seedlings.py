import scipy
import numpy as np
import cv2 as cv
from utils import draw_line


class Seedling:
    COLOR_RAIZ = (0,255,0)
    COLOR_HIPOCOTILO = (0,0,255)
    COLOR_COTILEDONE = (0,255,255)
    def __init__(self, raiz_prim, hipocotilo, cotiledone):
        if hipocotilo is not None and raiz_prim is not None:
            hip_bound = hipocotilo[[0,-1]]
            rp_bound = raiz_prim[[0,-1]]
            dists = np.zeros((2,2))
            for i in range(2):
                for j in range(2):
                    dist = np.linalg.norm(hip_bound[i] - rp_bound[j])
                    dists[i,j] = dist

            flat_index = np.argmin(dists)
            row,col = np.unravel_index(flat_index, dists.shape)

            if col == 0:
                raiz_prim = np.r_[[hipocotilo[-row]], raiz_prim]
            elif col == 1:
                raiz_prim = np.r_[raiz_prim, [hipocotilo[-row]]]

            if row == 0:
                hipocotilo = np.r_[[raiz_prim[-col]], hipocotilo]
            elif row == 1:
                hipocotilo = np.r_[hipocotilo, [raiz_prim[-col]]]

        self.raiz_prim = raiz_prim 
        self.hipocotilo = hipocotilo 
        self.cotiledone = cotiledone

    def draw(self, img):
        cv.circle(img, self.cotiledone, radius=5, color=Seedling.COLOR_COTILEDONE)

        for line,color in zip([self.hipocotilo, self.raiz_prim], [Seedling.COLOR_RAIZ, Seedling.COLOR_HIPOCOTILO]):
            if line is None: continue
            img = draw_line(img,line,color)
            img = draw_line(img, [self.cotiledone, line[0]], color=(255,127,0))

        return img

    def is_dead(self) -> bool:
        return self.hipocotilo is None and self.raiz_prim is None


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

    def get_raiz_prim(self):
        return self.raiz_prim

    def get_hipocotilo(self):
        return self.hipocotilo

    def get_cotiledone(self):
        return self.cotiledone
    
    def get_anchor(self):
        hipo = self.get_hipocotilo()
        anchor = None
        if hipo is None:
            anchor = np.array([self.get_cotiledone(), self.get_cotiledone()])
            #anchor = self.get_cotiledone()
        else:
            anchor = hipo[[0,-1]]
            #anchor = hipo[-1]
        return anchor

    def build(self):
        return Seedling(self.raiz_prim, self.hipocotilo, self.cotiledone)




class SeedlingSolver:
    def __init__(self, raiz_prim_links, hipocotilo_links, cotiledone, max_cost):
        self.raiz_prim_links = raiz_prim_links
        self.hipocotilo_links = hipocotilo_links
        self.cotiledone = cotiledone
        self.MAX_COST = max_cost
        self.sdl_builders = [SeedlingBuilder().set_cotiledone(cot) for cot in self.cotiledone]

    def _match_hipocotilo(self):
        hipocotilo_bounds = np.array([hipo[[0,-1]] for hipo in self.hipocotilo_links])
        expnd_hip = hipocotilo_bounds[None, ...]
        expnd_cot = self.cotiledone[:, None, None]
        # rows are cotiledone
        # columns are hipocotilo
        costs = np.linalg.norm(expnd_hip - expnd_cot, axis=-1)
        choosen_bound = np.eye(2)[np.argmin(costs,axis=-1)]
        min_costs = costs[choosen_bound.astype(bool)].reshape(costs.shape[:2])
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(min_costs)

        for i,j in zip(row_ind, col_ind):
            if min_costs[i,j] >= self.MAX_COST: continue
            sdl_builder = self.sdl_builders[i]

            if np.argmax(choosen_bound[i,j]) == 0:
                sdl_builder.set_hipocotilo(self.hipocotilo_links[j])
            else:
                sdl_builder.set_hipocotilo(self.hipocotilo_links[j][::-1])

    def _match_raiz_prim(self):
        choosen_anchors = np.array([sdl.get_anchor() for sdl in self.sdl_builders])
        raiz_prim_bounds = np.array([rp[[0,-1]] for rp in self.raiz_prim_links])
        expnd_rp = raiz_prim_bounds[np.newaxis, ..., np.newaxis, :, :]
        expnd_anchor = choosen_anchors[:, np.newaxis, ..., np.newaxis, :]
        # rows are choosen anchors
        # columns are raiz prim
        costs = np.linalg.norm(expnd_rp - expnd_anchor, axis=-1)
        min_costs3 = np.min(np.min(costs, axis=-2), axis=-1)
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(min_costs3)

        for i,j in zip(row_ind, col_ind):
            if min_costs3[i,j] >= self.MAX_COST: continue
            sdl_builder = self.sdl_builders[i]
            sdl_builder.set_raiz_prim(self.raiz_prim_links[j])


    def match(self):
        self._match_hipocotilo()
        self._match_raiz_prim()
        return [builder.build() for builder in self.sdl_builders]

