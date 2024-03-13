import numpy as np

class Ellipsoid(object):

    def __init__(self, radii, center):
        self.radii = radii
        self.center = center
        self.a = radii[0]
        self.b = radii[1]
        self.c = radii[2]
        self.x0 = center[0]
        self.y0 = center[1]
        self.z0 = center[2]

        self.plane_normal = center/radii**2
        self.intercept = np.sum((center/radii)**2) - 1
        self.polar()
        self.rotate()


    def polar(self):
        self.r = np.sqrt(np.sum(self.center**2))
        self.theta = np.arccos(self.z0/self.r)
        if self.theta == 0:
            self.phi = 0
        else:
            cos_arg = self.x0/np.sin(self.theta)/self.r
            if cos_arg>1:
                cos_arg = 1
            if cos_arg<-1:
                cos_arg = -1

            sin_arg = self.y0/np.sin(self.theta)/self.r
            if sin_arg>1:
                sin_arg = 1
            if sin_arg <-1:
                sin_arg = -1

            phi_x = np.arccos(cos_arg)
            phi_y = np.arcsin(sin_arg)
            if phi_x == phi_y:
                self.phi = phi_x
            elif phi_x <= np.pi/2:
                self.phi = phi_y + 2*np.pi
            elif phi_y>=0:
                self.phi = phi_x
            else:
                self.phi = 2*np.pi - phi_x


    def rotate(self):
        rot = np.array([[np.cos(self.theta)*np.cos(self.phi), np.cos(self.theta)*np.sin(self.phi), -np.sin(self.theta)],
                        [-np.sin(self.phi), np.cos(self.phi),0],
                        [np.sin(self.theta)*np.cos(self.phi), np.sin(self.theta)*np.sin(self.phi), np.cos(self.theta)]])
        #print(rot)
        self.plane_rot = rot.dot(self.plane_normal)
        self.radii_rot = (rot).dot((np.diag(1/self.radii**2)).dot(rot.T))


    def in_ellipsoid(self, theta, phi):
        unit = np.array([np.sin(theta)*np.cos(phi),
                         np.sin(theta)*np.sin(phi),
                         np.cos(theta)]).T
        #print(unit)
        t = self.intercept/(unit.dot(self.plane_rot))
        #print(unit.T*t)
        coord = unit.T*t - np.expand_dims(np.array([0,0,self.r]), axis = 1)
        #print(coord.T.dot(self.radii_rot))
        return np.sum((coord.T.dot(self.radii_rot))*coord.T, axis = 1) - 1


    def mc_int(self, num = 1000000):
        max_r = np.max(self.radii)
        min_cos = (self.r - max_r)/np.sqrt((self.r - max_r)**2 + max_r**2)

        diff = 1 - min_cos
        #print(diff)

        theta = np.arccos(np.random.random(num)*diff + min_cos)
        phi = np.random.random(num)*np.pi*2
        return np.sum(self.in_ellipsoid(theta, phi)<=0)/num/2*diff
