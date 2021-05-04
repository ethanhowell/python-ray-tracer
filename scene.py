import numpy as np


class DirectionalLight:
    def __init__(self, direction, color):
        self.direction = direction
        self.color = color

    def __repr__(self):
        return (
            f"Directional Light\n    Direction:{self.direction}\n    Color:{self.color}"
        )


class Sphere:
    def __init__(self, center, r, kd, ks, ka, od, os, kgls):
        self.center = center
        self.r = r
        self.kd = kd
        self.ks = ks
        self.ka = ka
        self.od = od
        self.os = os
        self.kgls = kgls

    def normal(self, point):
        return (point - self.center) / self.r

    def intersection(self, ray):
        oc = self.center - ray.start
        oc_len = np.linalg.norm(oc)
        inside_sphere = oc_len < self.r

        t_ca = ray.direction.dot(oc)
        if t_ca < 0 and not inside_sphere:
            return None
        t_hc_2 = self.r ** 2 - oc_len ** 2 + t_ca ** 2
        if t_hc_2 < 0:
            return None
        t_hc = np.sqrt(t_hc_2)
        time = t_ca + t_hc if inside_sphere else t_ca - t_hc
        return None if np.isclose(time, 0) else (time, self)

    def __repr__(self):
        return f"Sphere\n    Center: {self.center}\n    Radius:{self.r}\n    Kd: {self.kd}\n    Ks: {self.ks}\n    Ka: {self.ka}\n    Od: {self.od}\n    Os: {self.os}\n    Kgls: {self.kgls}\n"


class Triangle:
    def __init__(self, v1, v2, v3, kd, ks, ka, od, os, kgls):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.kd = kd
        self.ks = ks
        self.ka = ka
        self.od = od
        self.os = os
        self.kgls = kgls

        self.l1 = v2 - v1
        self.l2 = v3 - v2
        self.l3 = v1 - v3

        self._normal = np.cross((v2 - v1), -self.l3)
        self._normal /= np.linalg.norm(self._normal)
        self.d = self._normal.dot(v1)

    def normal(self, point):
        return self._normal

    def intersection(self, ray):
        denom = self._normal.dot(ray.direction)
        if denom == 0:
            return None
        t = (self.d - self._normal.dot(ray.start)) / denom
        if t < 0 or np.isclose(t, 0):
            return None
        p = ray.at(t)

        if (
            np.cross(self.l1, p - self.v1).dot(self._normal) < 0
            or np.cross(self.l2, p - self.v2).dot(self._normal) < 0
            or np.cross(self.l3, p - self.v3).dot(self._normal) < 0
        ):
            return None
        else:
            return t, self

    def __repr__(self):
        return f"Triangle\n    V1: {self.v1}\n    V2: {self.v2}\n    V3:{self.v3}\n    Kd: {self.kd}\n    Ks: {self.ks}\n    Ka: {self.ka}\n    Od: {self.od}\n    Os: {self.os}\n    Kgls: {self.kgls}\n"


class Camera:
    def __init__(self, look_at, look_from, look_up, fov):
        self.look_at = look_at
        self.look_from = look_from
        self.look_up = look_up
        self.fov = fov
        self.n = look_from - look_at
        self.n /= np.linalg.norm(self.n)
        self.u = np.cross(look_up, self.n)
        self.u /= np.linalg.norm(self.u)
        self.v = np.cross(self.n, self.u)
        self.v /= np.linalg.norm(self.v)

    def __repr__(self):
        return f"Camera\n    Look At: {self.look_at}\n    Look From:{self.look_from}\n    Look Up: {self.look_up}\n    Field of View: {self.fov}"


class Ray:
    def __init__(self, start, direction):
        self.start = start
        self.direction = direction / np.linalg.norm(direction)

    def at(self, t):
        return self.start + self.direction * t

    def __repr__(self):
        return f"Ray {self.start} {self.direction}"

    def _get_intersection(self, objects):
        time = np.inf
        shape = None
        for o in objects:
            if i := o.intersection(self):
                t, s = i
                if t < time:
                    time, shape = t, s
        return time, shape

    def _reaches_light(self, objects):
        for o in objects:
            if o.intersection(self):
                return False
        return True

    def get_color(self, scene):
        time, shape = self._get_intersection(scene.objects)
        if shape:
            point = self.at(time)
            normal = shape.normal(point)
            color = shape.ka * scene.ambient_color * shape.od

            shadow_ray = Ray(point, scene.light.direction)
            if shadow_ray._reaches_light(scene.objects):
                specular_reflection_vector = (
                    2 * normal * normal.dot(scene.light.direction)
                    - scene.light.direction
                )
                color += scene.light.color * (
                    shape.kd * shape.od * max(0, normal.dot(scene.light.direction)) +
                    shape.ks * shape.os * max(0, specular_reflection_vector.dot(-self.direction)) ** shape.kgls
                )

            reflection_direction = (
                self.direction -
                2 * normal * normal.dot(self.direction)
            )
            bounced_ray = Ray(point, reflection_direction)
            color += shape.ks * bounced_ray.get_color(scene)

        else:
            color = scene.background_color
        return color


class Scene:
    @staticmethod
    def from_file(filename):
        with open(filename, "r") as scene_file:
            look_at = np.array([float(c) for c in scene_file.readline().split()[1:]])
            look_from = np.array([float(c) for c in scene_file.readline().split()[1:]])
            look_up = np.array([float(c) for c in scene_file.readline().split()[1:]])
            fov = float(scene_file.readline().split()[1])
            camera = Camera(look_at, look_from, look_up, fov)

            light_line = scene_file.readline().split()
            light_direction = np.array([float(c) for c in light_line[1:4]])
            light_color = np.array([float(c) for c in light_line[5:8]])
            light = DirectionalLight(light_direction, light_color)

            ambient_color = np.array(
                [float(c) for c in scene_file.readline().split()[1:]]
            )
            background_color = np.array(
                [float(c) for c in scene_file.readline().split()[1:]]
            )

            objects = []
            for line in scene_file:
                split_line = line.split()
                if split_line[0] == "Sphere":
                    center = np.array([float(c) for c in split_line[2:5]])
                    r = float(split_line[6])

                    kd = float(split_line[8])
                    ks = float(split_line[10])
                    ka = float(split_line[12])
                    od = np.array([float(c) for c in split_line[14:17]])
                    os = np.array([float(c) for c in split_line[18:21]])
                    kgls = float(split_line[22])

                    object = Sphere(center, r, kd, ks, ka, od, os, kgls)
                elif split_line[0] == "Triangle":
                    v1 = np.array([float(c) for c in split_line[1:4]])
                    v2 = np.array([float(c) for c in split_line[4:7]])
                    v3 = np.array([float(c) for c in split_line[7:10]])

                    kd = float(split_line[11])
                    ks = float(split_line[13])
                    ka = float(split_line[15])
                    od = np.array([float(c) for c in split_line[17:20]])
                    os = np.array([float(c) for c in split_line[21:24]])
                    kgls = float(split_line[25])

                    obj = Triangle(v1, v2, v3, kd, ks, ka, od, os, kgls)
                else:
                    raise TypeError(f"Unrecognized shape {split_line[0]}")
                objects.append(obj)
            return Scene(camera, light, ambient_color, background_color, objects)

    def __init__(self, camera, light, ambient_color, background_color, objects):
        self.camera = camera
        self.light = light
        self.ambient_color = ambient_color
        self.background_color = background_color
        self.objects = objects

    def __repr__(self):
        objects_str = "\n".join(repr(o) for o in self.objects)
        return f"{self.camera}\n{self.light}\nAmbient Color: {self.ambient_color}\nBackground Color: {self.background_color}\n{objects_str}"
