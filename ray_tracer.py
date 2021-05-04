import numpy as np
import sys
from scene import Scene, Ray
from PIL import Image
from datetime import datetime
import multiprocessing as mp
from functools import partial
import random

class RayTracer:
    def __init__(self, scene, res_x, res_y):
        self.scene = scene
        self.res_x = res_x
        self.res_y = res_y

        window_y = np.sqrt(2) * np.tan(np.deg2rad(scene.camera.fov))
        window_x = res_x / res_y * window_y

        scaled_u = self.scene.camera.u * window_x
        scaled_v = self.scene.camera.v * window_y

        self.scale_p_x = 2 * scaled_u / res_x
        self.scale_p_y = 2 * scaled_v / res_y

        self.constant_term = scaled_u + scaled_v + self.scene.camera.n


    def _render(self, y, x):
        ray = self._generate_ray(x, y)
        return ray.get_color(self.scene)

    def _generate_ray(self, p_x, p_y):
        direction = p_x * self.scale_p_x + p_y * self.scale_p_y - self.constant_term
        return Ray(self.scene.camera.look_from, direction)

    def _print_progress(self, start, y):
        diff = datetime.now() - start
        remaining = diff * self.res_y / y - diff
        percent = y / self.res_y * 100

        print(f'{percent:.2f}% -- Estimated time remaining: {remaining}')


    def render(self):
        start = datetime.now()
        image_data = np.zeros((self.res_y, self.res_x, 3), dtype=np.dtype(float))
        pool = mp.Pool(mp.cpu_count())
        # randomize for more accurate time remaining estimate
        rows = list(range(self.res_y))
        random.shuffle(rows)
        for count, y in enumerate(rows):
            target = partial(self._render, self.res_y - 1 - y)
            bob = pool.map(target, range(self.res_x))
            image_data[y] = bob

            self._print_progress(start, count + 1)

        end = datetime.now()
        print(f'Render time: {end - start}')
        return image_data

if __name__ == '__main__':
    name = sys.argv[1][:sys.argv[1].find('.')]
    scene = Scene.from_file(sys.argv[1])
    
    image_data = RayTracer(scene, 500, 500).render()
    image = Image.fromarray(np.uint8(image_data * 255), 'RGB')
    image.save(f'{name}.png')