import quaternion
import mrcfile
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from Pommie.opengl import *
from Pommie.compute import reorient_volume, gaussian_filter
from Pommie import util
import quaternion as quat
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from itertools import cycle
import json
from scipy.ndimage import binary_erosion, binary_dilation
import copy


class Dataset:
    def __init__(self, particles, n=None):
        if isinstance(particles, str):
            self.path = particles
            mrcs = glob.glob(os.path.join(self.path, "*.mrc"))
            with mrcfile.open(mrcs[0], header_only=True) as mrc:
                self.apix = mrc.voxel_size.x
            self.particles = list()
            for path in mrcs if n is None else mrcs[:n]:
                self.particles.append(Particle(Dataset.read_mrc(path), path=path, apix=self.apix))
                if self.particles[0].data.shape != self.particles[-1].data.shape:
                    file_a = os.path.basename(self.particles[0].path)
                    file_b = os.path.basename(self.particles[-1].path)
                    raise Exception(f"Boxes of different sizes encountered in {self.path} for files:\n\t{file_a} ({self.particles[0].data.shape})\n\t{file_b} ({self.particles[-1].data.shape})")
        elif isinstance(particles, list):
            self.path = 'None'
            self.apix = particles[0].apix
            self.particles = particles
        elif particles is None:
            self.particles = list()
            self.apix = 1.0
        self.n = self.particles[0].data.shape[0]


    def __iter__(self):
        return self.particles.__iter__()

    def __getitem__(self, item):
        return self.particles[item]

    def __len__(self):
        return len(self.particles)

    def __str__(self):
        return f"Dataset at path '{self.path}': {len(self.particles)} particles with box size {self.n} and {self.apix} A/pix."

    @property
    def sum(self):
        sum = np.zeros_like(self.particles[0].data)
        n = 0
        for p in self.particles:
            sum += p.resample(p.get_transform_relative_to_original())[0].data
            n += 1
        return Particle(sum, apix=self.apix)

    @property
    def average(self):
        avg = self.sum
        avg.data /= len(self.particles)
        return avg

    @property
    def transforms(self):
        return [p.get_transform_relative_to_original() for p in self]

    @staticmethod
    def read_mrc(path):
        data = mrcfile.read(path)

        data = data.astype(np.float32)
        return data

    def invert(self):
        for p in self:
            p.invert()

    def add_constant(self, c):
        for p in self:
            p.data += c

    def normalise(self):
        for p in self.particles:
            p.normalise()

    def set_apix(self, apix):
        self.apix = apix
        for p in self:
            p.apix = apix

    def add_particle(self, particle):
        self.particles.append(particle)

    def remove_particle(self, particle):
        if particle in self.particles:
            self.particles.remove(particle)


class Transform:
    UP = np.array([1.0, 0.0, 0.0, 1.0])

    def __init__(self, q=quat.quaternion(1.0, 0.0, 0.0, 0.0), p=(0.0, 0.0, 0.0), r=quat.quaternion(1.0, 0.0, 0.0, 0.0)):
        """
        :param q: a quaternion that defines an orientation relative to the world up vector (vec3(0.0, 0.0, 1.0))
        :param p: position, in unit pixels.
        :param r: a quaternion that defines an initial orientation around the world up vector.

        Note that there are a number of static methods that generate Transforms based on more intuitive values, such as
        polar and azimuthal coordinates (to define q), or initial rotation angles (the 'turn', to define r). Transforms
        can be multiplied to combine their effects. For example, to sample the unit sphere as well as various turn
        values, one would use:

        transforms = Transform.meshgrid(Transform.sample_unit_sphere(20), Transform.sample_turn(np.linspace(0, 2 * np.pi, 10))

        This would return a list of 200 transform objects, sampling 20 uniformly distributed points on the unit sphere
        using 10 different initial local rotation values for each.

        To generate a single transform with a fully defined 3D orientation, use for example:

        a = Transform.sample_polar(np.pi / 4, 0.0)
        b = Transform.sample_turn(np.pi)
        transform = a * b
        """
        self._p = np.array(p)  # shift, in unit pixels.
        self._q = q
        self._r = r

    @property
    def six_vector(self):
        spherical = self.to_spherical_coords()
        turn = quaternion.as_spherical_coords(self.r)[0]
        return np.array([self.p[0], self.p[1], self.p[2], spherical[0], spherical[1], turn])

    @staticmethod
    def from_six_vector(v):
        t = Transform.from_spherical_coords(polar=v[3], azimuth=v[4], turn=v[5])
        t.p = v[0:3]
        return t

    @property
    def p(self):
        return self._p

    @property
    def q(self):
        return self._q

    @property
    def r(self):
        return self._r

    @p.setter
    def p(self, val):
        self._p = np.array(val)

    @q.setter
    def q(self, val):
        self._q = val

    @r.setter
    def r(self, val):
        self._r = val

    def info(self):
        q_angle = quaternion.as_spherical_coords(self.q)[0] / np.pi * 180.0
        r_angle = quaternion.as_spherical_coords(self.r)[0] / np.pi * 180.0
        p_len = np.sqrt(np.sum(self.p**2))
        print(f"{q_angle:.1f}° world, {r_angle:.1f}° local, {p_len:.1f} px.")

    def __str__(self):
        return f"Transform: q = {self.q}, p = {self.p}, r = {self.r}"

    def __mul__(self, other):
        # = self * other
        m = self.mat * other.mat
        # figure out the corresponding q, p, and r
        p = (m[0, 3], m[1, 3], m[2, 3])

        q = self.q * other.q
        r = self.r * other.r
        return Transform(q=q, p=p, r=r)

    def __sub__(self, other):
        """
        Returns the transform M that defines the difference between self (S) and other (O) as: M * S = O, where * is effectively a matrix multiplication
        """
        return self.inverse * other

    def __add__(self, other):
        return Transform.__sub__(self, other.inverse)

    def __getitem__(self, item):
        return self

    def __eq__(self, other):
        if isinstance(other, Transform):
            return self.q == other.q and np.all(self.p == other.p) and self.r == other.r
        return False

    def copy(self):
        return Transform(self.q, self.p, self.r)

    @property
    def rmat(self):
        mat = np.matrix(np.eye(4), dtype=np.float32)
        mat[:3, :3] = quaternion.as_rotation_matrix(self.r * self.q)
        return mat

    @property
    def tmat(self):
        mat = np.matrix(np.eye(4), dtype=np.float32)
        mat[:3, 3] = np.matrix(self.p).T
        return mat

    @property
    def mat(self):
        return self.tmat * self.rmat

    @property
    def inverse(self):
        return Transform(self.q.inverse(), -self.p, self.r.inverse())

    @property
    def orientation_vector(self):
        vector = np.asarray(self.rmat * np.matrix([Transform.UP]).T).squeeze()[:-1]
        return vector[0], vector[1], vector[2]

    def invert(self):
        i = self.inverse
        self.q = i.q
        self.p = i.p
        self.r = i.r

    def save(self, path):
        path = os.path.splitext(path)[0]+".pom"
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        with open(path, "w") as f:
            _q = quaternion.as_float_array(self.q).tolist()
            _r = quaternion.as_float_array(self.r).tolist()
            _p = self.p.tolist()
            json.dump({'q': _q, 'p': _p, 'r': _r}, f)

    @staticmethod
    def load(path):
        path = os.path.splitext(path)[0]+".pom"
        with open(path, "r") as f:
            values = json.load(f)
        _q = quaternion.from_float_array(values['q'])
        _r = quaternion.from_float_array(values['r'])
        return Transform(q=_q, p=values['p'], r=_r)

    @staticmethod
    def from_spherical_coords(polar=0.0, azimuth=0.0, turn=0.0):
        q = quat.from_spherical_coords(polar, azimuth)
        r = quat.from_rotation_vector(turn * Transform.UP[:3])
        return Transform(q=q, r=r)

    def to_spherical_coords(self):
        return quaternion.as_spherical_coords(self.q)

    @staticmethod
    def from_rotation_matrix(m):
        return quaternion.from_rotation_matrix(m)

    @staticmethod
    def sample_polar(polar, azimuth=0):
        transforms = list()
        for p in polar:
            q = quat.from_spherical_coords(p, azimuth)
            transforms.append(Transform(q))
        return transforms if len(transforms) > 1 else transforms[0]

    @staticmethod
    def sample_azimuth(azimuth, polar=0):
        transforms = list()
        for a in azimuth:
            q = quat.from_spherical_coords(polar, a)
            transforms.append(Transform(q))
        return transforms if len(transforms) > 1 else transforms[0]

    @staticmethod
    def sample_turn(turn):
        transforms = list()
        for t in turn:
            r = quat.from_rotation_vector(t * Transform.UP[:3])
            transforms.append(Transform(r=r))
        return transforms if len(transforms) > 1 else transforms[0]

    @staticmethod
    def sample_unit_sphere_random(n_samples, polar_range=(0.0, np.pi), azimuthal_range=(0.0, 2 * np.pi)):
        azimuth = np.random.uniform(*azimuthal_range, n_samples)

        # Map the polar range from [0, pi] to [-1, 1] for arccos
        polar_min, polar_max = polar_range
        polar_min_mapped = np.cos(polar_min)
        polar_max_mapped = np.cos(polar_max)
        polar = np.arcsin(np.random.uniform(polar_max_mapped, polar_min_mapped, n_samples))

        transforms = []
        for p, a in zip(polar, azimuth):
            q = quat.from_spherical_coords(p, a)
            transforms.append(Transform(q))

        return transforms

    @staticmethod
    def sample_unit_sphere(n_samples, azimuth_lims=(0, 2*np.pi), polar_lims=(-np.pi/2, np.pi/2)):
        azimuth = np.pi * (3 - 5**0.5) * np.arange(n_samples) % (2 * np.pi)
        sin_polar_min = (np.sin(polar_lims[0]) + 1) / 2
        sin_polar_max = (np.sin(polar_lims[1]) + 1) / 2
        polar_norm = np.linspace(sin_polar_min, sin_polar_max, n_samples)
        polar = np.arcsin(2 * polar_norm - 1)
        transforms = list()
        for p, a in zip(polar, azimuth):
            if polar_lims[0] <= p <= polar_lims[1] and azimuth_lims[0] <= a <= azimuth_lims[1]:
                q = quat.from_spherical_coords(p, a)
                transforms.append(Transform(q))
        return transforms

    @staticmethod
    def sample_grid(x=(0, ), y=(0, ), z=(0, ), meshgrid=False):
        transforms = list()
        if not meshgrid:
            nx = len(x)
            ny = len(y)
            nz = len(z)
            nmax = max((nx, ny, nz))
            _x = cycle(x) if nmax != nx else x
            _y = cycle(y) if nmax != ny else y
            _z = cycle(z) if nmax != nz else z
            if nx == ny == nz:
                _x = x
            for i, j, k in zip(_x, _y, _z):
                transforms.append(Transform(p=[i, j, k]))
            return transforms
        else:
            for i in x:
                for j in y:
                    for k in z:
                        transforms.append(Transform(p=[i, j, k]))
            return transforms

    @staticmethod
    def meshgrid(transforms_a, transforms_b):
        """
        :param transforms_a: list of N Transform objects
        :param transforms_b: list of M Transform objects
        :return: list of N * M Transform objects, combining the sampling of A and B.
        """
        out = list()
        for a in transforms_a:
            for b in transforms_b:
                out.append(a * b)
        return out

    @staticmethod
    def multiply(transforms_a, transforms_b):
        out = list()
        for a, b in zip(transforms_a, transforms_b):
            out.append(a * b)
        return out

    @staticmethod
    def plot_transforms(transforms):
        """
        :param transforms: list of Transform objects.
        """

        # Create a subplot figure with 1 row and 2 columns
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

        # Add scatter plots to each subplot
        cmap = plt.cm.get_cmap('twilight_shifted')

        for t in transforms:
            d = np.asarray(t.tmat.T * np.matrix([0.0, 0.0, 0.0, 1.0]).T).squeeze()[:-1]
            fig.add_trace(go.Scatter3d(x=[d[0]], y=[d[1]], z=[d[2]], mode='markers', marker=dict(color='black')), row=1,col=2)


            n = np.asarray(t.rmat * np.matrix([Transform.UP]).T).squeeze()[:-1]

            turn = 2 * np.arccos(t.r.w)
            c = (*cmap((turn / (2 * np.pi)) % 1.0)[:-1], 0.3)
            fig.add_trace(go.Scatter3d(x=[n[0]], y=[n[1]], z=[n[2]], mode='markers', marker=dict(color=c)), row=1, col=1)

        # Update layout for each subplot
        fig.update_layout(title="Transformation orientations (left) and translations (right)")
        fig.update_layout(showlegend=False)
        fig.update_scenes(selector=dict(row=1, col=1),
                          xaxis_title='nx',
                          yaxis_title='ny',
                          zaxis_title='nz',
                          xaxis=dict(range=[-1.5, 1.5]),
                          yaxis=dict(range=[-1.5, 1.5]),
                          zaxis=dict(range=[-1.5, 1.5]))
        fig.update_scenes(selector=dict(row=1, col=2),
                          xaxis_title='X',
                          yaxis_title='Y',
                          zaxis_title='Z')
        # Show the figure
        fig.show()

    @staticmethod
    def identity():
        return Transform()


class Volume:
    id_gen = count(0)

    def __init__(self):
        self.uid = next(Volume.id_gen)
        self.data = None
        self.apix = 1.0

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def from_path(path):
        with mrcfile.open(path) as mrc:
            volume = Volume()
            volume.apix = mrc.voxel_size.x
            volume.data = np.array(mrc.data, dtype=np.float32)
            return volume

    @staticmethod
    def from_array(array, apix=1.0):
        volume = Volume()
        volume.data = array
        volume.apix = apix
        return volume

    def get_particle(self, coordinates, n):
        return Particle(data=self.data[
                             coordinates[0] - n // 2:coordinates[0] - n // 2 + n,
                             coordinates[1] - n // 2:coordinates[1] - n // 2 + n,
                             coordinates[2] - n // 2:coordinates[2] - n // 2 + n],
                        apix=self.apix)

    def crop(self, jlim, klim, llim):
        data = self.data[jlim[0]:jlim[1], klim[0]:klim[1], llim[0]:llim[1]]
        return Volume.from_array(data, self.apix)

    @property
    def projections(self):
        xy = np.mean(self.data, axis=0)
        xz = np.mean(self.data, axis=1)
        yz = np.mean(self.data, axis=2)
        return xy, xz, yz

    def plot_projections(self):
        p = self.projections
        titles = ["XY", "XZ", "YZ"]
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(-p[i], cmap='Grays')
            plt.title(titles[i])
        plt.show()
        return self

    def save(self, path):
        dir = os.path.dirname(path)
        if dir and not os.path.exists(dir):
            os.makedirs(dir)
        if not os.path.splitext(path)[-1] == ".mrc":
            path += ".mrc"
        with mrcfile.new(path, overwrite=True) as f:
            f.set_data(self.data)
            f.voxel_size = self.apix

    def unbin(self, fac=2):
        self.data = np.repeat(np.repeat(np.repeat(self.data, fac, axis=0), fac, axis=1), fac, axis=2)
        self.apix = self.apix / fac
        return self

    def threshold(self, threshold=0.5):
        self.data = self.data > threshold
        return self

    def to_shell_mask(self, threshold=0.5, thickness_out=2, thickness_in=2):
        """
        threshold: value to threshold the volume at before performing binary operation
        thickness_out: thickness (in pixels) of the shell to append in the outward direction (negative values allowed);
        thickness_in: thickness (in pixels) of the shell to append in the inward direction (negative values allowed);
        """
        self.threshold(threshold)

        for j in range(self.data.shape[0]):
            if thickness_out > 1:
                img_a = binary_dilation(self.data[j, :, :], iterations=thickness_out).astype(np.float32)
            elif thickness_out == 0:
                img_a = self.data[j, :, :]
            else:
                img_a = binary_erosion(self.data[j, :, :], iterations=-thickness_out).astype(np.float32)
            if thickness_in > 1:
                img_b = binary_erosion(self.data[j, :, :], iterations=thickness_in).astype(np.float32)
            elif thickness_in == 0:
                img_b = self.data[j, :, :]
            else:
                img_b = binary_dilation(self.data[j, :, :], iterations=-thickness_in).astype(np.float32)
            self.data[j, :, :] = img_a - img_b
        return self

    def erode_2d(self, iterations=1):
        for j in range(self.data.shape[0]):
            self.data[j, :, :] = binary_erosion(self.data[j, :, :], iterations=iterations).astype(np.float32)
        return self

class Particle:
    id_gen = count(0)

    # TODO: a __new__ method that recognizes whether argument is ndarray or path
    def __init__(self, data=None, path='', apix=1.0):
        self.uid = next(Particle.id_gen)
        self.transform = Transform()
        self.data = data.astype(np.float32)
        self.path = path
        self.n = self.data.shape[0]
        self.apix = apix
        self.selected_transforms = list()
        self.scores = list()
        self.test_transform = None

    def __sub__(self, other):
        return Particle(self.data - other.data, apix=self.apix)

    @staticmethod
    def from_path(path):
        with mrcfile.open(path) as mrc:
            apix = mrc.voxel_size.x
            data = np.array(mrc.data, dtype=np.float32)
            return Particle(data, path, apix)

    def apply_mask(self, mask):
        if mask is None:
            return self
        self.data *= mask.data
        return self

    def remove_outside_mask(self, mask):
        if mask is None:
            return self
        self.data[mask.data == 0] = np.amin(self.data)
        return self

    def get_transform_relative_to_original(self):
        transform = Transform()
        for t in self.selected_transforms:
            transform = transform * t
        return transform

    @property
    def full_transform(self):
        return self.get_transform_relative_to_original()

    def resample(self, transforms, mask=None):
        if isinstance(transforms, Transform):
            transforms = [transforms]
        resampled = reorient_volume(self.data, transforms, None if mask is None else mask.data)
        for i in range(len(resampled)):
            resampled[i] = Particle(resampled[i], apix=self.apix)
            resampled[i].transform = transforms[i]
        return Dataset(resampled)

    def normalise(self):
        mu = np.mean(self.data)
        sd = np.std(self.data)
        self.data = (self.data - mu) / sd

    def invert(self):
        self.data = -self.data

    def inverse(self):
        return Particle(-self.data, apix=self.apix)

    def save(self, path):
        dir = os.path.dirname(path)
        if dir and not os.path.exists(dir):
            os.makedirs(dir)
        if not os.path.splitext(path)[-1] == ".mrc":
            path += ".mrc"
        with mrcfile.new(path, overwrite=True) as f:
            f.set_data(self.data)
            f.voxel_size = self.apix

    def change_pixel_and_box_size(self, apix=None, box_size=None):
        if not apix:
            apix = self.apix
        if not box_size:
            new_volume_size = self.n
        else:
            new_volume_size = box_size
        f = self.apix / apix
        print(f)
        new_vol = zoom(self.data, f, order=3)

        # Create an empty volume with the new size
        self.data = np.zeros((new_volume_size, new_volume_size, new_volume_size), dtype=np.float32)

        # Calculate the start indices to center the new volume within self.data
        start_indices = [(s - new_volume_size) // 2 for s in new_vol.shape]

        # Calculate the end indices based on the start indices and the shape of new_vol
        end_indices = [s + new_volume_size for s in start_indices]
        # Place new_vol in the center of self.data
        self.data = new_vol[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]]

        # Update the size attribute
        self.n = self.data.shape[0]
        return self

    @staticmethod
    def sum(particles):
        v = particles[0].data
        for p in particles:
            v.data += p.data
        return v

    @staticmethod
    def average(particles):
        v = Particle.sum(particles)
        v.data /= len(particles)
        return v

    @property
    def projections(self):
        xy = np.mean(self.data, axis=0)
        xz = np.mean(self.data, axis=1)
        yz = np.mean(self.data, axis=2)
        return xy, xz, yz

    def plot_projections(self, other_particle=None):
        p = self.projections
        titles = ["XY", "XZ", "YZ"]
        for i in range(3):
            plt.subplot(1 if other_particle is None else 2, 3, i + 1)
            plt.imshow(-p[i], cmap='Grays')
            plt.title(titles[i])
        if other_particle is not None:
            p = other_particle.projections
            titles = ["XY", "XZ", "YZ"]
            for i in range(3):
                plt.subplot(2, 3, i + 4)
                plt.imshow(-p[i], cmap='gray')
                plt.title(titles[i])
        plt.show()
        return self

    def plot_orthoslices(self, other_particle=None):
        plt.subplot(1 if other_particle is None else 2, 3, 1)
        plt.imshow(self.data[self.n//2, :, :], cmap='gray')
        plt.title("XY")
        plt.subplot(1 if other_particle is None else 2, 3, 2)
        plt.imshow(self.data[:, self.n // 2, :], cmap='gray')
        plt.title("XZ")
        plt.subplot(1 if other_particle is None else 2, 3, 3)
        plt.imshow(self.data[:, :, self.n // 2], cmap='gray')
        plt.title("YZ")
        if other_particle is not None:
            plt.subplot(1 if other_particle is None else 2, 3, 4)
            plt.imshow(other_particle.data[other_particle.n // 2, :, :], cmap='gray')
            plt.title("XY")
            plt.subplot(1 if other_particle is None else 2, 3, 5)
            plt.imshow(other_particle.data[:, other_particle.n // 2, :], cmap='gray')
            plt.title("XZ")
            plt.subplot(1 if other_particle is None else 2, 3, 6)
            plt.imshow(other_particle.data[:, :, other_particle.n // 2], cmap='gray')
            plt.title("YZ")
        plt.show()

    def clone(self):
        return copy.deepcopy(self)

class Mask(Particle):
    def __init__(self, particle):
        super().__init__(np.ones_like(particle.data), '', apix=particle.apix)

    @staticmethod
    def new(n=32):
        return Mask(Particle(np.zeros((n, n, n))))

    def invert(self):
        self.data = (1.0 - self.data)

    def spherical(self, radius_a=10.0, radius_px=None):
        r = radius_a / self.apix if radius_px is None else radius_px
        r_sq = r**2
        self.data = np.zeros_like(self.data)
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    if (i + 0.5 - self.n/2)**2 + (j + 0.5 - self.n/2)**2 + (k + 0.5 - self.n/2)**2 < r_sq:
                        self.data[i, j, k] = 1.0
        return self

    def cylindrical(self, radius_a=10.0, radius_px=None):
        r = radius_a / self.apix if radius_px is None else radius_px
        r_sq = r **2
        self.data = np.zeros_like(self.data)
        for i in range(self.n):
            for j in range(self.n):
                if (i + 0.5 - self.n / 2)**2 + (j + 0.5 - self.n / 2)**2 < r_sq:
                    self.data[:, i, j] = 1
        return self


class Filter:
    def __init__(self, filter_type="gaussian", param=10.0, kernel_size=10):
        """
        string filter_type: type of filter that the object applies, see below for options.
        float  param:       parameter of the filter, see below.
        int    kernel_size: radius of the kernel to apply to the image. Full size will be 2 * kernel_size + 1.

        Filter objects can be applied to Datasets to affect the pixel values of the volume in that dataset. E.g.:

        dataset = Pommie.Dataset("some_directory")
        gaussian_filter = Filter("gaussian", 10.0)
        dataset = gaussian_filter(dataset)

        Will apply a gaussian filter with standard deviation 10 Angstrom to the particles in dataset.

        Available filters:

        filter_type     "gaussian"
        param           float, standard deviation of the gaussian filter in Angstrom
        """

        self.type = filter_type
        self.param = param
        self.k = kernel_size

    def __call__(self, dataset):
        if self.type == "gaussian":
            return Filter.gaussian(dataset, self.param, self.k)
        else:
            return dataset

    @staticmethod
    def gaussian(dataset, sigma, kernel_size):
        return gaussian_filter(dataset, sigma, kernel_size)


