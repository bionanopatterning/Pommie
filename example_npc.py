import Pommie
import Pommie.compute as compute
import numpy as np
import matplotlib.pyplot as plt
import mrcfile
import os
import pandas as pd
from scipy.ndimage import zoom
import json
import glob


compute.initialize()


root = "Z:/mgflast/240911_Alg/"


def bin(data, bin_factor=2):
    b = bin_factor
    z, y, x = data.shape
    return data[:z // b * b, :y // b * b, :x // b * b].reshape((z // b, b, y // b, b, x // b, b)).mean(5).mean(3).mean(1)

transforms = Pommie.Transform.sample_unit_sphere(360, polar_lims=(-4/18 * np.pi, 4/18 * np.pi))
vectors = list()
for t in transforms:
    vectors.append(t.orientation_vector)
with open(os.path.join(root, "npc", "transforms.json"), 'w') as f:
    json.dump(vectors, f, indent=2)


template = bin(mrcfile.read(os.path.join(root, "emd_14321_1568.mrc")))
template = Pommie.Particle(template, apix = 15.68*2)
template = compute.gaussian_filter([template], sigma=50.0, kernel_size=32)[0]

template.plot_projections()

particle_mask = Pommie.Mask(template)
particle_mask = particle_mask.spherical(radius_px=particle_mask.n // 2**0.5)


compute.set_tm2d_n(n=template.n)

tomos = [os.path.basename(os.path.splitext(f)[0]).split("__")[0] for f in glob.glob(os.path.join(root, "npc", "*__NPC_TM_Mask.mrc"))]


for j, tomo in enumerate(tomos):
    # load mask
    volume_mask = Pommie.Volume.from_path(os.path.join(root, "npc", f"{tomo}__NPC_TM_Mask.mrc"))

    # Now do TM on the density.
    new_mask = (zoom(volume_mask.data, 2.0) >= 0.5).astype(np.float32)
    new_mask[:template.n // 2, :, :] = 0
    new_mask[-template.n // 2:, :, :] = 0
    volume_mask = Pommie.Volume.from_array(new_mask)


    density = bin(mrcfile.read(os.path.join(root, "full_dataset", f"{tomo}.mrc")))
    volume = Pommie.Volume.from_array(density)
    score = compute.find_template_in_volume(volume=volume,
                                            volume_mask=volume_mask,
                                            template=template,
                                            template_mask=particle_mask,
                                            transforms=transforms,
                                            dimensionality=2,     # 2 for 2D, 3 for 3D matching.
                                            stride=2,
                                            return_indices=False,
                                            similarity_function=2)
    with mrcfile.new(os.path.join(root, "npc", f"{tomo}__NPC_density.mrc"), overwrite=True) as f:
        f.set_data(score)
    with mrcfile.new(os.path.join(root, "npc", f"{tomo}__density.mrc"), overwrite=True) as f:
        f.set_data(density)