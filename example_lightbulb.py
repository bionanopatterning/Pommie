import Pommie
import Pommie.compute as compute
import numpy as np
import matplotlib.pyplot as plt
import mrcfile
import os
import pandas as pd
from scipy.ndimage import zoom, gaussian_filter
import json
import glob
import time
compute.initialize()

root = "/data/etaC/mgflast/240911_Alg/"
if os.name == 'nt':
    root = "Z:/mgflast/240911_Alg/"
segmentation_dir = "241205_output"
experiment_name = "Lightbulb_X"
experiment_dir = os.path.join(root, "lightbulb", experiment_name)

template_path = os.path.join(root, "lightbulb", "template_1.mrc")
template_mask_path = os.path.join(root, "lightbulb", "template_1__mask_manual.mrc")

# template picked from: tomo = "06022023_BrnoKrios_Arctis_xe_Position_110_bin2.mrc"
# good tmogram to test on: 20122022_BrnoKrios_Arctis_xe_grid2_Position_13_bin2
def generate_mask(tomo):
    mitochondrion = mrcfile.read(os.path.join(root, segmentation_dir, f"{tomo}__Mitochondrion.mrc"))
    void = mrcfile.read(os.path.join(root, segmentation_dir, f"{tomo}__Void.mrc"))
    thylakoid = mrcfile.read(os.path.join(root, segmentation_dir, f"{tomo}__Thylakoid.mrc"))
    mask = Pommie.Volume.from_array(mitochondrion)
    mask = mask.to_shell_mask(threshold=0.75, thickness_out=8, thickness_in=0)
    mask = mask.unbin(2)
    void = Pommie.Volume.from_array(void).unbin(2)
    thylakoid = Pommie.Volume.from_array(thylakoid).unbin(2)
    mask.data[void.data > 0.1] = 0
    mask.data[thylakoid.data > 0.1] = 0
    mask.data[:60, :, :] = 0
    mask.data[-60:, :, :] = 0
    mask.data *= 0
    mask.data[128, :, :] = 1
    return mask

def preprocess(volume):
    volume = gaussian_filter(volume, sigma=2.0)
    return volume

def bin(data, bin_factor=2):
    b = bin_factor
    z, y, x = data.shape
    return data[:z // b * b, :y // b * b, :x // b * b].reshape((z // b, b, y // b, b, x // b, b)).mean(5).mean(3).mean(1)

os.makedirs(experiment_dir, exist_ok=True)

# load data overview
data_filters = {'Mitochondrion': (0.0, 100.0),
                'Void': (0.0, 100.0)}

summary = pd.read_excel(os.path.join(root, "summary.xlsx"), index_col=0)
columns = summary.columns
for k in data_filters:
    k_range = data_filters[k]
    if k in columns:
        summary = summary[(summary[k] >= k_range[0]) & (summary[k] <= k_range[1])]

summary = summary.sort_values(by="Mitochondrion", ascending=False)
tomos = [t for t in summary.index]

np.random.shuffle(tomos)

print(f"{len(summary)} volumes remaining after applying filters.")

template = mrcfile.read(template_path)
template -= np.mean(template)
template /= np.std(template)
template = Pommie.Particle(template, apix=15.68)
template = compute.gaussian_filter([template], sigma=20.00, kernel_size=16)[0]

template_mask = mrcfile.read(template_mask_path)
template_mask = Pommie.Particle(template_mask)


compute.set_tm2d_n(template.n)

transforms = Pommie.Transform.sample_unit_sphere(500, polar_lims=(-np.pi * 3 / 18.0, +np.pi * 3 / 18.0))


vectors = list()

for t in transforms:
    vectors.append(t.orientation_vector)
with open(os.path.join(experiment_dir, "transforms.json"), 'w') as f:
    json.dump(vectors, f, indent=2)

t_start_loop = time.time()
t_total = 0
_templates_bound = False

N = 0
for j,tomo in enumerate(tomos):
    N += 1
    print(f"\n{j}")
    if os.path.exists(os.path.join(experiment_dir, f"{tomo}__score.mrc")):
        print(tomo)
        continue
    else:
        with mrcfile.new(os.path.join(experiment_dir, f"{tomo}__score.mrc"), overwrite=True) as f:
            f.set_data(np.zeros((10, 10, 10), dtype=np.float32))
    t_start = time.time()
    volume_mask = generate_mask(tomo)
    # volume_mask.data *= 0
    # volume_mask.data[128, :, :] = 1.0
    print(f"Generating mask {time.time() - t_start:.3f} s.")

    t_start = time.time()
    density = mrcfile.read(os.path.join(root, "full_dataset", f"{tomo}.mrc"))
    print(f"Loading density volume {time.time() - t_start:.3f} s.")
    t_start = time.time()
    density = preprocess(density)
    print(f"Preprocessing density volume {time.time() - t_start:.3f} s.")
    density_volume = Pommie.Volume.from_array(density)
    ti = time.time()
    scores, indices = compute.find_template_in_volume(volume=density_volume,
                                            volume_mask=volume_mask,
                                            template=template,
                                            template_mask=template_mask,
                                            transforms=transforms,
                                            dimensionality=2,
                                            stride=1,
                                            similarity_function=2,
                                            skip_binding=_templates_bound)
    _templates_bound = True
    t_total += (time.time() - ti)
    print(f"Time per voxel in mask: {(time.time() - ti) / np.sum(volume_mask.data) * 1e9} ns/voxel.")
    t_start = time.time()
    print(np.amax(scores))
    with mrcfile.new(os.path.join(experiment_dir, f"{tomo}__score.mrc"), overwrite=True) as f:
        f.set_data(scores)
        f.voxel_size = 15.68
    # with mrcfile.new(os.path.join(experiment_dir, f"{tomo}__indices.mrc"), overwrite=True) as f:
    #     f.set_data(indices)
    #     f.voxel_size = 15.68
    #volume_mask.save(os.path.join(experiment_dir, f"{tomo}__volume_mask.mrc"))
    print(f"Saving volumes {time.time() - t_start:.3f} s.")
    print(f"\nTM average cost so far {t_total / (j + 1):.3f} s/vol.")


print(f"TM time: {t_total} for {N} volumes")
print(f"Total time: {time.time() - t_start_loop} for {N} volumes.")