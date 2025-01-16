from Pommer.typedefs import *
import Pommer.compute as compute

ITERATION_COUNTER = -1


def iteration(template, dataset, transforms, mask=None, filter=None, name='iterations/unnamed', similarity_function=2, score_mask=None, resample_mask=None, add_identity_transform=True):
    """
    How transforms are handled: at the start of every iteration, the data for every particle is generated by applying
    the product of all previously selected transforms (the list .selected_transforms). A temporary dataset is then
    created that contains the resulting volumes. All particles in that dataset are matched to the templates, and the
    transform of the best matching template is appended to the corresponding particle in the input dataset (and not the
    temporary dataset).
    """
    if add_identity_transform:
        transforms.append(Transform.identity())
    global ITERATION_COUNTER
    if mask is not None:
        score_mask = mask
        resample_mask = mask
    if score_mask is None:
        score_mask = Mask(template)
    if resample_mask is None:
        resample_mask = Mask(template)

    # Update counter, create output dir
    ITERATION_COUNTER += 1
    name = os.path.join(os.path.dirname(name), f'{ITERATION_COUNTER}_'+os.path.basename(name))
    print(f'Starting iteration {name}')
    dir_name = os.path.dirname(name)
    if dir_name:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    # Create a temporary dataset in which all previously found transforms are applied.
    print(f'Applying previously found transforms.')
    transformed_particles = list()
    for p in dataset:
        new_p = p.resample(p.get_transform_relative_to_original())[0]
        transformed_particles.append(new_p)
    temp_dataset = Dataset(transformed_particles)
    if filter is not None:
        print("Applying filters.")
        temp_dataset = filter(temp_dataset)

    # Template matching
    print("Matching templates.")
    template.save(f'{name}_template.mrc')
    templates = template.resample(transforms, mask=resample_mask)

    matched_template, samples, scores = compute.match_sample_to_templates(templates, temp_dataset, mask=score_mask, similarity_function=similarity_function)

    # save the results in the original dataset.
    for i, d in enumerate(dataset):
        d.selected_transforms.append(matched_template[i].transform.inverse)
        d.scores.append(scores[i])

    # save transforms
    for v in dataset:
        transform = v.get_transform_relative_to_original()
        v_name = os.path.basename(v.path)
        transform.save(os.path.join(name, v_name))

    # generate the new average
    print("Generating new average")
    average = np.zeros_like(dataset[0].data)
    n = 0
    for p in dataset:
        average += p.resample(p.get_transform_relative_to_original())[0].data
        n += 1
    average = Particle(average / n, apix=dataset.apix)
    average.save(f'{name}_average.mrc')
    return dataset
