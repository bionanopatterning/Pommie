from Pommie.opengl import *
import glfw
import ctypes
import time
import datetime
from scipy.ndimage import zoom

cs_reorient = Shader()
cs_similarity = Shader()
cs_filter = Shader()
cs_com = Shader()
cs_empty = Shader()
cs_tm = Shader()
cs_tm_find_max = Shader()
cs_tm_2d = Shader()
cs_tm_2d_single_image = Shader()
cs_tm_2d_n = 32
volume_buffer_a = Texture3D(True)
volume_buffer_b = Texture3D(True)
volume_buffer_c = Texture3D(True)
tm_sample_ssbo = 0
tm_templates_ssbo = 0
tm_scores_ssbo = 0
tm_scores_pointer = 0
tm_mask_ssbo = 0
tm_n_templates = 0
tm_template_size = 0
tm_volume_texture = Texture3D(True)
tm_scores_texture = Texture3D(True)
image_buffer_a = Texture()
image_buffer_b = Texture()
image_buffer_c = Texture()


def initialize():
    global cs_reorient, cs_similarity, cs_empty, cs_com, cs_filter, volume_buffer_a, volume_buffer_b, volume_buffer_c, tm_volume_texture, tm_scores_texture, cs_tm, cs_tm_find_max, cs_tm_2d, image_buffer_a, image_buffer_b, image_buffer_c, cs_tm_2d_single_image
    if not glfw.init():
        raise Exception("Could not initialize GLFW library.")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
    window = glfw.create_window(1, 1, "", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Could not create GLFW window.")
    glfw.make_context_current(window)

    # Load shaders
    root = os.path.dirname(os.path.dirname(__file__))
    cs_reorient = Shader(os.path.join(root, "Pommie", "shaders", "resample_3d_texture.glsl"))
    cs_similarity = Shader(os.path.join(root, "Pommie", "shaders", "similarity_functions.glsl"))
    cs_tm = Shader(os.path.join(root, "Pommie", "shaders", "template_match.glsl"))
    cs_tm_find_max = Shader(os.path.join(root, "Pommie", "shaders", "write_max_score_to_volume.glsl"))
    cs_empty = Shader(os.path.join(root, "Pommie", "shaders", "empty_buffer.glsl"))
    cs_filter = Shader(os.path.join(root, "Pommie", "shaders", "separable_filter.glsl"))
    cs_com = Shader(os.path.join(root, "Pommie", "shaders", "center_of_mass.glsl"))
    cs_tm_2d = Shader(os.path.join(root, "Pommie", "shaders", "template_match_2d.glsl"))
    #cs_tm_2d_single_image = Shader(os.path.join(root, "Pommie", "shaders", "template_match_2d_single_image.glsl"))
    volume_buffer_a = Texture3D()
    volume_buffer_b = Texture3D()
    volume_buffer_c = Texture3D()
    tm_volume_texture = Texture3D()
    tm_scores_texture = Texture3D()
    tm_volume_texture.set_interpolation_mode(0)
    tm_volume_texture.set_edge_mode(0)
    tm_scores_texture.set_interpolation_mode(0)
    tm_scores_texture.set_edge_mode(0)
    image_buffer_a = Texture('r32f')
    image_buffer_b = Texture('r32f')
    image_buffer_c = Texture('rgba32f')


def print_platform_limits():
    print("\nPommie.compute - platform limits:")
    print(f"GL_MAX_COMPUTE_WORK_GROUP_COUNT = ({glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, 0)}, {glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, 0)}, {glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, 0)})")
    print(f"GL_MAX_COMPUTE_WORK_GROUP_SIZE = ({glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, 0)}, {glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, 0)}, {glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, 0)})")
    print(f"GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS = {glGetInteger(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS)}")
    print(f"GL_MAX_COMPUTE_SHARED_MEMORY_SIZE = {glGetInteger(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE)}")
    print("")


def reorient_volume(volume, transforms, mask=None):
    n = volume.shape[0]

    volume_buffer_a.update(volume)
    volume_buffer_b.update(np.zeros_like(volume))
    volume_buffer_c.update(np.ones_like(volume) if mask is None else mask)

    volumes = list()
    for t in transforms:
        # empty buffer b
        cs_empty.bind()
        cs_empty.uniform4f("val", (0.0, 0.0, 0.0, 0.0))
        volume_buffer_b.bind_image_slot(1, 1)
        glDispatchCompute(n // 4, n // 4, n // 4)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
        cs_empty.unbind()

        # resample volume
        cs_reorient.bind()
        cs_reorient.uniform1i("N", n)
        volume_buffer_a.bind(0)
        volume_buffer_b.bind_image_slot(1, 1)
        volume_buffer_c.bind_image_slot(2, 0)
        cs_reorient.uniformmat4("tmat", t.tmat)
        cs_reorient.uniformmat4("rmat", t.rmat.T)
        glDispatchCompute(n, n, n)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        volume_buffer_b.bind()
        resampled = glGetTexImage(GL_TEXTURE_3D, 0, GL_RED, GL_FLOAT)
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT)

        volumes.append(np.reshape(resampled, (n, n, n)))

    return volumes


def tm2d_bind_templates(templates, masks=None):
    global tm_templates_ssbo, tm_n_templates, tm_template_size, tm_mask_ssbo, tm_sample_ssbo, tm_scores_ssbo

    tm_n_templates = len(templates)
    if tm_n_templates > 1000:
        print(f"tm2d_bind_templates: maximum number of templates is currently 1000! (tried {tm_n_templates}.)")
    tm_template_size = templates[0].shape[0]
    templates_data = np.concatenate([t.flatten() for t in templates])
    masks_data = np.concatenate([m.flatten() for m in masks])
    templates_data[masks_data == 0] = -100
    tm_templates_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, tm_templates_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, templates_data.nbytes, templates_data, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, tm_templates_ssbo)

    # tm_mask_ssbo = glGenBuffers(1)
    # mask_data = np.concatenate([m.flatten() for m in masks])
    # glBindBuffer(GL_SHADER_STORAGE_BUFFER, tm_mask_ssbo)
    # glBufferData(GL_SHADER_STORAGE_BUFFER, mask_data.nbytes, mask_data, GL_STATIC_DRAW)
    # glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, tm_mask_ssbo)

    tm_sample_ssbo = glGenBuffers(1)

    # scores = np.zeros((len(templates)), dtype=np.float32)
    # tm_scores_ssbo = glGenBuffers(1)
    # glBindBuffer(GL_SHADER_STORAGE_BUFFER, tm_scores_ssbo)
    # glBufferData(GL_SHADER_STORAGE_BUFFER, scores.nbytes, scores, GL_STATIC_DRAW)

def tm2d_set_n(n=32):
    set_tm2d_n(n)

def set_tm2d_n(n=32):
    global cs_tm_2d, cs_tm_2d_n
    cs_tm_2d = Shader()
    with open(os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)), "Pommie", "shaders", "template_match_2d.glsl")), 'r') as f:
        glsl_src = f.readlines()
    glsl_src_new_n = list()
    for line in glsl_src:
        if "#define N" in line:
            glsl_src_new_n.append(f"#define N {int(n)}")
        else:
            glsl_src_new_n.append(line)
    cs_tm_2d.compile_from_string(glsl_src_new_n)
    cs_tm_2d_n = n


def tm2d_single_slice(image, image_mask=None, z_score=True):
    image_buffer_a.update(image)
    image_buffer_b.update(image_mask)
    image_buffer_c.update(np.zeros((image.shape[0], image.shape[1], 4)))

    # bind shader
    cs_tm_2d.bind()
    cs_tm_2d.uniform1i("T", tm_n_templates)
    cs_tm_2d.uniform1i("z_score", 1 if z_score else 0)
    image_buffer_a.bind_image_slot(2, 0)
    image_buffer_b.bind_image_slot(3, 0)
    image_buffer_c.bind_image_slot(4, 1)

    glDispatchCompute(image.shape[1], image.shape[0], 1)

    glFinish()

    image_buffer_c.bind()
    scores = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
    scores = np.frombuffer(scores, dtype=np.float32).reshape((image.shape[0], image.shape[1], 4))
    return scores


# def tm2d_single_image(image):
#     image_data = image.flatten()
#     glBindBuffer(GL_SHADER_STORAGE_BUFFER, tm_sample_ssbo)
#     glBufferData(GL_SHADER_STORAGE_BUFFER, image_data.nbytes, image_data, GL_STATIC_DRAW)
#     glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, tm_sample_ssbo)
#     glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, tm_scores_ssbo)
#     glBindBuffer(GL_SHADER_STORAGE_BUFFER, tm_scores_ssbo)
#
#     cs_tm_2d_single_image.bind()
#     cs_tm_2d_single_image.uniform1i("T", tm_n_templates)
#     glDispatchCompute(1, 1, 1)
#
#     scores = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4 * tm_n_templates)
#     glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
#     scores = np.frombuffer(scores, dtype=np.float32)
#
#     return scores
#

def tm_bind_templates(templates, mask=None, similarity_function=2):
    """
    See match_sample_to_templates.
    when samples don't all fit in memory, the above function is inefficient: calling it multiple times with different
    batches of samples is inefficient, as identical templates are-reuploaded in each call. The functions 'tm_*' split
    up the different calls to allow for more efficient data shuffling.
    """
    global tm_templates_ssbo, tm_scores_ssbo, tm_n_templates, tm_sample_ssbo, tm_mask_ssbo, tm_scores_pointer

    # cs_tm.bind()
    # cs_tm.uniform1i("N", templates[0].n)

    cs_similarity.bind()
    cs_similarity.uniform1i("N", templates[0].n)
    cs_similarity.uniform1i("similarity_function_enum", similarity_function)

    # cs_tm.bind()

    tm_sample_ssbo = glGenBuffers(1)
    tm_n_templates = len(templates)
    templates_data = np.concatenate([t.data.flatten() for t in templates])
    tm_templates_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, tm_templates_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, templates_data.nbytes, templates_data, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, tm_templates_ssbo)

    scores = np.zeros((len(templates)), dtype=np.float32)
    tm_scores_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, tm_scores_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, scores.nbytes, scores, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, tm_scores_ssbo)

    tm_mask_ssbo = glGenBuffers(1)
    mask_data = np.ones_like(templates_data) if mask is None else mask.data.flatten()
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, tm_mask_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, mask_data.nbytes, mask_data, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, tm_mask_ssbo)


def tm_single_sample(sample):
    """
    See match_sample_to_templates.
    when samples don't all fit in memory, the above function is inefficient: calling it multiple times with different
    batches of samples is inefficient, as identical templates are-reuploaded in each call. The functions 'tm_*' split
    up the different calls to allow for more efficient data shuffling.

    returns: tuple(
        int: index of best matching template,
        float: score of best matching template
        float: score of worst matching template
        list: list of all scores (except zero-value template)
        )
    """
    global tm_sample_ssbo, tm_scores_a_bound

    cs_similarity.bind()

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, tm_sample_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, sample.data.nbytes, sample.data.flatten(), GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, tm_sample_ssbo)

    glDispatchCompute(tm_n_templates, 1, 1)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, tm_scores_ssbo)
    scores = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4 * tm_n_templates)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
    scores = np.frombuffer(scores, dtype=np.float32)

    return scores


def tm_bind_volume(volume):
    if isinstance(volume, np.ndarray):
        tm_volume_texture.update(volume)
        tm_scores_texture.update(np.zeros_like(volume))
    else:
        tm_volume_texture.update(volume.data)
        tm_scores_texture.update(np.zeros_like(volume.data))


    glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT)
    tm_volume_texture.bind(1)

    cs_tm.bind()


def tm_at_coordinate(coordinate):
    cs_tm.bind()
    cs_tm.uniform3i("C", coordinate)
    cs_tm.uniform1i("n_templates", tm_n_templates)
    tm_scores_texture.bind_image_slot(4, 2)
    glDispatchCompute(tm_n_templates, 1, 1)


def tm_read_scores():
    glFinish()
    return tm_scores_texture.read()


def tm_clean():
    global tm_templates_ssbo, tm_scores_ssbo, tm_n_templates, tm_sample_ssbo

    glDeleteBuffers(3, [tm_templates_ssbo, tm_scores_ssbo, tm_sample_ssbo])
    tm_n_templates, tm_templates_ssbo, tm_scores_ssbo, tm_sample_ssbo = (0, 0, 0, 0)
    cs_similarity.unbind()



def find_template_in_volume(volume, volume_mask, template, template_mask, transforms, z_score=True, stride=1, dimensionality=2, skip_binding=False, verbose=True):
    """
    volume: Pommie.Volume object
    template: Pommie.Particle object
    mask: Pommie.Volume object with same size as volume, matching occurs only where mask > 0

    returns: tuple (z_scores, indices, m_scores, r_scores); each element is an nd.array of same size as the input, with different values:
        'z_scores': z-score value (m_score of pixel minus mean of scores, divided by stdev).
        'indices':  index of best matching element in transforms array
        'm_scores': raw score maximum; maximum similarity score.
        'r_scores': raw score range: maximum minus minimum similarity score.
    """

    def printv(s):
        if verbose:
            print(s)

    if not skip_binding:
        t_start = time.time()
        templates = template.resample(transforms)
        template_masks = template_mask.resample(transforms)
        printv(f"Resampling templates and masks {time.time() - t_start:.3f} s.")

    if dimensionality == 2:
        if template.n != cs_tm_2d_n:
            print(f"find_template_in_volume using 2D-matching is currently set to use a template size of {cs_tm_2d_n}! (tried: {template.n})")
            return

        if not skip_binding:
            t_start = time.time()
            templates_2d = [t.data[cs_tm_2d_n//2, :, :] for t in templates]
            template_masks_2d = [m.data[cs_tm_2d_n//2, :, :] for m in template_masks]
            printv(f"Two-dimensionalizing {time.time() - t_start:.3f} s.")
            tm2d_bind_templates(templates_2d, template_masks_2d)
            printv(f"Binding templates {time.time() - t_start:.3f} s.")

        volume_mask.data[:cs_tm_2d_n // 2, :, :] = 0
        volume_mask.data[-cs_tm_2d_n // 2:, :, :] = 0
        volume_mask.data[:, :cs_tm_2d_n // 2, :] = 0
        volume_mask.data[:, -cs_tm_2d_n // 2:, :] = 0
        volume_mask.data[:, :, :cs_tm_2d_n // 2] = 0
        volume_mask.data[:, :, -cs_tm_2d_n // 2:] = 0
        if stride > 1:
            for k in range(stride - 1):
                volume_mask.data[1 + k::stride, :, :] = 0
                volume_mask.data[:, 1 + k::stride, :] = 0
                volume_mask.data[:, :, 1 + k::stride] = 0

        J = volume.data.shape[0]
        K = volume.data.shape[1]
        L = volume.data.shape[2]
        scores = np.zeros(volume.data.shape, dtype=np.float32)
        indices = np.zeros(volume.data.shape, dtype=np.float32)
        ts = time.time()
        it = 0

        for j in range(0, J, stride):
            _progress = (j / (J - template.n//2))
            print(f"TEMPLATE MATCHING: [{'█' * int(_progress * 50)}{'-' * (50 - int(50 * _progress))}] {_progress * 100.0:.1f}%", end="\r")
            it += 1
            if not np.any(volume_mask.data[j, :, :]):
                continue
            slice_tm_out = tm2d_single_slice(volume.data[j, :, :], volume_mask.data[j, :, :], z_score=z_score)[::stride, ::stride, :]

            slice_scores = slice_tm_out[:, :, 0]
            slice_indices = slice_tm_out[:, :, 1]
            if stride > 1:
                slice_scores = zoom(slice_scores, (K / slice_scores.shape[0], L / slice_scores.shape[1]), order=0)
                scores[j:j + stride, :, :] = slice_scores
                slice_indices = zoom(slice_indices, (K / slice_indices.shape[0], L / slice_indices.shape[1]), order=0)
                indices[j:j + stride, :, :] = slice_indices
            else:
                scores[j, :, :] = slice_scores
                indices[j, :, :] = slice_indices
        printv(f"Template matching: {time.time() - ts:.3f} s.")
        print(f"\r{' '*256}", end="\r")
        return scores, indices
    # else:
    #     B = templates.n
    #     coordinates = list()
    #     J, K, L = volume.data.shape
    #     N = 0
    #     for j in range(B // 2, J - B // 2, stride):
    #         for k in range(B // 2, K - B // 2, stride):
    #             for l in range(B // 2, L - B // 2, stride):
    #                 if volume_mask[j, k, l] > 0:
    #                     coordinates.append((j, k, l))
    #                     N += 1
    #     tm_bind_templates(templates, mask=template_mask, similarity_function=similarity_function)
    #     scores = np.zeros_like(volume.data) + 1e9
    #     indices = np.zeros(volume.data.shape, dtype=np.int16) - 1
    #
    #     n = 0
    #     ts = time.time()
    #     for j, k, l in coordinates:
    #         template_scores = tm_single_sample(volume.get_particle((j, k, l), n=B))
    #         scores[j-stride:j, k-stride:k, l-stride:l] = np.max(template_scores)
    #         indices[j-stride:j, k-stride:k, l-stride:l] = np.argmax(template_scores)
    #         n += 1
    #         if n % 100 == 0:
    #             printv(f"Making calls to GPU: {n}/{N} = {n / N * 100:.1f}%")
    #             glFinish()
    #             ect = datetime.datetime.now() + datetime.timedelta(seconds=(len(coordinates) - n) / n * (time.time() - ts))
    #             printv(f"Estimated time until completion: {(len(coordinates) - n) / n * (time.time() - ts):.1f} seconds: {ect.strftime('%Y-%m-%d %H:%M:%S')}")
    #     scores[scores == 1e9] = np.amin(scores)
    #     if return_indices:
    #         return scores, indices
    #     else:
    #         return scores


def match_sample_to_templates(templates, samples, mask=None, similarity_function=0):
    """
    templates: list of Particle objects
    sample: Particle object
    similarity_function: int, 0 for msqe or 1 for mae
    returns: tuple (matching_templates, sample, scores), lists of Particles, Particles, or floats, respectively, where the best match for sample[i] was matching_templates[i], with a score of scores[i]
    """

    cs_similarity.bind()
    cs_similarity.uniform1i("N", templates[0].n)
    cs_similarity.uniform1i("similarity_function_enum", similarity_function)

    templates_data = np.concatenate([t.data.flatten() for t in templates])
    templates_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, templates_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, templates_data.nbytes, templates_data, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, templates_ssbo)

    sample_ssbo = glGenBuffers(1)

    scores = np.zeros((len(templates)), dtype=np.float32)
    scores_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, scores_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, scores.nbytes, scores, GL_STREAM_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, scores_ssbo)

    mask_ssbo = glGenBuffers(1)
    mask_data = np.ones_like(templates_data) if mask is None else mask.data.flatten()
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mask_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, mask_data.nbytes, mask_data, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, mask_ssbo)

    matching_templates = list()
    scores_list = list()
    i = 0
    for sample in samples:
        i += 1
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, sample_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, sample.data.nbytes, sample.data.flatten(), GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, sample_ssbo)

        glDispatchCompute(len(templates), 1, 1)  # could be made more efficient by sequentially launching the resample and the scoring computes, instead of doing resample, download volume, upload volume, score.
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        # read scores:
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, scores_ssbo)
        scores = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, scores.nbytes)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        scores = np.frombuffer(scores, dtype=np.float32)
        idx = np.argmin(scores)
        scores_list.append(scores[idx])
        matching_templates.append(templates[idx])

    cs_similarity.unbind()
    glDeleteBuffers(3, [templates_ssbo, sample_ssbo, scores_ssbo])

    return matching_templates, samples, scores_list


def apply_separable_filter(particles, kernel):
    """
    :param particles: iterable that returns objects of type Pommie.Particle (e.g. a list, or a Pommie.Dataset)
    :param kernel: odd-sized array with float32 convolution kernel values.
    :return: 3d numpy array with filtered volume.
    """

    n = particles[0].n
    K = int(np.size(kernel) - 1) // 2

    cs_filter.bind()
    cs_filter.uniform1i("K", K)
    cs_filter.uniform1i("N", n)

    kernel_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, kernel_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, kernel.nbytes, kernel, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, kernel_ssbo)

    for p in particles:
        volume_buffer_a.update(p.data)
        volume_buffer_a.bind_image_slot(0, 2)

        cs_filter.uniform1i("direction", 0)  # X
        glDispatchCompute(n, n, n)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        cs_filter.uniform1i("direction", 1)  # Y
        glDispatchCompute(n, n, n)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        cs_filter.uniform1i("direction", 2)  # Z
        glDispatchCompute(n, n, n)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        volume = glGetTexImage(GL_TEXTURE_3D, 0, GL_RED, GL_FLOAT)
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT)

        p.data = np.reshape(volume, (n, n, n))
    glDeleteBuffers(1, [kernel_ssbo])
    return particles


def gaussian_filter(particles, sigma=15.68*2.0, kernel_size=10):
    """
    :param particles: :param particles: iterable that returns objects of type Pommie.Particle (e.g. a list, or a Pommie.Dataset)
    :param sigma: parameter of the gaussian, in Angstrom.
    :param kernel_size: 'radius' of the convolution kernel (filter window will be sized 2 * K + 1, where K is kernel_size)
    :return:
    """

    sigma_px = sigma / particles[0].apix
    kernel = np.exp(-np.linspace(-kernel_size, kernel_size, 2 * kernel_size + 1) ** 2 / sigma_px ** 2, dtype=np.float32)
    kernel /= np.sum(kernel)
    return apply_separable_filter(particles, kernel)


def center_of_mass(particles):
    # TODO: in batches, rather than all particles at once, to avoid overloading the GPU.

    cs_com.bind()
    cs_com.uniform1i("N", particles[0].n)

    volumes_data = np.concatenate([p.data.flatten() for p in particles])
    volumes_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, volumes_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, volumes_data.nbytes, volumes_data, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, volumes_ssbo)

    coms = np.zeros((len(particles) * 4), dtype=np.float32)
    coms_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, coms_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, coms.nbytes, coms, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, coms_ssbo)

    glDispatchCompute(len(particles), 1, 1)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, coms_ssbo)
    coms = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, coms.nbytes)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    centers_of_mass = np.frombuffer(coms, dtype=np.float32).reshape((len(particles), 4))[:, :3]

    cs_com.unbind()
    glDeleteBuffers(2, [volumes_ssbo, coms_ssbo])
    print(centers_of_mass)
    return centers_of_mass

# #compute
# #version 430
#
# #define MAX_TEMPLATES 500
# #define N 32
#
# layout(local_size_x = MAX_TEMPLATES, local_size_y = 1, local_size_z = 1) in;
#
# layout(std430, binding = 0) buffer templates_in {
#     float data[];
# } templates;
#
# layout(std430, binding = 1) buffer template_mask_in {
#     float data[];
# } mask;
#
# layout(binding = 2, r32f) uniform readonly image2D slice;
# layout(binding = 3, r32f) uniform readonly image2D slice_mask;
# layout(binding = 4, r32f) uniform writeonly image2D slice_score;
#
# uniform int T; // actual number of templates (may be lower than MAX_TEMPLATES)
#
# shared float img[N * N];
# shared float scores[MAX_TEMPLATES];
#
# void main(void)
# {
#     ivec2 c = ivec2(gl_GlobalInvocationID.xy);
#     int template_number = int(gl_LocalInvocationID.x);
#     float score = 10.0f;
#
#     if ((imageLoad(slice_mask, c).r > 0) && (template_number < T))
#     {
#         if (gl_LocalInvocationID.x == 0)
#         {
#             for (int j=0; j<N; j++)
#             {
#                 for (int k=0; k<N; k++)
#                 {
#                     img[j * N + k] = imageLoad(slice, c + ivec2(j - N/2, k - N/2)).r;
#                 }
#             }
#         }
#
#         barrier();
#
#         int template_base_idx = template_number * N * N;
#         float mu_template = 0.0f;
#         float mu_sample = 0.0f;
#         float n = 0.0f;
#
#         for (int j=0; j<N*N; j++)
#         {
#             if (mask.data[j] > 0)
#             {
#                 n += 1.0f;
#                 mu_template += templates.data[template_base_idx + j];
#                 mu_sample += img[j];
#             }
#         }
#
#         mu_template /= n;
#         mu_sample /= n;
#
#         float ts_dot = 0.0f;
#         float t_mag = 0.0f;
#         float s_mag = 0.0f;
#
#         for (int j=0; j<N*N; j++)
#         {
#             if (mask.data[j] > 0)
#             {
#                 ts_dot += (templates.data[template_base_idx + j] - mu_template) * (img[j] - mu_sample);
#                 t_mag += pow(templates.data[template_base_idx + j] - mu_template, 2);
#                 s_mag += pow(img[j] - mu_sample, 2);
#             }
#         }
#
#         score = ts_dot / (sqrt(t_mag) * sqrt(s_mag));
#     }
#
#     scores[template_number] = score;
#
#     memoryBarrierShared();
#     barrier();
#
#     float max_score = scores[0];
#     if (gl_LocalInvocationID.x == 0)
#     {
#         float max_score = scores[0];
#         for (int j=1; j<T; j++)
#         {
#             max_score = max(max_score, scores[j]);
#         }
#         imageStore(slice_score, c, vec4(max_score));
#     }
# }
