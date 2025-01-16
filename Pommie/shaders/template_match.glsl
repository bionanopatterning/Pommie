#compute
#version 430

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer templates_in {
    float data[];
} template_volumes;

layout(binding = 1) uniform sampler3D input_volume;

layout(std430, binding = 3) buffer mask_in {
    float data[];
} mask;

layout(binding = 4, r32f) writeonly uniform image3D scores_volume;

shared float shared_scores[1024];
uniform int n_templates;
uniform int N;
uniform ivec3 C;

float volume_value(ivec3 c)
{
    return texelFetch(input_volume, c, 0).r;
}

void main(void)
{
    uint template_id = gl_GlobalInvocationID.x;
    uint template_base_idx = template_id * N * N * N;
    ivec3 offset_coordinate = C - ivec3(N, N, N) / 2;
    float score = 0.0f;

    // Initialize shared_scores only once per workgroup
    if (gl_LocalInvocationIndex == 0) {
        for (int i = 0; i < 1024; i++)
        {
            shared_scores[i] = -1e38; // Initial value for finding maximum
        }
    }

    barrier();

    float mu_template = 0.0f;
    float mu_sample = 0.0f;
    float n = 0.0f;
    for (uint i=0; i<N; i++)
    {
        for (uint j=0; j<N; j++)
        {
            for (uint k=0; k<N; k++)
            {
                uint lin_idx = i * N * N + j * N + k;
                if (mask.data[lin_idx] > 0)
                {
                    mu_template += template_volumes.data[template_base_idx + lin_idx];
                    mu_sample += volume_value(offset_coordinate + ivec3(i, j, k)).r;
                    n += 1.0f;
                }
            }
        }
    }

    mu_template /= n;
    mu_sample /= n;

    float ts_dot = 0.0f;
    float t_mag = 0.0f;
    float s_mag = 0.0f;
    for (uint i=0; i<N; i++)
    {
        for (uint j=0; j<N; j++)
        {
            for (uint k=0; k<N; k++)
            {
                uint lin_idx = i * N * N + j * N + k;
                if (mask.data[lin_idx] > 0)
                {
                    float v = volume_value(offset_coordinate + ivec3(i, j, k)).r;
                    ts_dot += (template_volumes.data[template_base_idx + lin_idx] - mu_template) * (v - mu_sample);
                    t_mag += pow((template_volumes.data[template_base_idx + lin_idx] - mu_template), 2);
                    s_mag += pow(v - mu_sample, 2);
                }
            }
        }
    }

    score = ts_dot / (sqrt(t_mag) * sqrt(s_mag));
    shared_scores[template_id] = score;

    memoryBarrierShared();
    barrier();

    if (gl_GlobalInvocationID.x == 0)
    {
        float max_score = -1e38;
        for (int i=0; i<n_templates; i++)
        {
            if (shared_scores[i] > max_score)
            {
                max_score = shared_scores[i];
            }
        }

        imageStore(scores_volume, C, vec4(max_score, 0.0, 0.0, 0.0));
    }
}