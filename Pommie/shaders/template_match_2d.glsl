#compute
#version 430

#define MAX_TEMPLATES 1000
#define N 32

layout(local_size_x = MAX_TEMPLATES, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer templates_in {
    float data[];
} templates;


layout(binding = 2, r32f) uniform readonly image2D slice;
layout(binding = 3, r32f) uniform readonly image2D slice_mask;
layout(binding = 4, r32f) uniform writeonly image2D slice_score;

uniform int T; // actual number of templates (may be lower than MAX_TEMPLATES)

shared float img[N * N];
shared float scores[MAX_TEMPLATES];

void main(void)
{
    ivec2 c = ivec2(gl_WorkGroupID.xy);
    int template_number = int(gl_LocalInvocationID.x);
    float score = 0.0f;

    if ((imageLoad(slice_mask, c).r > 0) && (template_number < T))
    {
        if (gl_LocalInvocationID.x == 0)
        {
            for (int j=0; j<N; j++)
            {
                for (int k=0; k<N; k++)
                {
                    img[j * N + k] = imageLoad(slice, c + ivec2(j - N/2, k - N/2)).r;
                }
            }
        }

        barrier();

        int template_base_idx = template_number * N * N;
        float mu_template = 0.0f;
        float mu_sample = 0.0f;
        float n = 0.0f;

        for (int j=0; j<N*N; j++)
        {
            float t = templates.data[template_base_idx + j];
            if (t > -100) // when uploading templates, the mask is applied s.t. templates values are -100 where the mask was 0.
            {
                n += 1.0f;
                mu_template += t;
                mu_sample += img[j];
            }
        }

        mu_template /= n;
        mu_sample /= n;

        float ts_dot = 0.0f;
        float t_mag = 0.0f;
        float s_mag = 0.0f;

        for (int j=0; j<N*N; j++)
        {
            float t = templates.data[template_base_idx + j];
            if (t > -100)
            {
                ts_dot += (t - mu_template) * (img[j] - mu_sample);
                t_mag += pow(t - mu_template, 2);
                s_mag += pow(img[j] - mu_sample, 2);
            }
        }

        score = ts_dot / (sqrt(t_mag) * sqrt(s_mag));
    }

    scores[template_number] = score;

    memoryBarrierShared();
    barrier();

    float max_score = scores[0];
    if (gl_LocalInvocationID.x == 0)
    {
        float max_score = scores[0];
        float max_index = 0.0f;
        for (int j=1; j<T; j++)
        {
            if (scores[j] > max_score)
            {
                max_score = scores[j];
                max_index = float(j);
            }
        }

        imageStore(slice_score, c, vec4(max_score * 0.5 + 0.5 + max_index * 100));
    }
}
