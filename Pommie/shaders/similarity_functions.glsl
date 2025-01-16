#compute
#version 430

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer templates_in {
    float data[];
} template_volumes;

layout(std430, binding = 1) buffer sample_in {
    float data[];
} sample_volume;

layout(std430, binding = 2) buffer values_out {
    float value[];
} similarity_scores;

layout(std430, binding = 3) buffer mask_in {
    float data[];
} mask;

uniform int N;
uniform int similarity_function_enum; // 0 for msqe, 1 for mae, 2 for maskless msqe (temporary workaround)

void main(void)
{
    uint template_id = gl_GlobalInvocationID.x;
    uint template_base_idx = template_id * N * N * N;
    float score = 0.0;

    if (similarity_function_enum == 0) // Squared error
    {
        for (int i=0; i<N*N*N; i++)
        {
            score += mask.data[i] * pow((template_volumes.data[template_base_idx + i] - sample_volume.data[i]), 2);
        }
        score = -score;
    }
    else if (similarity_function_enum == 1) // Absolute error masked
    {
        for (int i=0; i<N*N*N; i++)
        {
            score += mask.data[i] * abs(template_volumes.data[template_base_idx + i] - sample_volume.data[i]);
        }
        score = -score;
    }
    else if (similarity_function_enum == 2) //  Roseman local correlation
    {
        float mu_template = 0.0f;
        float mu_sample = 0.0f;
        float n = 0.0f;
        for (int i=0; i<N*N*N; i++)
        {
            if (mask.data[i] > 0)
            {
                mu_template += template_volumes.data[template_base_idx + i];
                mu_sample += sample_volume.data[i];
                n += 1.0f;
            }
        }
        mu_template /= n;
        mu_sample /= n;

        float ts_dot = 0.0f;
        float t_mag = 0.0f;
        float s_mag = 0.0f;
        for (int i=0; i<N*N*N; i++)
        {
            if (mask.data[i] > 0)
            {
                ts_dot += (template_volumes.data[template_base_idx + i] - mu_template) * (sample_volume.data[i] - mu_sample);
                t_mag += pow((template_volumes.data[template_base_idx + i] - mu_template), 2);
                s_mag += pow((sample_volume.data[i] - mu_sample), 2);
            }
        }

        score = ts_dot / (sqrt(t_mag) * sqrt(s_mag));
    }
    else if (similarity_function_enum == 3)  // Simplified Roseman for zero-mean input volumes
    {
        float ts_dot = 0.0f;
        float t_mag = 0.0f;
        float s_mag = 0.0f;
        for (int i=0; i<N*N*N; i++)
        {
            if (mask.data[i] > 0)
            {
                ts_dot += template_volumes.data[template_base_idx + i] * sample_volume.data[i];
                t_mag += pow((template_volumes.data[template_base_idx + i]), 2);
                s_mag += pow((sample_volume.data[i]), 2);
            }
        }
        score = ts_dot / (sqrt(t_mag) * sqrt(s_mag));
    }
    similarity_scores.value[template_id] = score;
}