#compute
#version 430

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 2) buffer scores {
    float value[];
} similarity_scores;

layout(binding = 4, r32f) writeonly uniform image3D scores_volume;

uniform ivec3 C;
uniform int n_templates;

void main(void)
{
    float max_score = -1e38;
    for (int i=0; i<n_templates; i++)
    {
        float score = similarity_scores.value[i];
        if (score > max_score)
        {
            max_score = score;
        }
    }

    imageStore(scores_volume, C, vec4(max_score, 0.0, 0.0, 0.0));
}
