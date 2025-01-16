#compute
#version 430

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer volumes_in {
    float data[];
} volumes;

layout(std430, binding = 1) buffer center_of_mass {
    vec4 values[];
} com;

uniform int N;

void main(void)
{
    // compute COM for one volume.
    uint volume_id = gl_GlobalInvocationID.x;
    uint base_idx = volume_id * N * N * N;

    vec3 cmp = vec3(0.0f, 0.0f, 0.0f);
    float mp = 0.0f;
    vec3 cmn = vec3(0.0f, 0.0f, 0.0f);
    float mn = 0.0f;

    for (int x=0;x<N;x++)
    {
        for (int y=0;y<N;y++)
        {
            for (int z=0;z<N;z++)
            {
                uint idx = base_idx + x + y * N + z * N * N;
                float mass = volumes.data[idx];
                if (mass > 0.0)
                {
                    cmp += vec3(x, y, z) * mass;
                    mp += mass;
                }
                else
                {
                    cmn += vec3(x, y, z) * -mass;
                    mn -= mass;
                }
            }
        }
    }

    vec3 cm_overall = (cmp + cmn) / (mp + mn);
    com.values[volume_id] = vec4(cm_overall, 0.0f);
}