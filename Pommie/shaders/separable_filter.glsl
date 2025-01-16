#compute
#version 430

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0, r32f) uniform image3D vol;

layout(std430, binding = 1) buffer kernel_in {
    float val[];
} kernel;

uniform int N; // box size.
uniform int K; // kernel radius (full size is 2 * K + 1)
uniform int direction;

void main(void)
{
    ivec3 idx = ivec3(gl_GlobalInvocationID);

    // Apply filter along X
    float value = 0.0f;
    float weight = 0.0f;
    if (direction == 0)
    {
        for (int i=-K; i<=K; i++)
        {
            value += kernel.val[i + K] * imageLoad(vol, ivec3(idx.x + i, idx.y, idx.z)).r;
            weight += kernel.val[i + K];
        }
    }
    else if (direction == 1)
    {
        for (int i=-K; i<=K; i++)
        {
            value += kernel.val[i + K] * imageLoad(vol, ivec3(idx.x, idx.y + i, idx.z)).r;
            weight += kernel.val[i + K];
        }
    }
    else
    {
        for (int i=-K; i<=K; i++)
        {
            value += kernel.val[i + K] * imageLoad(vol, ivec3(idx.x, idx.y, idx.z + i)).r;
            weight += kernel.val[i + K];
        }
    }

    barrier();
    imageStore(vol, idx, vec4(value / weight, 0.0f, 0.0f, 0.0f));
}
