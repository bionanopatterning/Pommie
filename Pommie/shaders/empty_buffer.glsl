#compute
#version 430

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 1, r32f) writeonly uniform image3D volume;

uniform vec4 val;

void main(void)
{
    ivec3 idx = ivec3(gl_GlobalInvocationID.xyz);
    imageStore(volume, idx, val);
}