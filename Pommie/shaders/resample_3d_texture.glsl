#compute
#version 430

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) uniform sampler3D vol_in;
layout(binding = 1, r32f) writeonly uniform image3D vol_out;
layout(binding = 2, r32f) readonly uniform image3D mask;

uniform mat4 tmat;
uniform mat4 rmat;
uniform int N;

void main(void)
{
    ivec3 idx = ivec3(gl_GlobalInvocationID.xyz);

    vec4 c = vec4(idx, 1.0f);

    // Translate, to center particle
    c = tmat * c;

    // Rotate in world space (and before, offset coordinates s.t. origin is in center of volume)
    // Then rotate in local space
    // (rotation matrix contais both operations)
    c.xyz -= float(N) / 2;
    c = rmat * c;
    c.xyz += float(N) / 2;

    // Add 0.5f to sample center instead of corners of pixels.
    c += 0.5f;

    float val = texture(vol_in, c.xyz / N).r;
    float mask_val = imageLoad(mask, idx).r;
    imageStore(vol_out, idx, vec4(val * mask_val, 0.0f, 0.0f, 0.0f));
}

