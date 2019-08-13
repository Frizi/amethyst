#ifndef EVSM_FRAG
#define EVSM_FRAG

// TODO: move into spec constants
const float positive_exponent = 800.0;
const float negative_exponent = 100.0;

vec2 get_evsm_exponents(float z_scale) {
    vec2 lightSpaceExponents = vec2(positive_exponent, negative_exponent);
    
    // Make sure exponents stay consistent in light space regardless of partition
    // scaling. This prevents the exponentials from ever getting too rediculous
    // and maintains consistency across partitions.
    // Clamp to maximum range of fp32 to prevent overflow/underflow
    return min(lightSpaceExponents / vec2(z_scale, z_scale), vec2(42.0f, 42.0f));
}

// Input depth should be in [0, 1]
vec2 warp_depth(float depth, vec2 exponents)
{
    // Rescale depth into [-1, 1]
    depth = 2.0f * depth - 1.0f;
    float pos =  exp( exponents.x * depth);
    float neg = -exp(-exponents.y * depth);
    return vec2(pos, neg);
}

float chebyshev_upper_bound(vec2 moments, float mean, float minVariance)
{
    // Compute variance
    float variance = moments.y - (moments.x * moments.x);
    variance = max(variance, minVariance);
    
    // Compute probabilistic upper bound
    float d = mean - moments.x;
    // float p_max = variance / (variance + (d * d));
    float p_max = smoothstep(0.20f, 1.0f, variance / (variance + d*d));
    // One-tailed Chebyshev
    return (mean <= moments.x ? 1.0f : p_max);
}

float shadow_contribution(sampler2DArray shadow_map, vec2 tex_coord, float depth, float z_scale, uint textureArrayIndex)
{
    vec2 exponents = get_evsm_exponents(z_scale);
    vec2 warpedDepth = warp_depth(depth, exponents);
    
    vec4 occluder = texture(shadow_map, vec3(tex_coord, textureArrayIndex));

    // Derivative of warping at depth
    // TODO: Parameterize min depth variance
    vec2 depthScale = 0.00001f * exponents * warpedDepth;
    vec2 minVariance = depthScale * depthScale;
    
    float pos_contrib = chebyshev_upper_bound(occluder.xz, warpedDepth.x, minVariance.x);
    float neg_contrib = chebyshev_upper_bound(occluder.yw, warpedDepth.y, minVariance.y);
    return min(pos_contrib, neg_contrib);
}

#endif
