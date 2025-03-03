# Vision Encoder Model Capacity Variants

This document provides a summary of the different capacity variants of the Vision Encoder model created for experimental purposes.

## Model Variants Overview

| Model Name | Relative Size | Total Capacity | Total FLOPs | Communication Volume |
|------------|---------------|---------------|-------------|---------------------|
| VisionEncoder_Small | 60% | ~5.6 GB | ~17.7 GFLOPS | ~2.16 GB |
| VisionEncoder_Medium | 80% | ~7.5 GB | ~23.4 GFLOPS | ~2.85 GB |
| VisionEncoder_Original | 100% | ~9.35 GB | ~29.5 GFLOPS | ~3.5 GB |
| VisionEncoder_Large | 120% | ~11.2 GB | ~35.4 GFLOPS | ~4.25 GB |
| VisionEncoder_XLarge | 150% | ~14.06 GB | ~44.1 GFLOPS | ~5.3 GB |

## Scaling Methodology

Each variant was created by applying a consistent scaling factor to all components of the original model:

1. **Component Capacities**: Memory requirements of each component scaled proportionally
2. **Computational Requirements**: FLOPS per sample scaled by the same factor
3. **Communication Volumes**: Inter-partition communication volumes scaled accordingly
4. **Partition Structure**: All models maintain the same partition structure for fair comparison

## Running Experiments

To use these model variants in your experiments, use the `--model` parameter with the appropriate XML file:

```bash
python swift_example.py --model settings/visionencoder_small.xml --cluster settings/clusterwith5member.xml
python swift_example.py --model settings/visionencoder_medium.xml --cluster settings/clusterwith5member.xml
python swift_example.py --model settings/visionencoder.xml --cluster settings/clusterwith5member.xml
python swift_example.py --model settings/visionencoder_large.xml --cluster settings/clusterwith5member.xml
python swift_example.py --model settings/visionencoder_xlarge.xml --cluster settings/clusterwith5member.xml
```

## Expected Outcomes

These model variants enable several interesting experiments:

1. **Scalability Analysis**: How optimization performance changes with model size
2. **Resource Constraints**: How different cluster configurations handle increasing model sizes
3. **Partition Strategy Evolution**: How optimal partition strategies change as model size increases
4. **Optimization Algorithm Robustness**: How well different algorithms handle various model sizes
5. **Cost-Performance Trade-offs**: Establishing the relationship between model size and execution performance

## Notes

- Memory overhead and device utilization parameters may need adjustment for very large or small models
- The largest model variant may require clusters with sufficient total memory capacity
- Some scaling factors may need fine-tuning based on specific hardware constraints
