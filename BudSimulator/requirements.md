##Objective

The objective of Bud simulator is to build a layer around GenZ with additional capabilities & an API layer on top of it, so that we can add new hardwares, models (directly from HF & through batch processing), quantisation methods, different parallelisation strategies etc.

1. Build an easy to consume OpenAPI compatiable RESTful APIs for different functionalities that existis with in GenZ currently, and will be added in the future.
2. Add a persistence layer for GenZ functionalities now through SQLLite - Ensuring easy to port persistence for Models, Hardware, quantisation etc.
3. Add support for CPUs, Different Inference engine features etc.