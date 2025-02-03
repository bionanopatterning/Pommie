[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/bionanopatterning/Pommie/blob/main/Pom/license.txt)
[![Downloads](https://img.shields.io/pypi/dm/Pommie)](https://pypi.org/project/Pommie/)
![Last Commit](https://img.shields.io/github/last-commit/bionanopatterning/Pommie)


## Pommie

This is a supplementary repository to the main [Pom](https://github.com/bionanopatterning/Pom), containing the code for area-selective GPU-accelerated 2D real-space template matching (ASTM) in OpenGL. As discussed in the article, this implementation could be much improved. For the time being it could be a useful resource for teaching and preliminary tests. Hopefully we will be able to develop a more sophisticated library for ASTM in the future.

### Examples

See 'example_lightbulb.py' and 'example_npc.py' for the scripts used in the ASTM experiments in the Pom article. The main high-level function for ASTM is 'find_template_in_volume' in compute.py; for implementation details, see compute.tm2d_bind_templates, compute.tm2d_single_slice, and shaders/template_match_2d.glsl   

### Author

Mart G. F. Last

mlast@mrc-lmb.cam.ac.uk