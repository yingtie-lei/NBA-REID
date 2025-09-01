# NBA Re-ID Annotation Tool

This repository provides an annotation tool for **NBA Re-Identification (Re-ID)**, built on top of **[Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)** and **[Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO)**.

## Installation

Before running the tool, please make sure you have correctly set up the environments for both **Grounded-SAM-2** and **Grounding-DINO** by following their official installation instructions.

- [Grounded-SAM-2 Installation Guide](https://github.com/IDEA-Research/Grounded-SAM-2)
- [Grounding-DINO Installation Guide](https://github.com/IDEA-Research/GroundingDINO)

## Usage

Once the environments are set up, you can start the annotation pipeline by running:

```bash
python pipeline.py
```

## Additional Recommendation

It is **highly recommended** to run this tool on a Linux server.
Running on Windows may cause unexpected errors such as:

```bash
NameError: name '_C' is not defined
```
