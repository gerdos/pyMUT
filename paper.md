---
title: 'Gala: A Python package for galactic dynamics' tags:

- Python
- biology
- bioinformatics authors:
- name: Erdős Gábor orcid: 0000-0001-6218-5192 affiliation: 1 affiliations:
- name: Department of Biochemistry, Eötvös Loránd University, Pázmány Péter stny 1/c, Budapest H-1117, Hungary. index: 1
  date: 23 March 2021 bibliography: paper.bib

# Summary

Mutations are the essential driving force of evolutions as well as the basis of critical diseases. PyMut is a
method to introduce mutations computationally into PBDs, the most used three-dimensional protein structures files. PyMUT
uses the 2010 Dunbrack rotamer library [REF], which is a collection of the most likely orientations a residue can be
found in biological systems.

PyMut features multiple options to select the desired rotamer, as the highest probability rotamer is often not the
desired one. To account for structural clashes PyMut can select a rotamer based on the Van deer Waals energy it would
obtain from neighboring residues and select the one with the minimal energy. For large scale studies rotamers can be
selected randomly based on their respective probabilities from the library.

PyMut is distributed as an importable Python3 library, which is designed for large scale studies. The software has only
one dependency - the numpy library - which is one of the most used external python libraries as of today.

# Statement of need

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python enables wrapping low-level languages (e.g.,
C) for speed without losing flexibility or ease-of-use in the user-interface. The API for `Gala` was designed to provide
a class-based and user-friendly interface to fast (C or Cython-optimized) implementations of common operations such as
gravitational potential and force evaluation, orbit integration, dynamical transformations, and chaos indicators for
nonlinear dynamics. `Gala` also relies heavily on and interfaces well with the implementations of physical units and
astronomical coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by students in courses on gravitational dynamics or
astronomy. It has already been used in a number of scientific publications [@Pearson:2017] and has also been used in
graduate courses on Galactic dynamics to, e.g., provide interactive visualizations of textbook material [@Binney:2008].
The combination of speed, design, and support for Astropy functionality in `Gala` will enable exciting scientific
explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Description

# Examples

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l} 0\textrm{ if } x < 0\cr 1\textrm{ else} \end{array}\right.$$

You can also use plain \LaTeX for equations \begin{equation}\label{eq:fourier} \hat f(\omega) = \int_{-\infty}^{\infty}
f(x) e^{i\omega x} dx \end{equation} and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred citation) then you can do it
with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:

- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong Oh, and support from Kathryn Johnston
during the genesis of this project.

# References