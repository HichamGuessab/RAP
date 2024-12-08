# Reconnaissance automatique de la parole

Ce projet est un TP de reconnaissance automatique de la parole.
L'objectif est de reconnaître un mot prononcé à partir de fichiers audio en comparant ses caractéristiques acoustiques avec celles des mots-clés enregistrés dans un dictionnaire.

Le système utilise des coefficients cepstraux (MFCC) extraits des signaux audio et l'algorithme DTW (Dynamic Time Warping) pour mesurer la similarité entre les signaux.

## Installation

1. install poetry
```bash
pip install poetry
```

2. install dependencies
```bash
poetry install
```

3. run the project
```bash
poetry run python main.py
```