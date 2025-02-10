# Todo
- Ne pas oublier de mettre un graph de comparaison de performance sur jeux de donnée classiques?
- Docu avec sphinx? Voir quel format de docstrings, extension vscode docstrings auto?
- Problème du training GPU poor:
    - Checkpointing: Make sure your training loop saves progress frequently so you can resume after a session timeout.
    - Mixed Precision: Leverage PyTorch’s AMP to take advantage of the T4’s tensor cores.

# Requirements
- Python 3.12.8
- Other dependencies listed in `requirements.txt`

# References
- [Distill](https://distill.pub/2018/feature-wise-transformations/)
- [Arxiv](https://arxiv.org/pdf/1709.07871)
- [Github](https://github.com/ethanjperez/film)
