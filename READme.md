# Todo
- Rapport
- Sur quel dataset? Idem que l'article/Réduction de types d'objets/Réduction de résolution
- Quelle application? Quelle interactivité?
    - Visualisation des tSNE?
    - Appli qui tourne juste sur CPU laptop? Ou faire aussi une partie plus complexe?
        - Plutôt faire au moins un gros modèle efficace, pré-entraîné avec les poids importés dans streamlit, et un petit modèle entraînable laptop CPU même si réusltats dégeulasses
    - Idées interaction :
        - Visualisation des tSNE, visualisation de ce que le MLP voit, et lien entre les 2?
        - Un modèle classique, un modèle avec zero-shot ?
        - Question "fait main" avec un drop down menu pour poser des questions sur l'image

- Ne pas oublier de mettre un graph de comparaison de performance sur jeux de donnée classiques?
- Docu avec sphinx? Voir quel format de docstrings, extension vscode docstrings auto?
- Problème du training GPU poor:
    - Checkpointing: Make sure your training loop saves progress frequently so you can resume after a session timeout.
    - Mixed Precision: Leverage PyTorch's AMP to take advantage of the T4's tensor cores.

# Requirements
- Python 3.12.8
- Other dependencies listed in `requirements.txt`

# References
- [Distill](https://distill.pub/2018/feature-wise-transformations/)
- [Arxiv](https://arxiv.org/pdf/1709.07871)
- [Github](https://github.com/ethanjperez/film)
