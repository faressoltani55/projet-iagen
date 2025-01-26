# Projet IA Générative

Ce répo représente mon travail fait pour le projet
de la matière IA Générative du Master Big Data.

Pour faire tourner le projet il faut:
* Installer les requirements comme suit:
```
pip install -r requirements.txt
```

* Ajouter le modèle GPT2 fine tuné dans le dossier
**models/gpt2**.

* Ensuite lancer l'application:
```
python -m streamlit run Home.py
```

L'application doit générer deux poèmes:
1. Une générée par GPT2 fine tuné.
2. Une générée par GPT NEO 125M.

Ensuite, il y a un bloc qui affiche les scores bleu et rouge.