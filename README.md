# sy32-pers
Projet visant à encadrer les silhouettes de personnes présents dans une image.

Ce projet a été réalisé en 2017 en utilisant principalement le package sklearn de Python.

Après avoir testé plusieurs modèle, celui avec lequel nous avons obtenu les meilleurs résultats est le C-Support Vector Classification.

Le but de ce projet était d'entrainer notre modèle sur les 200 images du dossier "train" dont on possède les étiquettes et de le tester ensuite sur les 109 images du dossier test, dont on ne connait pas les étiquettes et qu'il faut donc générer.
Les performances de notre modèle était ensuite testé par un programme qui possédait les étiquettes des images de test et qui les comparait au fichier étiquette que nous avions généré. 
