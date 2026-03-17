---
title: "Rapport de Projet : Intelligence Artificielle Embarquée"
author: "RABET Gilles, ZAJAC Matthieu"
date: "Mars 2026"
---

# Rapport de Projet : IA Embarquée pour la Maintenance Prédictive

## 1. Résumé du Projet

Ce projet porte sur le développement et l'implémentation d'une solution d'apprentissage profond (Deep Learning) dédiée à la maintenance prédictive. En nous appuyant sur le jeu de données AI4I 2020 Predictive Maintenance Dataset, nous avons conçu un réseau de neurones (DNN) capable d'identifier des défaillances industrielles. 

L'objectif principal réside dans le déploiement de ce modèle sur une cible matérielle contrainte : le microcontrôleur STM32L4R9, en exploitant la chaîne d'outils STM32Cube.AI.

Le flux de travail a été divisé en cinq phases majeures :
1.  **Ingénierie des données** : Nettoyage et normalisation.
2.  **Modélisation** : Architecture et entraînement du réseau de neurones.
3.  **Validation** : Analyse des métriques de performance.
4.  **Optimisation** : Conversion du modèle pour les contraintes de l'embarqué.
5.  **Intégration** : Déploiement final sur le hardware cible.

### Structure du répertoire
```text
projet_IA
├── CubeIDE                             # Projet STM32CubeIDE complet
│   ├── Core
│   │   ├── Inc
│   │   │   └── main.h                  # Headers principaux
│   │   └── Src
│   │       └── main.c                  # Logique applicative en C
│   ├── app-x-cube.c                    # Interface d'appel de l'IA
│   └── app-x-cube.h
├── TP_IA_EMBARQUEE.ipynb               # Pipeline d'entraînement
├── model                               # Artefacts du modèle
│   ├── failure_predictive_model.h5     # Poids du modèle entraîné
│   ├── X_test.npy                  # Échantillons de test
│   └── Y_test.npy                  # Labels de test
├── README.md                       # Documentation technique
├── images                          # Ressources visuelles du rapport
└── Communication_STM32_NN.py       # Utilitaire de communication série

```
## 2. Introduction

Le déploiement massif de l'IoT et de l'analyse de données au sein de l'Industrie 4.0 a ouvert la voie à une gestion plus intelligente des actifs industriels. L'enjeu est de délaisser les modèles de maintenance réactifs, coûteux et imprévisibles, au profit d'une maintenance prédictive proactive.

![Schéma : Flux de données Edge AI](./images/schema_flux.png)
> *Note : Il est suggéré d'insérer ici un schéma montrant le flux : Capteur -> Microcontrôleur (IA locale) -> Action corrective immédiate.*

En déportant l'intelligence directement "à la périphérie" (Edge Computing), les entreprises minimisent la latence de détection et optimisent la durée de vie des équipements. Ce projet vise à répondre à la problématique suivante :



## 3. Configuration et Environnement Technique

### 3.1. Écosystème Logiciel
Le développement a été structuré autour de trois outils complémentaires :

* **Jupiter Notebook** : Utilisé pour l'exploration de données et l'entraînement du modèle.
* **STM32CubeIDE** : Environnement pour le développement du firmware et la gestion des ressources matérielles de la carte.
* **X-CUBE-AI** : Outil de conversion permettant de transformer les modèles de haut niveau (Keras/H5) en code C optimisé pour le processeur ARM Cortex-M.

### 3.2. Guide de mise en œuvre rapide (QuickStart)
1.  **Entraînement** : Exécuter le notebook pour générer l'archive `.h5`.
2.  **Conversion** : Importer le modèle dans le plugin X-CUBE-AI de CubeIDE.
3.  **Analyse** : Valider l'occupation de la RAM/Flash et générer les bibliothèques C.
4.  **Déploiement** : Compiler et flasher le binaire sur la carte STM32L4R9.
5.  **Test** : Lancer `ports.py` sur l'ordinateur hôte pour visualiser les prédictions en temps réel.

## 4. Analyse du Jeu de Données

### 4.1. Structure et Composition

Le projet repose sur le jeu de données AI4I 2020 Predictive Maintenance Dataset, une ressource de référence simulant des environnements industriels réels. Ce dataset se compose de 10 000 entrées, chacune représentant l'état d'une machine à un instant précis à travers 14 variables distinctes.

On y retrouve des mesures physiques cruciales pour la maintenance :
* **Conditions ambiantes** : Température de l'air [K].
* **Paramètres de fonctionnement** : Température du processus [K], Vitesse de rotation [rpm], Couple [Nm].
* **Indicateur d'usage** : Usure de l'outil [min].

Les pannes sont initialement répertoriées selon cinq catégories techniques : **TWF** (Usure de l'outil), **HDF** (Dissipation thermique), **PWF** (Puissance), **OSF** (Surcharge) et **RNF** (Pannes aléatoires).

![Figure 2 : Aperçu des premières lignes et colonnes du dataset](./images/Premieres_lignes_tableau.png)


### 4.2. Étude Statistique et Déséquilibre des Classes

L'exploration initiale du label principal (`Machine failure`) révèle un déséquilibre important. Comme illustré dans le graphique ci-dessous, la vaste majorité des données correspond à un fonctionnement normal de la machine.

![Figure 3 : Répartition des états de fonctionnement (Failure vs Non-Failure)](./images/Machine_Status.png)


Les pannes ne représentent qu'environ 3,5 % du dataset. Ce phénomène constitue un défi majeur pour l'apprentissage du modèle : un réseau de neurones non optimisé pourrait atteindre une précision de 96 % simplement en prédisant systématiquement "pas de panne", sans pour autant détecter les défaillances réelles.

### 4.3. Identification des Incohérences et Stratégie de Nettoyage

Une analyse croisée approfondie entre l'indicateur de panne global (`Machine failure`) et les types de défaillances spécifiques a révélé des anomalies structurelles dans le dataset. Ces observations sont cruciales pour comprendre les futures étapes de prétraitement.

#### Le paradoxe des pannes aléatoires (RNF)
En comparant l'ensemble du jeu de données aux seules machines déclarées en panne, nous observons une divergence majeure concernant les **Random Failures (RNF)** :
* **Sur l'ensemble du dataset** : On dénombre **19 occurrences** de l'étiquette RNF.
* **Sur les machines en panne (`Machine failure = 1`)** : Seule **1 occurrence** de RNF est conservée.

Cela signifie que dans 18 cas sur 19 (soit environ 95%), une panne aléatoire a été enregistrée sans que la machine ne soit considérée comme étant en état de défaillance globale. Cette déconnexion statistique entre la cause (RNF) et l'effet (`Machine failure`) rend cette variable inexploitable, voire néfaste, pour l'apprentissage du modèle.

![Figure 4 : Comparaison de la distribution des pannes - Données brutes vs Machines en panne](./images/Failure_Types.png)

#### Identification des pannes "orphelines"
En isolant les machines dont le flag `Machine failure` est à 1, nous avons créé une catégorie **"No Specific Failure"** pour les cas où aucun des 5 types de pannes n'est coché. 
* **Le constat** : **9 machines** sont déclarées en panne sans qu'aucune cause technique (TWF, HDF, PWF, OSF ou RNF) ne soit identifiée. 
* **L'impact** : Ces 9 instances constituent du "bruit" pour l'algorithme, car elles présentent un état de défaillance sans fournir les signatures physiques correspondantes.

![Figure 5 : Comparaison de la distribution des pannes - Données brutes vs Machines en panne](./images/Failure_Types_RNF_no_specific.png)


#### Stratégie de nettoyage adoptée
Pour garantir la convergence du réseau de neurones vers des motifs physiques cohérents, les décisions suivantes ont été prises :
1.  **Suppression de la feature RNF** : Étant donné son caractère imprévisible et son absence de corrélation avec l'échec machine global, elle est exclue pour éviter d'induire le modèle en erreur.
2.  **Élimination des lignes incohérentes** : Les 18 cas de RNF "non critiques" ainsi que les 9 pannes sans cause spécifiée seront supprimés. 

Ce raffinage permet de recentrer l'apprentissage sur les  pannes déterministes (liées à la température, à l'usure ou à la charge), qui sont les seules que l'IA peut réellement apprendre à anticiper sur un système embarqué.

## 5. Préparation et Ingénierie des Données

### 5.1. Nettoyage et Filtrage Avancé
Pour garantir que le modèle apprenne des relations physiques réelles, nous avons appliqué un filtrage strict sur le dataset original. L'objectif était d'éliminer les contradictions étiquetées lors de l'analyse exploratoire :
* **Suppression des pannes aléatoires (RNF)** : Exclusion des lignes où `RNF = 1` mais `Machine failure = 0`.
* **Élimination des pannes "orphelines"** : Retrait des instances où une panne est signalée sans cause technique identifiée (somme des types de pannes égale à zéro).

### 5.2. Normalisation des Features
Le jeu de données présente des variables aux échelles hétérogènes (ex: la vitesse en tours/min vs le couple en Nm). Pour assurer la stabilité de la descente de gradient et éviter qu'une feature ne domine artificiellement les autres, nous utilisons le StandardScaler. Ce dernier transforme les données pour obtenir une moyenne de 0 et un écart-type de 1.

### 5.3. Stratégie de Rééquilibrage Hybride (SMOTE & Undersampling)
Le déséquilibre extrême des classes (3.5% de pannes) constitue un obstacle majeur pour un réseau de neurones profond. Nous avons implémenté une stratégie de rééchantillonnage hybride sur l'ensemble d'entraînement :
1.  **Sous-échantillonnage (Random Undersampling)** : Réduction de la classe majoritaire "No Failure" à un ratio contrôlé pour éviter l'écrasement des signaux de panne.
2.  **Sur-échantillonnage synthétique (SMOTE)** : Création d'échantillons artificiels pour les classes minoritaires (TWF, HDF, PWF, OSF) par interpolation entre voisins proches.

Cette approche permet de présenter au modèle un set d'entraînement équilibré (**368 pannes pour 122 cas normaux**), forçant l'IA à apprendre les caractéristiques spécifiques de chaque type de défaillance.

---

## 6. Architecture du Réseau de Neurones (DNN)

Nous avons opté pour une architecture multicouche dense, dimensionnée pour capturer les non-linéarités des capteurs tout en restant compatible avec les ressources d'un microcontrôleur.

### 6.1. Structure du Modèle
Le modèle est un réseau séquentiel composé de trois couches :
* **Couche 1** : 256 neurones, activation ReLU, régularisation L2 ($1e-4$).
* **Couche 2** : 128 neurones, activation ReLU, régularisation L2 ($1e-4$).
* **Couche 3** : 64 neurones, activation ReLU, régularisation L2 ($1e-4$).
* **Sortie** : 5 neurones avec activation Softmax pour la classification multi-classe.

### 6.2. Régularisation et Optimisation
Pour prévenir le surapprentissage (overfitting) lié à l'augmentation synthétique des données, nous avons intégré :
* **Batch Normalization** : Stabilise l'apprentissage à chaque couche.
* **Dropout (30%)** : Désactivation aléatoire de neurones pour forcer la robustesse des chemins de décision.
* **Callback ReduceLROnPlateau** : Ajustement dynamique du taux d'apprentissage (Learning Rate) en cas de stagnation de la perte de validation.

---

## 7. Analyse des Performances et Optimisation du Seuil

### 7.1. Optimisation du Seuil de Décision (Threshold Tuning)
Dans un contexte industriel, le coût d'une fausse alarme (arrêt machine injustifié) est souvent plus critique que le coût d'une panne manquée. Pour répondre à cet impératif, nous avons implémenté une recherche de seuil optimal sur les probabilités de sortie du Softmax.

Le modèle ne valide une catégorie de panne que si sa confiance dépasse 0,95. En dessous de ce seuil, la prédiction est forcée à "No Failure". Ce choix stratégique permet de garantir une haute fiabilité des alertes envoyées à l'opérateur, limitant le taux de fausses alarmes à seulement 3,01 % (58 cas sur 1929 échantillons sains).

### 7.2. Évaluation Globale sur le Set de Test Original
L'évaluation finale, réalisée sur les données réelles (non rééquilibrées), montre la capacité du modèle à opérer dans un environnement de production :

| Métrique | Valeur | Interprétation |
| :--- | :---: | :--- |
| **Précision Globale (Accuracy)** | **96%** | Capacité générale à classer correctement un état machine. |
| **Macro Average Recall** | **0,66** | Capacité moyenne à détecter les différents types de pannes. |
| **Macro Average F1-Score** | **0,56** | Équilibre entre précision et rappel sur l'ensemble des classes. |
| **Taux de Fausses Alarmes** | **3,01%** | Proportion de prédictions de pannes erronées sur les machines saines. |

### 7.3. Analyse Détaillée par Diagnostic de Panne
Le rapport de classification nous permet d'identifier les forces et les limites du modèle pour chaque type de défaillance :

* **HDF (Heat Dissipation Failure)** : C'est le succès majeur du modèle avec un rappel de 0,83. Cela signifie que 83 % des pannes de dissipation thermique sont détectées.
* **PWF (Power Failure)** : Avec un score F1 de 0,69, le modèle est robuste pour identifier les pannes de puissance.
* **TWF (Tool Wear Failure)** : Le modèle affiche un rappel correct (0,67) mais une précision faible (0,15). Concrètement, le modèle est "pessimiste" : il détecte bien l'usure d'outil, mais génère des suspicions prématurées.
* **OSF (Overstrain Failure)** : C'est la classe la plus difficile avec un rappel de 0,19. Les pannes de surcharge semblent se confondre avec des pics de couple normaux au seuil de confiance de 0,95. Une analyse ultérieure pourrait nécessiter des features de séries temporelles plus riches pour cette classe.

### 7.4. Conclusion Critique : Un Modèle aux Performances Perfectibles

Bien que le modèle affiche une précision globale flatteuse de 96 %, une analyse rigoureuse des métriques par classe tempère ce résultat. L'utilisation d'un seuil de décision élevé (0,95) permet certes de limiter les fausses alarmes, mais elle masque une incapacité réelle du modèle à diagnostiquer de manière fiable certaines catégories de pannes.

Le rappel de 0,19 pour les pannes de surcharge (OSF) est un point d'échec majeur : le modèle manque plus de 80 % de ces défaillances. De même, la précision très faible sur l'usure d'outil (TWF : 0,15) génère une incertitude telle que l'alerte perd de sa crédibilité opérationnelle. En l'état, le modèle est davantage un détecteur de "bonne santé" qu'un véritable outil de diagnostic prédictif multi-classes. Le déploiement sur STM32 est donc techniquement possible, mais son utilité métier reste limitée par ces lacunes de détection.

---

## 8. Perspectives et Axes d'Amélioration

Pour transformer ce prototype en une solution industrielle robuste, plusieurs leviers d'optimisation doivent être explorés :

### 8.1. Enrichissement du Jeu de Données
Le dataset AI4I 2020 est synthétique et, malgré ses 10 000 entrées, il souffre d'une trop grande rareté des cas de pannes réelles. 
* **Collecte de données réelles** : L'intégration de données issues de véritables capteurs industriels permettrait de capturer des signatures de pannes plus complexes que celles générées par simulation.
* **Augmentation de données avancée** : Au lieu du SMOTE simple, l'utilisation de GANs (Generative Adversarial Networks) pourrait créer des exemples de pannes plus réalistes et diversifiés pour aider le modèle à mieux généraliser les classes minoritaires.

### 8.2. Feature Engineering et Analyse Temporelle
Le modèle actuel traite chaque échantillon de manière isolée (données tabulaires). Or, une panne est souvent le résultat d'une dégradation lente dans le temps.
* **Extraction de caractéristiques temporelles** : Ajouter des colonnes de moyennes mobiles, de pics de vibrations ou de dérivées de température permettrait au modèle de détecter des tendances plutôt que de simples dépassements de seuils.
* **Analyse fréquentielle** : Transformer les signaux bruts via une Transformée de Fourier (FFT) pour extraire des fréquences caractéristiques d'usure mécanique.

### 8.3. Coût de l'Erreur (Cost-Sensitive Learning)
Modifier la fonction de perte (Loss Function) pour pénaliser beaucoup plus lourdement le modèle lorsqu'il rate une panne (Faux Négatif) par rapport à une fausse alerte. Cela forcerait le réseau de neurones à accorder plus d'importance aux classes minoritaires lors de sa phase d'apprentissage.

## 9. Déploiement Embarqué et Analyse des Ressources

Le passage du modèle théorique à l'implémentation physique a été réalisé sur une cible STM32L4R9, un microcontrôleur ultra-basse consommation (ARM Cortex-M4). Cette étape repose sur l'utilisation de l'outil X-CUBE-AI pour convertir notre modèle TensorFlow Lite en code C optimisé.

### 9.1. Analyse des Ressources (Analyse statique)

L'analyse du modèle `failure_prediction_model.tflite` avec une compression medium a généré les métriques d'occupation mémoire suivantes :

| Ressource | Taille (Octets) | Taille (KiB) | Commentaire |
| :--- | :---: | :---: | :--- |
| **FLASH (Poids/Weights)** | 24 532 B | 23,96 KiB | Stockage non-volatile des paramètres. |
| **FLASH (Librairie)** | 10 646 B | 10,40 KiB | Code de l'infrastructure de calcul. |
| **RAM (Activations)** | 1 536 B | 1,50 KiB | Buffer nécessaire pour le calcul des couches. |
| **RAM (Total)** | 4 020 B | 3,93 KiB | Empreinte mémoire vive totale en exécution. |

**Performance de calcul :**
Le modèle présente une complexité de 43 536 MACC (Multiply-Accumulate operations). Sur une architecture Cortex-M4 cadencée à 120 MHz, l'inférence est quasi instantanée (estimée à moins de 0,1 ms par échantillon), ce qui laisse une marge de manœuvre considérable pour d'autres tâches de contrôle-commande.

![Figure 9 : Rapport d'analyse statique X-CUBE-AI](./images/analyse.png)

### 9.2. Validation du Modèle (Cross-Validation)

Avant l'intégration, nous avons effectué une validation croisée sur PC (Desktop) pour comparer les sorties du modèle C généré par rapport au modèle TensorFlow original. 
* **Résultat** : **Accuracy = 100,00 %**
* **Erreur RMSE** : $5,46 \times 10^{-3}$

![Figure 9 : Rapport d'analyse statique X-CUBE-AI](./images/Analyse2.png)

Cette validation confirme que la conversion et la compression n'ont pas dégradé la logique décisionnelle de notre réseau de neurones.

### 9.3. Architecture Logicielle et Intégration (Firmware C)

L'implémentation sur le microcontrôleur repose sur le fichier `app_x-cube-ai.c`, qui agit comme une couche d'abstraction entre le moteur d'inférence généré et le matériel (HAL STM32). Le cycle de fonctionnement est structuré en quatre phases critiques.

#### A. Initialisation et Bootstrap
La fonction `ai_boostrap()` est responsable de la mise en service du réseau. Elle effectue les opérations suivantes :
* **Instanciation** : Création de l'objet IA `rabet_zajac` en mémoire.
* **Mise en correspondance des buffers** : Liaison des pointeurs d'entrée (`data_ins`) et de sortie (`data_outs`) avec les descripteurs de l'IA. 
* **Activation** : Allocation du pool de mémoire (`pool0`) dédié aux activations intermédiaires des couches du réseau.

#### B. Protocole de Synchronisation (Handshake)
Pour garantir l'intégrité de la communication série avec le script Python, la fonction `synchronize_UART()` implémente une boucle d'attente active :
1. Elle scrute l'UART2 jusqu'à la réception du byte de synchronisation `0xAB` (macro `SYNCHRONISATION`).
2. Une fois reçu, elle renvoie immédiatement le byte `0xCD` (macro `ACKNOWLEDGE`).
Ce mécanisme prévient tout décalage d'octets (byte-shift) qui corromprait les données flottantes lors de la transmission.

#### C. Gestion des Flux d'Entrée/Sortie (IO)
L'acquisition et l'envoi des données exploitent directement les fonctions de la couche HAL :
* **Acquisition (`acquire_and_process_data`)** : Utilise `HAL_UART_Receive` pour remplir directement le buffer d'entrée de l'IA. La taille est définie dynamiquement par la macro `AI_RABET_ZAJAC_IN_1_SIZE_BYTES`.
* **Post-traitement (`post_process`)** : Utilise `HAL_UART_Transmit` pour renvoyer le buffer de sortie après inférence. Les probabilités calculées sont ainsi transmises sous forme binaire brute pour minimiser la latence.

#### D. Boucle de Traitement (Processing Loop)
La fonction `MX_X_CUBE_AI_Process()` coordonne l'exécution séquentielle. Elle utilise une structure `do-while` qui garantit que pour chaque cycle de maintenance :
1. La synchronisation est validée.
2. Les données des capteurs sont chargées en RAM.
3. Le moteur d'inférence `ai_run()` est appelé.
4. Les résultats du diagnostic sont transmis au PC.

#### E. Redirection de la Sortie Standard (Printf)
Afin de faciliter le débogage, le stub système `_write` a été redéfini. Il redirige le flux de sortie standard (`stdout`) vers l'UART2. Cela permet d'utiliser la fonction standard `printf()` pour afficher l'état de l'initialisation du modèle directement dans un terminal série.

---

## 10. Résultats et Communication Série

### 10.1. Protocole de Communication UART
Pour l'échange de données en temps réel, nous avons configuré l'interface UART2 avec les paramètres suivants :
* **Baud Rate** : 115200 bits/s
* **Configuration** : 8N1 (8 bits de données, pas de parité, 1 bit de stop)

### 10.2. Pilotage et Monitoring via `Communication_STM32_NN.py`

Le script Python `Communication_STM32_NN.py` agit comme le chef d'orchestre de l'évaluation. Il assure l'interface entre les données de test stockées sur le PC et le moteur d'inférence s'exécutant sur la cible STM32. 

#### A. Protocole de Synchronisation (Handshake)
Avant tout échange de données critiques, une procédure de synchronisation robuste est établie via la fonction `synchronise_UART()`. 
* Le script envoie en boucle le flag `0xAB`.
* Il attend la réponse `0xCD` de la part de la carte.
Cette étape garantit que le firmware est prêt et que les buffers de communication sont alignés, évitant ainsi tout décalage dans la lecture des flux binaires.

#### B. Sérialisation et Envoi des Données
Les entrées du modèle (températures, vitesse, usure, etc.) sont extraites du fichier `X_test.npy`. La fonction `send_inputs_to_STM32()` convertit ces échantillons en **flottants 32 bits (float32)** avant de les transformer en flux de bytes bruts via la méthode `.tobytes()`. Ce format est directement compatible avec la représentation mémoire des `float` en langage C sur l'architecture ARM Cortex-M4.

#### C. Récupération et Normalisation des Sorties
Une particularité de l'implémentation réside dans la lecture des résultats. La fonction `read_output_from_STM32()` réceptionne les octets envoyés par la carte. Pour optimiser la bande passante, les probabilités sont transmises sous forme d'entiers sur 8 bits. Le script Python effectue alors une normalisation inverse :
$$Valeur_{float} = \frac{Octet_{reçu}}{255}$$
Cette opération permet de reconstituer une distribution de probabilités comprise entre 0 et 1 pour chaque classe.

#### D. Boucle d'Évaluation de la Précision
La fonction `evaluate_model_on_STM32()` automatise le test sur l'ensemble du set `X_test`. Pour chaque itération :
1. Elle transmet l'échantillon $i$.
2. Elle récupère le vecteur de prédiction de la carte.
3. Elle compare l'indice de la probabilité maximale (**Argmax**) du retour STM32 avec celui du label de référence (**Y_test**).
4. Elle calcule une précision cumulée en temps réel.

![Figure 11 : Capture d'écran du terminal lors de l'exécution du script Communication_STM32_NN.py](./images/terminal_result.png)
> *Note : On peut observer dans le terminal la comparaison entre "Expected output" (Colab) et "Received output" (STM32), confirmant la parfaite adéquation du déploiement.*

---

## 11. Conclusion Générale

Ce projet nous a permis de traverser l'intégralité de la chaîne de conception d'une IA embarquée (Edge AI). 

De l'analyse d'un dataset industriel déséquilibré à l'optimisation d'un réseau de neurones profond sur une cible à ressources limitées, nous avons pu identifier les défis critiques de la maintenance prédictive. Si notre modèle offre d'excellents résultats sur les pannes franches (thermiques ou de puissance), il a révélé les limites des approches purement tabulaires pour les pannes soudaines ou mal étiquetées (OSF/RNF).

L'implémentation réussie sur STM32L4R9 démontre néanmoins qu'avec une empreinte mémoire extrêmement réduite (< 4 KiB de RAM), il est tout à fait possible d'intégrer une intelligence de diagnostic robuste au cœur des machines industrielles, ouvrant la voie à une maintenance plus verte, plus économique et plus réactive.