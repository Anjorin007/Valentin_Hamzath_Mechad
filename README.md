# Projet Machine Learning - Analyse et Prédiction

## 👥 Contributions

Ce projet a été réalisé **en collaboration complète** entre les trois membres de l'équipe :

- **NOUGBOLOYIN Valentin** : Coordination Git, Architecture générale, Classification (Logistic Regression & MLP)
- **DRAMANE Hamzath** : Régression (Linear & Neural Network), Feature Engineering, Visualisations
- **HOUSSA Mechard** : Preprocessing, Nettoyage des données, Hyperparameter tuning

**Note** : Tous les fichiers ont été développés ensemble en sessions de travail collaboratif. Le dépôt Git a été initialisé par Hamzath, mais chaque membre a contribué activement au code et aux analyses.

---

## Vue d'ensemble

Ce projet explore deux approches différentes du Machine Learning appliquées à des données réelles : la **classification** et la **régression**. On a implémenté plusieurs modèles pour comparer leurs performances et comprendre leurs forces/faiblesses.

---

## 📊 Partie 1 : Classification (Untitled2.ipynb)

### Contexte
On travaille avec le dataset `Social_Network_Ads.csv` qui contient des infos sur des utilisateurs (âge, salaire, genre) et si oui ou non ils ont acheté un produit. L'objectif est de prédire si un nouveau client va acheter ou pas.

### Modèles testés

**1. Régression Logistique**
- Modèle de base pour la classification binaire
- Très conservateur dans ses prédictions
- Résultats : Accuracy 78.75%, Recall 39.29%, Precision 100%
- Problème : détecte peu d'acheteurs potentiels (trop prudent)

**2. MLP Classifier (sklearn)**
- Réseau de neurones simple avec 2 couches cachées (100 et 50 neurones)
- Meilleur équilibre global
- Résultats : Accuracy 80%, Recall 64.29%, Precision 75%
- Détecte beaucoup plus d'acheteurs tout en restant fiable

**3. Deep Learning (TensorFlow/Keras)**
- Architecture personnalisée : 128 → 64 → 32 → 1
- Entraînement sur 50 epochs avec dropout pour éviter l'overfitting
- Résultats : Accuracy 78.75%, Recall 64.29%, Precision 72%
- Bon compromis, visualisation de l'apprentissage incluse

### Preprocessing
- Encodage OneHot pour les variables catégorielles (Genre)
- Standardisation avec StandardScaler
- Split train/test 80/20

### Conclusion classification
Le MLP sklearn s'est révélé le plus performant pour ce cas d'usage, avec le meilleur taux de détection des acheteurs potentiels.

---

## 📈 Partie 2 : Régression (rn.ipynb)

### Contexte
Dataset `retail_data.csv` avec des données transactionnelles complètes. L'objectif est de prédire une valeur continue : la **valeur moyenne des transactions** d'un client.

### Approche méthodologique

**Features sélectionnées :**
- Numériques : age, membership_years, online_purchases, purchase_frequency, etc.
- Catégorielles : gender, income_bracket, loyalty_program, education_level, etc.

**Preprocessing professionnel :**
- ColumnTransformer pour gérer différents types de features
- Pipeline numériques : imputation médiane + standardisation
- Pipeline catégorielles : imputation mode + one-hot encoding
- Nettoyage des valeurs Yes/No converties en binaire

### Modèles comparés

**1. Régression Linéaire**
- Baseline simple et interprétable
- Entraînement rapide
- Métriques : MAE, MSE, R²

**2. Réseau de Neurones (Keras)**
- Architecture : 64 → 32 → 1 (sortie linéaire)
- Optimiseur Adam avec learning rate adaptatif
- Callbacks intelligents :
  - EarlyStopping (arrêt si pas d'amélioration)
  - ReduceLROnPlateau (ajustement automatique du learning rate)
- Entraînement max 200 epochs avec validation split

### Visualisations
- Courbes d'apprentissage (loss vs epochs)
- Scatter plots Prédit vs Réel pour chaque modèle
- Tableau comparatif des métriques

---

## 🛠️ Technologies utilisées

- **Python 3.10+**
- **Pandas** : manipulation de données
- **Scikit-learn** : preprocessing, modèles classiques, métriques
- **TensorFlow/Keras** : deep learning
- **Matplotlib** : visualisations
- **NumPy** : calculs numériques

---

## 📁 Structure du projet

```
.
├── README.md
├── Untitled2.ipynb          # Classification - Social Network Ads
├── rn.ipynb                 # Régression - Retail Data
├── Social_Network_Ads.csv   # Dataset classification
└── retail_data.csv          # Dataset régression (si présent)
```

---

## 🚀 Comment exécuter

1. Installer les dépendances :
```bash
pip install pandas scikit-learn tensorflow matplotlib numpy
```

2. Lancer Jupyter :
```bash
jupyter notebook
```

3. Ouvrir le notebook souhaité et exécuter les cellules dans l'ordre

---

## 📝 Apprentissages clés

1. **Preprocessing** : crucial pour la performance des modèles, surtout avec des features mixtes
2. **Overfitting** : les techniques de régularisation (Dropout, EarlyStopping) sont essentielles
3. **Métriques** : l'accuracy seule ne suffit pas, il faut regarder precision/recall selon le contexte
4. **Deep Learning** : pas toujours meilleur que les modèles classiques, tout dépend du dataset
5. **Pipeline scikit-learn** : permet un code propre et reproductible

---

## 🎯 Perspectives d'amélioration

- Tester d'autres architectures de réseaux (CNN, LSTM si données temporelles)
- Grid search pour optimiser les hyperparamètres
- Feature engineering plus poussé
- Cross-validation pour des résultats plus robustes
- Ensembling de modèles

---

**Date de réalisation** : Janvier 2026


