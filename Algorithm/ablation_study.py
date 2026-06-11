# -*- coding: utf-8 -*-
"""
Ablation Study — contribuição individual de cada módulo forense
===============================================================
Treina o Random Forest com subconjuntos das features para medir
quanto cada módulo contribui isoladamente vs. a combinação completa.

Configurações avaliadas:
  VP isolado    : score_vp + n_linhas
  ELA isolado   : score_ela + ela_mean + ela_std
  Ruído isolado : score_ruido + cv_ruido
  Escala isolado: score_escala + n_anomalias + n_anom_graves
  4 scores      : os 4 scores normalizados (baseline)
  10 features   : vetor completo (melhor configuração)

Uso:
  python Algorithm/ablation_study.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from detector_ml import extrair_matriz, _listar_imagens, FEATURES
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Índices das features por módulo (baseado em FEATURES):
# ['score_vp','score_ela','score_ruido','score_escala',
#  'ela_mean','ela_std','cv_ruido','n_linhas','n_anomalias','n_anom_graves']
CONFIGS = {
    'VP isolado'     : [0, 7],        # score_vp, n_linhas
    'ELA isolado'    : [1, 4, 5],     # score_ela, ela_mean, ela_std
    'Ruido isolado'  : [2, 6],        # score_ruido, cv_ruido
    'Escala isolado' : [3, 8, 9],     # score_escala, n_anomalias, n_anom_graves
    '4 scores'       : [0, 1, 2, 3],  # 4 scores normalizados
    '10 features'    : list(range(10)),
}


def rodar_ablation(X, y, n_estimators=200, seed=42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y)

    resultados = {}
    for nome, idx in CONFIGS.items():
        Xtr = X_tr[:, idx]
        Xte = X_te[:, idx]
        Xfull = X[:, idx]

        clf = RandomForestClassifier(
            n_estimators=n_estimators, class_weight='balanced',
            random_state=seed)
        clf.fit(Xtr, y_tr)
        yp = clf.predict(Xte)
        yprob = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)]

        cv_acc = cross_val_score(clf, Xfull, y, cv=5, scoring='accuracy')
        cv_f1  = cross_val_score(clf, Xfull, y, cv=5, scoring='f1')

        resultados[nome] = {
            'acuracia' : accuracy_score(y_te, yp),
            'precisao' : precision_score(y_te, yp, zero_division=0),
            'recall'   : recall_score(y_te, yp, zero_division=0),
            'f1'       : f1_score(y_te, yp, zero_division=0),
            'auc'      : roc_auc_score(y_te, yprob),
            'cv_acc_m' : cv_acc.mean(),
            'cv_acc_s' : cv_acc.std(),
            'cv_f1_m'  : cv_f1.mean(),
            'cv_f1_s'  : cv_f1.std(),
            'n_feat'   : len(idx),
        }
    return resultados


def imprimir(resultados):
    header = f"{'Configuração':<20} {'Prec':>6} {'Rev':>6} {'F1':>6} {'Acc':>6} {'AUC':>6}  CV-acc(±)      CV-F1(±)"
    print(header)
    print('-' * len(header))
    for nome, r in resultados.items():
        print(f"{nome:<20} "
              f"{r['precisao']*100:5.1f}% "
              f"{r['recall']*100:5.1f}% "
              f"{r['f1']*100:5.1f}% "
              f"{r['acuracia']*100:5.1f}% "
              f"{r['auc']:.3f}  "
              f"{r['cv_acc_m']*100:.1f}%±{r['cv_acc_s']*100:.1f}  "
              f"{r['cv_f1_m']*100:.1f}%±{r['cv_f1_s']*100:.1f}")


if __name__ == '__main__':
    au_dir = os.path.join(ROOT, 'Dataset', 'Au', 'sample')
    sp_dir = os.path.join(ROOT, 'Dataset', 'Tp', 'sample')
    au_paths = _listar_imagens(au_dir)
    sp_paths = _listar_imagens(sp_dir)
    print(f"Carregando features de {len(au_paths)} Au + {len(sp_paths)} Sp...")
    X, y = extrair_matriz(au_paths, sp_paths, workers=1)
    print(f"Features extraidas: {X.shape}\n")

    resultados = rodar_ablation(X, y)
    imprimir(resultados)
