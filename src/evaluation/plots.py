# src/evaluation/plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_distribution(df):
    plt.figure()
    sns.histplot(df['G3'], bins=20)
    plt.title("Distribution of Final Grade")
    plt.savefig("outputs/figures/grade_distribution.png")
    plt.close()


def plot_absence_vs_grade(df):
    plt.figure()
    sns.scatterplot(x='absences', y='G3', data=df)
    plt.title("Absences vs Final Grade")
    plt.savefig("outputs/figures/absence_vs_grade.png")
    plt.close()


def plot_studytime(df):
    plt.figure()
    sns.countplot(x='studytime', data=df)
    plt.title("Study Time Distribution")
    plt.savefig("outputs/figures/studytime.png")
    plt.close()


def plot_failures(df):
    plt.figure()
    sns.countplot(x='failures', data=df)
    plt.title("Failures Distribution")
    plt.savefig("outputs/figures/failures.png")
    plt.close()


def plot_heatmap(df):
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig("outputs/figures/heatmap.png")
    plt.close()


def plot_confusion_matrix(cm):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.savefig("outputs/figures/confusion_matrix.png")
    plt.close()