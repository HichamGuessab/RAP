# -*- coding: utf-8 -*-
"""
         hello.py
         Auteur : Remy Kessler
		Rôle : Calcule les coefficients d'énergie dans différentes bandes de fréquences (échelle de Mel)

		Args:
        Signal: Le vecteur total du signal échantillonné à Fs kHz
        n1: Début de la trame
        lgsig: Longueur de la trame
        Fs: Fréquence d'échantillonnage
        nfilt: Nombre de filtres
        titre: Titre du graphique

    Returns:
        en: Vecteur contenant les énergies dans chaque bande de fréquence
    """
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pprint as pprint
import math



def canaux(Signal, n1, lgsig, Fs, nfilt, titre):

    n2=n1+lgsig-1
    signal = Signal[n1:n2]
    presignal = signal

    
    # Supposons que presignal soit votre signal original
    lgsig = len(presignal)  # Longueur de la portion du signal à fenêtrer

    # Création d'une fenêtre de Hamming de taille lgsig
    fen = np.hamming(lgsig)

    # Application de la fenêtre au signal
    fensignal = presignal[:lgsig] * fen	


    # calcul de la transformée de Fourrier du vecteur de parole
    spect = np.fft.fft(fensignal)
    nspect=abs(spect);

    #frequence = 16000/2*linspace(0,1,lgsig/2);
    frequence = np.linspace(0, 16000/2, int(lgsig/2))


    #calcul des nfilt filtres triangulaires a partir des frequences de coupure (echelle Mel)
    fcoup=[0,100,200,300,400,500,600,700,800,900,1000,1150,1300,1500,1700,2000,2350,2700,3100,3550,4000,4500,5050,5600,6200,6850,7500];

    lgsig1=lgsig/2
    lgsig1 = math.floor(lgsig1)    
    x2 = np.zeros(lgsig1)
    x1 = np.zeros(lgsig1)
    nm = np.zeros(lgsig1)
    
    for i in range(0, lgsig1):
      #nm[i] = 0
      print(i)
      x1[i] = 0
      f = Fs * 500 * i / lgsig1
    
      for n in range(1, nfilt + 2):
        if fcoup[n] <= f < fcoup[n + 1]:
            nm[i] = n
            x1[i] = (f - fcoup[n]) / (fcoup[n + 1] - fcoup[n])
            x2[i] = 1 - x1[i]
    
    x2sig = x2 * nspect[:lgsig1]
    x1sig = x1 * nspect[:lgsig1]
    
    echantillon = np.arange(0, lgsig)
    temps = echantillon/16000;

    # Afficher le signal
    plt.figure(figsize=(12, 6))
    plt.subplot(3,1,1)
    plt.plot(temps,signal)
   

    plt.xlabel('secondes')  # Ajout de l'étiquette
    plt.ylabel('domaine temporel')
    plt.title(titre)


    
    plt.subplot(3, 1, 2)
    plt.plot(frequence, nspect[:lgsig1])
    plt.xlim(0, 6000)
    plt.xlabel('Fréquences (Hz)')
    plt.ylabel('FFT')

    
    plt.subplot(3, 1, 3)
    
    en = np.zeros(nfilt)
    
    
    for i in range(1,nfilt):
      i2=int(nm[i])
      i1=int(i2-1)
      
      
      
      if(i1 > 0) :
         en[i1]=en[i1]+x2sig[i]
      if ((i2 > 0) & (i2 <= nfilt)):
         en[i2]=en[i2]+x2sig[i]
    
    plt.step(fcoup[:nfilt], en, where='post')
    plt.xlabel('Fréquences (Hz)')
    plt.ylabel('Échelle Mel')
    

    plt.grid(True)
    plt.show()
    

    return signal





