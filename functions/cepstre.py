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


def cepstre(Signal, n1, lgsig, Fs,  titre):
    #Signal est le vecteur total de signal echantillonne a Fs kHz
    #n1 et n2 sont les echantillons de debut et fin de trame, de longueur lgsig
    Signal = Signal * 1e17
    n2=n1+lgsig
    signal = Signal[n1:n2]

    
    #fenetrage de Hamming sur le morceau de signal extrait
    presignal = signal
    fen=np.hamming(lgsig)
    fensignal = presignal[:lgsig] * fen	
    
    
    #abscisse pour la représentation des spectres
    N=512
    absc = np.arange(1, N+1) / N * Fs / 2
    
    
    # log de la partie réelle de la transformée de Fourier du signal
    # calcul de la transformée de Fourrier du vecteur de parole
    spect = np.fft.fft(fensignal,1024)
    #spect=fft(fensignal,1024)
    nspect=np.log(np.abs(spect))
    
    # lgsig1=lgsig/2;
    lgsig1=257
    
    echantillon = np.arange(0, lgsig)
    temps = echantillon/16000
    
    
    # Afficher le signal
    plt.figure(figsize=(12, 6))
    plt.subplot(4,1,1)
    plt.plot(temps,signal)
    
    plt.xlabel('secondes')  # Ajout de l'étiquette
    plt.ylabel('domaine temporel')
    plt.title(titre)

    #log du module du spectre
    plt.subplot(4,1,2)
    
    absc = np.arange(1, N+1) / N * 8000
    plt.plot(absc, nspect[0:512])
    plt.xlabel('Frequence (Hz))')  # Ajout de l'étiquette
    plt.ylabel('FTTl')
    
    #cepstre réel d'un signal réel
    plt.subplot(4,1,3)
    cep1 = np.fft.fft(nspect,1024)
    cep  = np.fft.ifft(nspect,1024)
    
    
    rcep = np.real(cep)
    plt.plot(temps[2:lgsig], rcep[2:512]);
    plt.xlabel('secondes');
    plt.ylabel('domaine cepstral');


    #fft des premiers coefficients du cepstre (conduit) 
    # les extrema correspodent aux formants.
    plt.subplot(4,1,4)
    ceplisse = np.fft.fft(rcep[1:32],1024)
    plt.plot(absc, np.abs(ceplisse[0:512]));
    
    #absc = [1:512]/512 * 4000;
    #plot(absc,(abs(ceplisse(1:512))));
    plt.xlabel('Frequence (Hz))')  # Ajout de l'étiquette
    plt.ylabel('FFT')
    plt.title('FFT des premiers echantillons du cepstre')


    
    plt.grid(True)
    plt.show()

