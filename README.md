# AccelSpy, riconoscimento tramite accelerometro reso semplice
![Logo](https://i.ibb.co/PmxnCs5/logo.png) 

AccelSpy è un applicazione mobile in grado di riconoscere, tramite l'accelerometro e un supporto lato server, il testo parlato dato un file csv.
Tramite AccelSpy possiamo controllare l'accelerometro e possiamo decidere, mentre riproduciamo un audio, di usarlo per mappare l'audio riprodotto e di salvarlo in memoria. Una volta salvato il file, sarà possibile inviarlo ad un server che lo elaborerà e ci restituirà nella console il testo riconosciuto. Il tutto in un interfaccia intuitiva ed user-friendly.
Il cuore pulsante di AccelSpy è l'algoritmo evolutivo di segmentazione, che prende in input un file .csv e riconosce le varie speech unit presenti.
Il codice del nostro lavoro è in chiaro e tranquillamente consultabile: sono presenti sia il codice dell'algoritmo evolutivo (in python) che il codice dell'applicazione (in Java).


## Screenshots

![App Screenshot](https://i.ibb.co/NWSXQHy/schermata-principale-dell-app.png)


## Features in app

- Riprodurre audio (pre-registrati e non)
- Registrare audio
- Riconoscimento tramite invio al server
- Possibilità di eliminare audio
- Possibilità di selezionare il .csv da inviare



## Manuale utente
E' possibile consultare qui il [manuale utente](https://smallpdf.com/it/result#r=a599e56ab7c8aed98c94c3cf76e053cb&t=share-document) riguardante l'applicazione mobile.


## Ottimizzazioni

L'app è ottimizzata per il risparmio batteria ed è in grado di capire quando l'accelerometro non è in uso, così da poterlo spegnere.



## Installazione

Per far partire l'app è necessiaria l'ultima versione di [Android Studio](https://developer.android.com/studio), reperibile qui.
Per il test dell'algoritmo evolutivo di segmentazione, invece, bisogna installare [Python](https://www.python.org/downloads/).

    
## Demo

**Da inserire una gif o un video dimostrativo**


## Tech Stack

**App:** Java

**Server:** Python


## Badges
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)


## Autori

- [@Givaa](https://github.com/Givaa)
- [@Michele-iodice](https://github.com/Michele-iodice)
