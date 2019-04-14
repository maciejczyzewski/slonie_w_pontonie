# :muscle: __Słonie w pontonie__ :muscle:

<img src="keypoints_pose_18.png" alt="lewy w pelnej klasie" width="400" align="right"/>

### Wprowadzenie

Aplikacja dla osób trenujących. Pozwala śledzić progres treningu przy jednoczesnej kontroli stanu zdrowia. Na podstawie serii zdjęć, za pomocą sieci neuronowych analizuje objętość i kształt mięśni oraz posturę, zwracając uwagę na ostrzeżenia chorobowe (takie jak otyłość, skrzywienia kręgosłupa, rak skóry).

### Skutecznosc ML

Ponizej prezentujemy rezultaty sieci do analizy **stopnia rozbudowy/masy
miesniowej/tluszczu**.

<img src="epoch_acc.png" alt="epoch_acc" width="460" align="left"/>
<img src="epoch_loss.png" alt="epoch_loss" width="460" align="left"/>

| Net (spec.)   |      Acc.     |
|---------------|:-------------:|
| lewy_biceps   | 0.81          |
| prawe_ramie   | 0.78          |
| lewe_ramie    | 0.71          |
| prawe_udo     | 0.78          |
| lewe_udo      | 0.74          |
| prawa_lydka   | 0.63          |
| lewa_lydka    | 0.62          |
| lewy_abs      | 0.52          |
| prawy_abs     | 0.73          |
| prawa_klatka  | 0.80          |
| lewa_klatka   | 0.76          |
| prawy_biceps  | 0.72          |

### Preprocessing (dane 3d/auto. maska/postura)

![posture](posture.png)

### Funkcje

- analiza objętości i kształtu mięśni
- klasyfikacja sylwetki na skali
- wizualizacja progresu i statystyki
- możliwość wyznaczenia przez klienta celu i dopasowanie treningu
- analiza postury
- ostrzeżenia chorobowe

### Założenia aplikacji

<img src="docs/1.jpg" alt="miesniak" height="400"/>
<i>Analiza stanu ciała ze zdjęcia</i>
<img src="docs/2.jpg" alt="skala" height="400"/>
<i>Umieszczenie sylwetki na skali</i>



```
$ pipreqs . --ignore hmr                        # generacja listy pakietow
$ pip3 install -r requirements.txt --no-index   # instalacja pakietow
$ autopep8 --in-place --aggressive --aggressive <filename>  # czystosc
$ python3 debug.py  # szybki wizualny debug na "lewym"
```
