import numpy as np
import math
import cv2
import vector_prof
from neuronska import iseci_broj,napravi_model,ispraviSlova


def nadji_liniju(slika):
    hsv = cv2.cvtColor(slika, cv2.COLOR_BGR2HSV)
    #plava
    donja_granica = np.array([100, 50, 50])
    gornja_granica = np.array([130, 255, 255])
    maska = cv2.inRange(hsv, donja_granica, gornja_granica)
    #cv2.imshow("maska", maska)
    plava = cv2.bitwise_and(slika, slika, mask=maska)
    ivice_plave = cv2.Canny(plava, 50, 200, None, 3)
    #vraca koordinate linija
    plave_linije = cv2.HoughLinesP(ivice_plave, 1, np.pi / 180, 50, None, 50, 10)
    tacke_linije = [(0, 0), (0, 0)]
    duzinaMax = 0
    if plave_linije is not None:
        for i,li in enumerate(plave_linije):
            linija = plave_linije[i][0]
            duzinaLinije = math.sqrt((linija[2] - linija[0]) ** 2
                                 + (linija[3] - linija[1]) ** 2)
            if duzinaMax < duzinaLinije:
                tacke_linije[0] = (linija[0], linija[1])
                tacke_linije[1] = (linija[2], linija[3])
                duzinaMax = duzinaLinije

    return tacke_linije
def nadjiElement(sviEl, element):
    pronadjeni = []
    for i,el in enumerate(sviEl):
        (eX, eY) = element['centar']
        (x, y) = el['centar']
        distanca = math.sqrt(math.pow((x - eX), 2) + math.pow((y - eY), 2))
        if distanca < 20:
            pronadjeni.append(i)
    return pronadjeni
def prepoznajKonture(slika):
    donja_bela = np.array([230, 230, 230], dtype="uint8")
    gornja_bela = np.array([255, 255, 255], dtype="uint8")
    mask = cv2.inRange(slika, donja_bela, gornja_bela)
    slika = cv2.bitwise_and(slika, slika, mask=mask)
    siva = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)
    blurovana = cv2.GaussianBlur(siva, (5, 5), 0)
    #cv2.imshow("blur", blurovana)
    im, konture, _ = cv2.findContours(blurovana.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    konture_brojeva = []
    for kontura in konture:
        (x, y, w, h) = cv2.boundingRect(kontura)
        povrsina = cv2.contourArea(kontura)
        if h > 12 and povrsina > 30 and povrsina < 1000:
            koordinate = (x, y, w, h)
            konture_brojeva.append(koordinate)
    return konture_brojeva
def prepoznajBroj(slika, kontura, klasifikator):
    (x, y, w, h) = kontura
    centarx = int(x + w / 2)
    centary = int(y + h / 2)
    siva = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)
    broj = siva[centary-12:centary+12, centarx-12:centarx+12]
    #cv2.imshow("broj", broj)
    (tr, broj) = cv2.threshold(broj, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    broj = ispraviSlova(broj)
    broj = iseci_broj(broj)
    #cv2.imshow("broj finalno", broj)
    br = klasifikator.predict_classes(broj.reshape(1, 28, 28, 1))
    return int(br)

videoIme = 'video-9'
video = cv2.VideoCapture(videoIme + '.avi')
ucitaoFrame, frame = video.read()
klasifikator = napravi_model((28,28,1),10)
klasifikator.load_weights(''
                          'weights.h5')

kernel = np.ones((2, 2), np.uint8)
tacke_linije = nadji_liniju(cv2.morphologyEx(frame,cv2.MORPH_OPEN,kernel=kernel))

brFrejma = 0
sum = 0
sviBrojevi = []

while ucitaoFrame:
    ucitaoFrame, frame = video.read()
    if not ucitaoFrame:
        break
    konture = prepoznajKonture(frame)
    for i, kontura in enumerate(konture):
        (x, y, w, h) = kontura
        centarx = int(x + w / 2)
        centary = int(y + h / 2)
        element = {'centar': (centarx, centary), 'brojFrejma': brFrejma, 'istorija': []}
        pronadjeni = nadjiElement(sviBrojevi, element)
        if len(pronadjeni) == 0:
            element['vrednost'] = prepoznajBroj(frame, kontura, klasifikator)
            element['presaoLiniju'] = False
            sviBrojevi.append(element)
        elif len(pronadjeni) == 1:
            # update elementa i dodavanje istorije
            index = pronadjeni[0]
            ist = {'brojFrejma':brFrejma,'centar': element['centar']}
            sviBrojevi[index]['istorija'].append(ist)
            sviBrojevi[index]['brojFrejma'] = brFrejma
            sviBrojevi[index]['centar'] = element['centar']
    #presao liniju?
    for element in sviBrojevi:
        if (brFrejma - element['brojFrejma']) > 3:
            continue
        if not element['presaoLiniju']:
            distanca, _, r = vector_prof.pnt2line(element['centar'], tacke_linije[0], tacke_linije[1])
            # cv2.line(frame, (tacke_linije[0][0], tacke_linije[0][1]),
            #          (tacke_linije[1][0], tacke_linije[1][1]), (0, 0, 255), 1, cv2.LINE_AA)
            if distanca < 10.0 and r == 1:
                sum += int(element['vrednost'])
                element['presaoLiniju'] = True
        cv2.circle(frame, element['centar'], 18, [220, 66, 244], 1)
        cv2.putText(frame, str(element['vrednost']), (element['centar'][0] + 12, element['centar'][1] + 12),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (220, 66, 244), 3)
        cv2.putText(frame, "Suma: " + str(sum), (15, 20), cv2.FONT_HERSHEY_COMPLEX,
                    0.5,(220, 66, 244), 1)
        cv2.putText(frame, "Broj trenutnog frejma: " + str(brFrejma), (15, 40), cv2.FONT_HERSHEY_COMPLEX,
                    0.4, (220, 66, 244), 1)

        for istorija in element['istorija']:
            if (brFrejma - istorija['brojFrejma'] < 80):
                cv2.circle(frame, istorija['centar'], 1, (200, 200, 200), 1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 13:
        break
    brFrejma += 1

cv2.destroyAllWindows()
video.release()

f = open('out.txt', 'a')
f.write('\n' + videoIme + '.avi' + '\t' + str(sum))
f.close()