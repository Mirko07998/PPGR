import numpy as np
import math
from termcolor import colored
import json
import sys

#Implementacija naivnog algoritma
def naive_algorith(a, b, c, d, ap, bp, cp, dp):
    A = [ a[0], a[1], a[2] ]
    B = [ b[0], b[1], b[2] ]
    C = [ c[0], c[1], c[2] ]
    #Pravimo matricu koeficijanata uz nepoznate za resavanje sistema
    m_list = [[ a[0], b[0], c[0] ], [ a[1], b[1], c[1] ], [ a[2], b[2], c[2] ]]
    As = np.array(m_list)
    
    Bs = np.array([d[0], d[1], d[2]])
    #Resavamo sistem da bi dobili l1 l2 i l3
    X = np.linalg.solve(As, Bs)

    #izvlacimo vrednosti iz dobijene matrice
    l1 = X[0]
    l2 = X[1]
    l3 = X[2]

    #Mnozimo odgovarajuce vektore sa dobijenim vrednostima   
    A = list(map(lambda x: x*l1, A))
    B = list(map(lambda x: x*l2, B))
    C = list(map(lambda x: x*l3, C))

    #dobijena prva matrica
    P1 = np.array([A, B, C]).transpose()

    #Ponavljamo isti postupak sa dobijanje druge matrice
    Ap = [ ap[0], ap[1], ap[2] ]
    Bp = [ bp[0], bp[1], bp[2] ]
    Cp = [ cp[0], cp[1], cp[2] ]

    m_listp = [[ ap[0], bp[0], cp[0] ], [ ap[1], bp[1], cp[1] ], [ ap[2], bp[2], cp[2] ]]
    Asp = np.array(m_listp)

    Bsp = np.array([dp[0], dp[1], dp[2]])
    Xp = np.linalg.solve(Asp, Bsp)
    
    l1p = Xp[0]
    l2p = Xp[1]
    l3p = Xp[2]
    
    Ap = list(map(lambda x: x*l1p, Ap))
    Bp = list(map(lambda x: x*l2p, Bp))
    Cp = list(map(lambda x: x*l3p, Cp))

    P2 = np.array([Ap, Bp, Cp]).transpose()
    
    P1inv = np.linalg.inv(P1)

    #dobijamo resenje matrice
    P = np.matmul(P2, P1inv)

    #vracamo resenje
    return P

def DLTpure(tacke, slike):
    #Pravimo potrebne vektore za matricu korespondencije
    mat = []
    for i in range(len(slike)):
        A1, A2 = makeCorespondence(tacke[i], slike[i])
        mat.append(A1)
        mat.append(A2)

    #Pravimo matricu
    Mat = np.array(mat)

    #Radimo svd dekompoziciju
    _, _, vh  = np.linalg.svd(Mat)

    #Matica cije su kolone resenje, dobijena primenom svd-a
    P = np.transpose(vh)

    pom =P[0:9,8]

    #Pravimo matricu projektivnog preslikavanja
    res = np.array([[pom[0], pom[1], pom[2]], [pom[3], pom[4], pom[5]], [ pom[6], pom[7], pom[8] ] ] )
    
    #Stampamo resenje DLT algoritma
    #print(res)

    return res

#Implementacija DLT algoritma
def DLT(a, b, c, d, e, f, ap, bp, cp, dp, ep, fp):

    #Pravimo potrebne vektore za matricu korespondencije
    A1, A2 = makeCorespondence(a, ap)
    B1, B2 = makeCorespondence(b, bp)
    C1, C2 = makeCorespondence(c, cp)
    D1, D2 = makeCorespondence(d, dp)
    E1, E2 = makeCorespondence(e, ep)
    F1, F2 = makeCorespondence(f, fp)

    #Pravimo matricu
    Mat = np.array([A1, A2, B1, B2, C1, C2, D1, D2, E1, E2, F1, F2])

    #Radimo svd dekompoziciju
    _, _, vh  = np.linalg.svd(Mat)

    #Matica cije su kolone resenje, dobijena primenom svd-a
    P = np.transpose(vh)

    pom =P[0:9,8]

    #Pravimo matricu projektivnog preslikavanja
    res = np.array([[pom[0], pom[1], pom[2]], [pom[3], pom[4], pom[5]], [ pom[6], pom[7], pom[8] ] ] )
    
    #Stampamo resenje DLT algoritma
    #print(res)

    return res

#Funkcija koja pravi koponente za matricu korespondencije
def makeCorespondence(a, ap):
    v1 = [0, 0, 0, (-1)*ap[2]*a[0], (-1)*ap[2]*a[1], (-1)*ap[2]*a[2], ap[1]*a[0], ap[1]*a[1], ap[1]*a[2]]
    v2 = [ap[2]*a[0], ap[2]*a[1], ap[2]*a[2], 0, 0, 0, (-1)*ap[0]*a[0], (-1)*ap[0]*a[1], (-1)*ap[0]*a[2]]

    return v1, v2

#Funkcija za normalizaciju tacaka
def normalize(t):
    #Vrsimo tranformaciju da dobijemo afine koordinate
    afineTacke = list(map(lambda x:(x[0]/x[2], x[1]/x[2]), t))

    #Racunjanje tezista
    n = len(t)
    xKord = list(map(lambda m: m[0], afineTacke))
    yKord = list(map(lambda m: m[1], afineTacke))

    sumaX = sum(xKord)
    sumaY = sum(yKord)

    #dobijanje x i y koordinata tezista
    xt = float(sumaX)/n
    yt = float(sumaY)/n

    xK = list(map(lambda m: m - xt, xKord))
    yK = list(map(lambda m: m - yt, yKord))

    #Matrica translacije
    T = np.array([[1, 0, (-1)*xt], [0, 1, (-1)*yt], [0, 0, 1]])

    #Racunamo udaljenosti tacaka od koordiantnog pocetka
    udaljenosti = list(map(lambda q, k: math.sqrt( (q)**2 + (k)**2 ), xK, yK))
    
    r = sum(udaljenosti)/n

    s = math.sqrt(2)/r

    #Dobijena matrica skaliranja
    S = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])

    #Racunamo povratnu matricu za normalizaciju
    value = np.matmul(S, T)
    
    #Primenjujemo normalizaciju na tacke i vracamo povratnu vrednost
    res = list(map(lambda x: np.dot(value, np.transpose(x)).tolist() , t))
    
    return value, res

#Normalizovani DLT algoritam koji radi sa neogranicenim brojem tacak krositi se kada se tacke unsoe preko json-a
def DLTnormalizedPure(originali, slike):
    #Primenjujemo f-ju za normaliaciju koja nam vraca noramlizovane tacke i matricu za normalizaciju
    To, tackeo = normalize(originali)
    Tp, tackep = normalize(slike)

    #Primenjujemo DLT algoritam na normalizovane tacke
    Ppom = DLTpure(tackeo, tackep)

    #Vrsimo izracunavanje za koancnu matircu
    Tpinv = np.linalg.inv(Tp)
    
    P = np.dot(np.dot(Tpinv, Ppom), To)
    
    #Stampamo dobijeno resenje
    #print ("Matrica P dobijena normalizovanim DLT algoritmom:\n",P)
    
    return P


#Normalizovani DLT algoritam koji radi sa samo pet tacak Nije obrisan zbog rada preko console
def DLTnormalized(a, b, c, d, e, f, ap, bp, cp, dp, ep, fp):
    #Primenjujemo f-ju za normaliaciju koja nam vraca noramlizovane tacke i matricu za normalizaciju
    To, tackeo = normalize([a, b, c, d, e, f])
    Tp, tackep = normalize([ap, bp, cp, dp, ep, fp])

    #Primenjujemo DLT algoritam na normalizovane tacke
    Ppom = DLT(tackeo[0], tackeo[1], tackeo[2], tackeo[3], tackeo[4], tackeo[5], tackep[0], tackep[1], tackep[2], tackep[3], tackep[4], tackep[5])

    #Vrsimo izracunavanje za koancnu matircu
    Tpinv = np.linalg.inv(Tp)
    
    P = np.dot(np.dot(Tpinv, Ppom), To)
    
    #Stampamo dobijeno resenje
    #print ("Matrica P dobijena normalizovanim DLT algoritmom:\n",P)
    
    return P

def zaokruzi(P, naKolikoCifara):
    A =[]
    for t in P:
        A.append(list(map(lambda x: round(x, naKolikoCifara), t))) 
    
    Pr = np.array(A)
    
    return Pr

def makepoint(j, op):
    if op == 2:
        a = (float(j['a']['X1']), float(j['a']['X2']), float(j['a']['X3']))
        b = (float(j['b']['X1']), float(j['b']['X2']), float(j['b']['X3']))
        c = (float(j['c']['X1']), float(j['c']['X2']), float(j['c']['X3']))
        d = (float(j['d']['X1']), float(j['d']['X2']), float(j['d']['X3']))
        e = (float(j['e']['X1']), float(j['e']['X2']), float(j['e']['X3']))
        f = (float(j['f']['X1']), float(j['f']['X2']), float(j['f']['X3']))

        ap = (float(j['ap']['X1']), float(j['ap']['X2']), float(j['ap']['X3']))
        bp = (float(j['bp']['X1']), float(j['bp']['X2']), float(j['bp']['X3']))
        cp = (float(j['cp']['X1']), float(j['cp']['X2']), float(j['cp']['X3']))
        dp = (float(j['dp']['X1']), float(j['dp']['X2']), float(j['dp']['X3']))
        ep = (float(j['ep']['X1']), float(j['ep']['X2']), float(j['ep']['X3']))
        fp = (float(j['fp']['X1']), float(j['fp']['X2']), float(j['fp']['X3']))

        return [a, b, c, d, e, f, ap, bp, cp, dp , ep, fp ]
    
    else:
        a = (float(j['a']['X1']), float(j['a']['X2']), float(j['a']['X3']))
        b = (float(j['b']['X1']), float(j['b']['X2']), float(j['b']['X3']))
        c = (float(j['c']['X1']), float(j['c']['X2']), float(j['c']['X3']))
        d = (float(j['d']['X1']), float(j['d']['X2']), float(j['d']['X3']))

        ap = (float(j['ap']['X1']), float(j['ap']['X2']), float(j['ap']['X3']))
        bp = (float(j['bp']['X1']), float(j['bp']['X2']), float(j['bp']['X3']))
        cp = (float(j['cp']['X1']), float(j['cp']['X2']), float(j['cp']['X3']))
        dp = (float(j['dp']['X1']), float(j['dp']['X2']), float(j['dp']['X3']))

        return [a, b, c, d, ap, bp, cp, dp]

def makePoint(j):
    tackeImena = j.keys()
    tacke = []
    for i in tackeImena:
        a = float(j[i]['X1'])
        b = float(j[i]['X2'])
        c = float(j[i]['X3'])
        tacke.append((a, b, c))
    if len(tacke) % 2 !=0 :
        print(colored('Niste uneli dovoljno tacka u json fajl', 'red'))
        sys.exit()
    
    pola = int(len(tacke)/2)
    originali = tacke[0:pola]
    slike = tacke[pola:int(len(tacke))]

    return originali, slike
        
def scale(K):
    prva = K[0][0]
    for i in range(len(K)):
        for j in range(len(K[i])):
            K[i][j] = K[i][j]/prva
    return K

def compareMatrixs(M, N):
    scaleM = scale(M)
    scaleN = scale(N)
    
    scaleM = zaokruzi(scaleM, 5)
    scaleN = zaokruzi(scaleN, 5) 
    equal = True

    print(colored(scaleN,'yellow'))
    print()
    print(colored(scaleM,'yellow'))
    print()
    for i in range(len(M)):
        for j in range(len(M[i])):
            if scaleM[i][j] != scaleN[i][j]:
                equal = False

    if equal:
        print(colored("Matrice su jedanke\n", 'red'))
        return 1
    else:
        print(colored("Matrice nisu jedanke\n", 'red'))
        return 0

#Implementacija ulaza u Program


if __name__ == "__main__":
    c = int(input(colored("Ako zelite naivan algoritam pritisnite 1\nAko zelite DLT algoritam pritisinte 2\n", 'blue' ,attrs=['bold'])))

    #Poziv naivnog algoritma
    if c == 1:
        c = input(colored("Da li zelite da unesete tacle iz konzole ili iz tacke.json datoteke?\njson\\console\n",'blue'))
        if c == 'console':
            tacke = {}
            tacka = ['a', 'b', 'c', 'd', 'ap' , 'bp', 'cp', 'dp']

            print(colored("*********************************", "red"))
            for i in range(0, 8):
                x1 = int(input('Unesite X1 koordinatu tacke {kor}: '.format(kor =tacka[i])))
                x2 = int(input('Unesite X2 koordinatu tacke {kor}: '.format(kor =tacka[i])))
                x3 = int(input('Unesite X3 koordinatu tacke {kor}: '.format(kor =tacka[i])))
                print(colored("*********************************", "red"))
                tacke[tacka[i]] = (x1, x2, x3)

            for t in tacke.values():
                print(t)

            
            P = naive_algorith(tacke['a'], tacke['b'], tacke['c'], tacke['d'], tacke['ap'], tacke['bp'], tacke['cp'], tacke['dp'])

            print(colored("Matrica projektivnog preslikavanaj: ", 'magenta'))
            print(colored(P, 'green'))

            print()

            c = input(colored("Da li zelite da se ispisu zaokruzene vrednosti matrice projektovanja?\nd/n\n","blue"))
            if c == 'd':
                decimale = int(input(colored("Na koliko decimala zelite da zaokruzite vrednosti matrice?\n", 'blue')))

                R = zaokruzi(P, decimale)

                print(colored("Zakoruzena matrica:", 'magenta'))

                print(colored(R, 'green'))

        elif c == 'json':
            with open('Naivetacke.json','r') as f:
                x = json.load(f)
            
            a = makepoint(x, 1)

            P = naive_algorith(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])

            print(colored("Matrica projektivnog preslikavanaj: ", 'magenta'))
            print(colored(P, 'green'))

            print()

            c = input(colored("Da li zelite da se ispisu zaokruzene vrednosti matrice projektovanja?\nd/n\n","blue"))
            if c == 'd':
                decimale = int(input(colored("Na koliko decimala zelite da zaokruzite vrednosti matrice?\n", 'blue')))

                R = zaokruzi(P, decimale)

                print(colored("Zakoruzena matrica:", 'magenta'))

                print(colored(R, 'green'))



    #Poziv za DLT algoritam i normalizovani DLT algoritam
    elif c == 2:

        #Nacin unosa tacka
        c = input(colored("Da li zelite da unesete tacle iz konzole ili iz tacke.json datoteke?\njson(preko jsona moze beskonacno)\\console(preko console moze samo 5 tacaka da se unese u DLT)\n",'blue'))
        if c == 'console':
            tacke = {}
            tacka = ['a', 'b', 'c', 'd', 'e', 'f', 'ap', 'bp', 'cp', 'dp', 'ep', 'fp']

            print(colored("*********************************", "red"))
            for i in range(0, 12):
                x1 = int(input('Unesite X1 koordinatu tacke {kor}: '.format(kor =tacka[i])))
                x2 = int(input('Unesite X2 koordinatu tacke {kor}: '.format(kor =tacka[i])))
                x3 = int(input('Unesite X3 koordinatu tacke {kor}: '.format(kor =tacka[i])))
                print(colored("*********************************", "red"))
                tacke[tacka[i]] = (x1, x2, x3)
            
            for t in tacke.values():
                print(t)

            print()
            P = DLT(tacke['a'], tacke['b'], tacke['c'], tacke['d'], tacke['e'], tacke['f'], tacke['ap'], tacke['bp'], tacke['cp'], tacke['dp'], tacke['ep'], tacke['fp'])

            print(colored("Matrica projektivnog preslikavanaj: ", 'magenta'))
            print(colored(P, 'green'))

            c = input(colored("Da li zelite da se ispisu zaokruzene vrednosti matrice projektovanja?\nd/n\n","blue"))
            if c == 'd':
                decimale = int(input(colored("Na koliko decimala zelite da zaokruzite vrednosti matrice?\n", 'blue')))

                R = zaokruzi(P, decimale)

                print(colored("Zakoruzena matrica:", 'magenta'))

                print(colored(R, 'green'))


            print ()

            c = input(colored("Da li zelite i nomalizovani DLP algoritam?\n d/n\n",'blue'))
            if c == 'd':
                P = DLTnormalized(tacke['a'], tacke['b'], tacke['c'], tacke['d'], tacke['e'], tacke['f'], tacke['ap'], tacke['bp'], tacke['cp'], tacke['dp'], tacke['ep'], tacke['fp'])

                print(colored("Matrica projektivnog preslikavanaj: ", 'magenta'))
                print(colored(P, 'green'))

                c = input(colored("Da li zelite da se ispisu zaokruzene vrednosti matrice projektovanja?\nd/n\n","blue"))
                if c == 'd':
                    decimale = int(input(colored("Na koliko decimala zelite da zaokruzite vrednosti matrice?\n", 'blue')))

                    R = zaokruzi(P, decimale)

                    print(colored("Zakoruzena matrica:", 'magenta'))

                    print(colored(R, 'green'))

        elif c == 'json':
            with open('DLTtacke.json','r') as f:
                x = json.load(f)
            
            #a = makepoint(x, 2)

            #P = DLT(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11])

            originali, slike = makePoint(x)
            P1 = DLTpure(originali, slike)

            #print(colored("Matrica projektivnog preslikavanaj: ", 'magenta'))
            #print(colored(P, 'green'))
            #print()
            print(colored(P1, 'green'))

            #_ = compareMatrixs(P, P1)
            c = input(colored("Da li zelite da se ispisu zaokruzene vrednosti matrice projektovanja?\nd/n\n","blue"))
            if c == 'd':
                decimale = int(input(colored("Na koliko decimala zelite da zaokruzite vrednosti matrice?\n", 'blue')))

                R = zaokruzi(P1, decimale)

                print(colored("Zakoruzena matrica:", 'magenta'))

                print(colored(R, 'green'))
            
            c = input(colored("Da li zelite i nomalizovani DLP algoritam?\n d/n\n",'blue'))
            if c == 'd':
                P = DLTnormalizedPure(originali, slike)

                print(colored("Matrica projektivnog preslikavanaj: ", 'magenta'))
                print(colored(P, 'green'))

                c = input(colored("Da li zelite da se ispisu zaokruzene vrednosti matrice projektovanja?\nd/n\n","blue"))
                if c == 'd':
                    decimale = int(input(colored("Na koliko decimala zelite da zaokruzite vrednosti matrice?\n", 'blue')))

                    R = zaokruzi(P, decimale)

                    print(colored("Zakoruzena matrica:", 'magenta'))

                    print(colored(R, 'green'))


            k = input(colored("Da li zelite da uporedite DLT i normalizovani DLT u odnosu na pomerene tacke iz fajlaDLTCompare.json fajla?\nd/n\n",'blue'))

            if k == 'd':
                with open('DLTCompare.json','r') as f:
                     y = json.load(f)

                originali, slike = makePoint(y)
                Q = DLTpure(originali, slike)
                Q1 = DLTnormalizedPure(originali, slike)
                P = DLTnormalizedPure(originali, slike)

                print(colored("Kada uporedimo dve matrice preslikavanja dobijene DLT algoritmom za transliranje tacke\n"), "blue")
                compareMatrixs(P1, Q)
                print()

                print(colored("Kada uporedimo dve matrice dobijene normalizovani DLT algoritmom za translirane tacke\n", 'blue'))
                compareMatrixs(P, Q1)


#Zakljucak iz mog test primera je da je normalizovani DLT invarijantan u odnosu na promenu koordinatnog sistema
