import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
from code import naive_algorith
from PIL import Image
import json

def DLT(a, b, c, d, ap, bp, cp, dp):

    #Pravimo potrebne vektore za matricu korespondencije
    A1, A2 = makeCorespondence(a, ap)
    B1, B2 = makeCorespondence(b, bp)
    C1, C2 = makeCorespondence(c, cp)
    D1, D2 = makeCorespondence(d, dp)

    #Pravimo matricu
    Mat = np.array([A1, A2, B1, B2, C1, C2, D1, D2])

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

def makeCorespondence(a, ap):
    v1 = [0, 0, 0, (-1)*ap[2]*a[0], (-1)*ap[2]*a[1], (-1)*ap[2]*a[2], ap[1]*a[0], ap[1]*a[1], ap[1]*a[2]]
    v2 = [ap[2]*a[0], ap[2]*a[1], ap[2]*a[2], 0, 0, 0, (-1)*ap[0]*a[0], (-1)*ap[0]*a[1], (-1)*ap[0]*a[2]]

    return v1, v2

def distance(A, B):
    affinize = lambda x: [round(x[0]/x[2]), round(x[1]/x[2])]
    A =affinize(A)
    B=affinize(B)
    return np.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)

def tackeProjekcije(dots, odnos):
    A, _, _, D = dots[0], dots[1], dots[2], dots[3]
    Ap,Bp,Cp,Dp = [0,0,1], [0,0,1], [0,0,1], [0,0,1]

    Ap[0] = A[0]
    Ap[1] = A[1]

    Dp[0] = A[0]
    Dp[1] = D[1]

    Bp[1] = A[1]

    visina = distance(Dp, A)
    sirina = odnos * visina

    Bp[0] = A[0] + sirina

    Dp[1] = A[1] - visina

    Cp[0] = Bp[0]
    Cp[1] = Dp[1]

    return Ap, Bp, Cp, Dp

def makepoint(j):
    a = [int(j['a']['X1']), int(j['a']['X2']), int(j['a']['X3'])]
    b = [int(j['b']['X1']), int(j['b']['X2']), int(j['b']['X3'])]
    c = [int(j['c']['X1']), int(j['c']['X2']), int(j['c']['X3'])]
    d = [int(j['d']['X1']), int(j['d']['X2']), int(j['d']['X3'])]

    return a, b, c, d


print(colored("Mozete da izmenite koordinate tacka u json fajlu distorzija.json: ", "blue"))
with open('distorzija.json','r') as f:
    x = json.load(f)
        
a, b, c, d = makepoint(x)

print(colored("Unesite odnost u obliku veriklana ivica kroz horizontalana:", 'red'))
m = float(input("Verikalna ivica:\n"))
print()
n = float(input("Horizontalna ivica:\n"))

odnos  = m/n

ap, bp, cp, dp = tackeProjekcije([a, b, c, d], odnos)

print (colored("Unesite 1 ako zelite DLT algoritam za uklanajnje distorzije, a 2 za naivni algoritam:\n", 'blue'))
opcija = int(input())

if opcija == 1: 
    P = DLT(a, b, c, d, ap, bp, cp, dp)
    Q = np.linalg.inv(P)
else:
    P = naive_algorith(a, b, c, d, ap, bp, cp, dp)
    Q = np.linalg.inv(P)

image = Image.open("Slika.bmp")
pixels = image.load()
mode = 'RGB'
color = (0, 0, 0)
size = (2*image.size[0], 2*image.size[1])
new_image = Image.new(mode, size ,color)

new_pixels = new_image.load()

affinize = lambda x: [round(x[0]/x[2]), round(x[1]/x[2])]

for i in range(image.size[0]):
    for j in range (image.size[1]):
        A = np.matmul(P, [i, j, 1])
        A = affinize(A)
        if A[0]>0 and A[0]<new_image.size[0] and A[1]>0 and A[1]<new_image.size[1]:
            new_pixels[A[0], A[1]] = pixels[i, j]

for i in range(new_image.size[0]):
    for j in range(new_image.size[1]):
       if new_pixels[i,j] == (0,0,0):
                A = np.matmul(Q,[i,j,1])
                A = affinize(A)
                if A[0]>0 and A[0]<image.size[0] and A[1]>0 and A[1]<image.size[1]:
                    new_pixels[i,j] = pixels[A[0],A[1]]

if opcija == 1:
    new_image.save("DistozijaDLT.bmp")
else:
    new_image.save("DistorzijaNaive.bmp")