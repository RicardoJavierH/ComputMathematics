import math

# Coeficientes dos wavelets de Daubechies.
D6=[4.70467210E-01,1.14111692E+00,6.50365000E-01,-1.90934420E-01,-1.20832210E-01,4.98175000E-02]
D6 = [x/math.sqrt(2.0) for x in D6]
sq3 = math.sqrt(3.0)
D4 = [1 + sq3, 3+sq3, 3-sq3, 1-sq3]
D4 = [x/(4*math.sqrt(2)) for x in D4]
D2 = [1.0/math.sqrt(2), 1.0/math.sqrt(2)]

def make_filters(h0):
    """
        Deriva os filtros passa alta e os filtros reversos
        a partir do filtro passa baixa.
    """
    f0 = reverse(h0)
    f1 = mirror(h0)
    h1 = reverse(f1)
    return (h0, h1, f0, f1)

def reverse(h):
    """
        Inverte os elementos de um vetor
    """
    return reversed(h)

def mirror(h):
    """
        Troca o sinal dos elementos em posições
        ímpares.
    """
    return [x * (-1)**i for i,x in enumerate(h)]

def downsample(h):
    """
        Retira o segundo elemento de cada dois.
    """
    return h[::2]

def upsample(h):
    """
        Intercala o número 0 entre os valores de um vetor.
    """
    ret = []
    for x in h:
        ret.extend([x, 0])
    return ret

def add(v1, v2):
    """
        soma os elementos de dois vetores.
    """
    return (a+b for a,b in zip(v1, v2))

def rotate_left(v, size):
    """
        Usado para compensar o desvio temporal do
        banco de filtros
    """
    return v[size:] + v[:size]

def convolution(filter, data):
    """
        Calcula uma convolução discreta de dois vetores.
    """
    return [sum(data[(i-j) % len(data)] * filter[j]
                for j in range(len(filter)))
            for i in range(len(data))]

def dwt(data, filter):
    """
        Decompõe um sinal usando a transformada
        discreta de wavelet aplicada recursivamente
        (encadeamentode bancos de filtros) conforme
        visto em:
        http://pt.wikipedia.org/wiki/Transformada_discreta_de_wavelet
    """
    (h0, h1, f0, f1) = make_filters(filter)
    alfa = list(data)
    beta = []
    while len(alfa) > len(filter):
        tmp = downsample(rotate_left(convolution(h1, alfa),len(filter)-1))
        alfa = downsample(rotate_left(convolution(h0, alfa),len(filter)-1))
        beta = tmp + beta
    return alfa + beta

def idwt(data, filter):
    """
        Recompõe o sinal decomposto pela DWT, conforme:
        http://pt.wikipedia.org/wiki/Transformada_discreta_de_wavelet
    """
    (h0, h1, f0, f1) = make_filters(filter)
    size = 1
    while size < len(filter):
        size *= 2
    size /= 2
    ret = list(data)
    while size < len(data):
        alfa = convolution(f0, upsample(ret[:size]))
        beta = convolution(f1, upsample(ret[size:2*size]))
        ret = add(alfa, beta) + ret[2*size:]
        size *= 2
    return ret

filter = D6
(h0, h1, f0, f1) = make_filters(filter)
data = [53,75,97,29,11,33,44,66,88,130,62,33,674,45,36,67]
ret = dwt(data, filter)
print(ret)
ret = idwt(ret, filter)
print(ret)