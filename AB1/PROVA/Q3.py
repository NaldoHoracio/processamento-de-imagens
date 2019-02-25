a = float(input('Digite um número: '))
b = float(input('Digite outro número: '))
c = float(input('Digite algum numero: '))
if a == b == c:
    print('É um triangulo equilatero')
elif a != b and a != c and b != c:
    print('É um triangulo escaleno')
else:
    print('Triangulo isosceles')
