

class A:
    def __init__(self, fac=1):
        self._fac = fac
        self._a = 1

    def update(self, x):
        self._a = x

    @property
    def a(self):
        self._a *= self._fac
        return self._a

    @a.setter
    def a(self, x):
        self._a = x

class B(A):

    def __init__(self):
        super().__init__()
        self._a = 2

    @property
    def a(self):
        return self._a * 2
        
    @A.a.setter
    def a(self, x):
        self._a = 2 * x


a = A()
b = B()
b.a = 4

print("##### a ######")
print(a.a)
print(a.a)
print(a.a)

print("##### b ######")
print(b.a)
print(b.a)
print(b.a)

