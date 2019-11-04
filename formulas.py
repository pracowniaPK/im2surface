import sympy as sym

from utils import pl


u1 = sym.symbols('u_1')
u2 = sym.symbols('u_2')
u3 = sym.symbols('u_3')
u4 = sym.symbols('u_4')
v1 = sym.symbols('v_1')
v2 = sym.symbols('v_2')
v3 = sym.symbols('v_3')

um1 = sym.symbols('u-')
up1 = sym.symbols('u+')
d = sym.symbols('d')

uc =  (um1 - up1) / (2 * d)

ux = uc.subs(up1, u2).subs(um1, u4)
uy = uc.subs(up1, u1).subs(um1, u3)
e1 = (ux*v1 + uy*v2 - v3)/((sym.sqrt(v1**2 + v2**2 + v3**2))*sym.sqrt(ux**2 + uy**2 + 1))

# pl(e1)
e1_f = sym.lambdify((u1, u2, u3, u4, v1, v2, v3, d), e1)
