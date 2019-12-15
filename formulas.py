import sympy as sym

from utils import pl


# funkcja zwracająca wartość pixela
u1, u2, u3, u4 = sym.symbols('u_1 u_2 u_3 u_4')
v1, v2, v3 = sym.symbols('v_1 v_2 v_3')
um1, up1, d = sym.symbols('u- u+ d')

uc =  (um1 - up1) / (2 * d)
ux = uc.subs(um1, u2).subs(up1, u4)
uy = uc.subs(um1, u3).subs(up1, u1)
e1 = (ux*v1 + uy*v2 - v3)/((sym.sqrt(v1**2 + v2**2 + v3**2))*sym.sqrt(ux**2 + uy**2 + 1))
# pl(e1)
e1_f = sym.lambdify((u1, u2, u3, u4, v1, v2, v3, d), e1)
#   +------------------> 
#   |   x  u1  x      x
#  y|   u4 i,j u2
#   v   x  u3  x


# funkcja zwracająca gradient dla danego wymiaru
ud0, ud1, ud2, ud3, ud4, ud5, ud6, ud7, ud8 = sym.symbols(
    'u_d0 u_d1 u_d2 u_d3 u_d4 u_d5 u_d6 u_d7 u_d8')
w1, w2, w3, w4 =  sym.symbols('w_1 w_2 w_3 w_4')

e4 = (
    (e1.subs(u1, ud1).subs(u2, ud2).subs(u3, ud0).subs(u4, ud8) - w1)**2 +
    (e1.subs(u1, ud2).subs(u2, ud3).subs(u3, ud4).subs(u4, ud0) - w2)**2 + 
    (e1.subs(u1, ud0).subs(u2, ud4).subs(u3, ud5).subs(u4, ud6) - w3)**2 +
    (e1.subs(u1, ud8).subs(u2, ud0).subs(u3, ud6).subs(u4, ud7) - w4)**2
    )

de4 = sym.diff(e4, ud0)
# pl(de4)
de4_f = sym.lambdify(
    (ud0, ud1, ud2, ud3, ud4, ud5, ud6, ud7, ud8,
    w1, w2, w3, w4,
    v1, v2, v3, d),
    de4
    )
#  +->x       1
# y|       8  w1 2
#  v    7  w4 0=i,j 3
#          6  w3 4
#             5

