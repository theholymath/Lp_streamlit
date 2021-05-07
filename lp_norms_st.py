import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize


def lp_ball_pts(radius, mp):
    """ For plotting - plots the LP-Ball of radius, radius

    Parameters:
    -----------
    radius (float): The "radius" of the LP-ball
    p (float): L(p) space
    """
    alpha = np.linspace(0, 2*np.pi, 2000, endpoint=True)
    x = np.cos(alpha)
    y = np.sin(alpha)

    vecs = np.array([x, y])

    norms = np.sum(np.abs(vecs)**p, axis=0)**(1/p)
    norm_vecs = radius*vecs/norms

    return norm_vecs


st.title('Exploring the differences between $||x||_1, ||x||_2$, and $||x||_{\infty}$.')
st.subheader('Minimizing $||x||_p$ given $ax + by + c = 0$')
p = st.slider('p:', 1.0, 10.0, step=0.25)

st.sidebar.markdown('$ax + by + c = 0$')

form = st.sidebar.form(key='my-form')
a = float(form.text_input('a', 4))
b = float(form.text_input('b', 2))
c = float(form.text_input('c', 1))
submit = form.form_submit_button('Submit')


if submit:
    st.sidebar.markdown(f'${a}x + {b}y + {c} = 0$')
#st.sidebar.slider('a', min_value=None , max_value=None , value=None , step=None , format=None , key=None )



def func_to_minimize(x):
    return np.power((np.abs(x[0]))**p + (np.abs(x[1]))**p, 1/p)


def eqn_of_line_optim(x):
    return a*x[0] + b*x[1] + c


def eqn_line_y_equals(x):
    return (-c - a*x)/b

if p != 1.0:
    p_temp = p
    p = 1.0
    cons = ({'type': 'eq', 'fun': eqn_of_line_optim})
    res = minimize(func_to_minimize, (2, 0), method='SLSQP', constraints=cons)
    width = res.fun
    p = p_temp
    cons = ({'type': 'eq', 'fun': eqn_of_line_optim})
    res = minimize(func_to_minimize, (2, 0), method='SLSQP', constraints=cons)
else:
    cons = ({'type': 'eq', 'fun': eqn_of_line_optim})
    res = minimize(func_to_minimize, (2, 0), method='SLSQP', constraints=cons)
    width = res.fun

sns.set(style='ticks')

x = np.linspace(-2*width,2*width,1000)
y = eqn_line_y_equals(x)
lp_func = lp_ball_pts(res.fun, p)
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(x, y)
ax.scatter(res.x[0], res.x[1],c='red',label = 'min point for Lp norm')
ax.plot(lp_func[0],lp_func[1])
ax.set_aspect('equal')
ax.grid(True, which='both')
sns.despine(ax=ax, offset=0) # the important part here
ax.set_ylim([-2*width,2*width])
ax.set_xlim([-2*width,2*width])
st.pyplot(fig, height=100)

