import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation


def a_alpha(rt, t, p):
	r1,v1,theta1,omega1,r2,v2,theta2,omega2,r3,v3,theta3,omega3=rt
	m1,m2,m3,G=p
	c12=np.cos(theta1-theta2)
	s12=np.sin(theta1-theta2)
	c13=np.cos(theta1-theta3)
	s13=np.sin(theta1-theta3)
	c23=np.cos(theta2-theta3)
	s23=np.sin(theta2-theta3)
	r12=((r1**2)+(r2**2)-2*r1*r2*c12)**(-3/2)
	r13=((r1**2)+(r3**2)-2*r1*r3*c13)**(-3/2)
	r23=((r2**2)+(r3**2)-2*r2*r3*c23)**(-3/2)
	r12c=r1-r2*c12
	r13c=r1-r3*c13
	r21c=r2-r1*c12
	r23c=r2-r3*c23
	r31c=r3-r1*c13
	r32c=r3-r2*c23
	C1=-2*G*(m2*r12*r12c+m3*r13*r13c)
	C2=-2*G*(m1*r12*r21c+m3*r23*r23c)
	C3=-2*G*(m1*r13*r31c+m2*r23*r32c)
	D1=-2*(G/r1)*(m2*r2*s12*r12+m3*r3*s13*r13)
	D2=2*(G/r2)*(m1*r1*s12*r12-m3*r3*s23*r23)
	D3=2*(G/r3)*(m1*r1*s13*r13+m2*r2*s23*r23)
	return [v1,r1*(omega1**2)+C1,omega1,-2*(v1/r1)*omega1+D1,v2,r2*(omega2**2)+C2,\
	omega2,-2*(v2/r2)*omega2+D2,v3,r3*(omega3**2)+C3,omega3,-2*(v3/r3)*omega3+D3]

rng=np.random.default_rng(9348382)

mass_max=1
mass_min=2
size_factor=1/100
radius_max=1
radius_min=5
velocity_max=0
velocity_min=0
angular_velocity_max=-1
angular_velocity_min=1

m1=(mass_max-mass_min)*rng.random()+mass_min
m2=(mass_max-mass_min)*rng.random()+mass_min
m3=(mass_max-mass_min)*rng.random()+mass_min
ro1=(radius_max-radius_min)*rng.random()+radius_min
ro2=(radius_max-radius_min)*rng.random()+radius_min
ro3=(radius_max-radius_min)*rng.random()+radius_min
vo1=(velocity_max-velocity_min)*rng.random()+velocity_min
vo2=(velocity_max-velocity_min)*rng.random()+velocity_min
vo3=(velocity_max-velocity_min)*rng.random()+velocity_min
theta1=360*rng.random()
theta2=360*rng.random()
theta3=360*rng.random()
omega1=(angular_velocity_max-angular_velocity_min)*rng.random()+angular_velocity_min
omega2=(angular_velocity_max-angular_velocity_min)*rng.random()+angular_velocity_min
omega3=(angular_velocity_max-angular_velocity_min)*rng.random()+angular_velocity_min
G=1

cnvrt=np.pi/180
theta1*=cnvrt
theta2*=cnvrt
theta3*=cnvrt


p=[m1,m2,m3,G]
rt=[ro1,vo1,theta1,omega1,ro2,vo2,theta2,omega2,ro3,vo3,theta3,omega3]

tf = 60
nfps = 60
nframes = tf * nfps
t = np.linspace(0, tf, nframes)

rth = odeint(a_alpha, rt, t, args = (p,))

r1=rth[:,0]
th1=rth[:,2]
r2=rth[:,4]
th2=rth[:,6]
r3=rth[:,8]
th3=rth[:,10]

x1=r1*np.cos(th1)
y1=r1*np.sin(th1)
x2=r2*np.cos(th2)
y2=r2*np.sin(th2)
x3=r3*np.cos(th3)
y3=r3*np.sin(th3)

xamax=[]
xamin=[]
yamax=[]
yamin=[]
xamax.append(max(x1))
xamax.append(max(x2))
xamax.append(max(x3))
xamin.append(min(x1))
xamin.append(min(x2))
xamin.append(min(x3))
yamax.append(max(y1))
yamax.append(max(y2))
yamax.append(max(y3))
yamin.append(min(y1))
yamin.append(min(y2))
yamin.append(min(y3))

dx=max(xamax)-min(xamin)
dy=max(yamax)-min(yamin)
dr=np.sqrt(dx**2+dy**2)
maxmr=size_factor*dr
mass_list=[m1,m2,m3]
mmax=max(mass_list)
mmax=1/mmax
f=maxmr*mmax
mr=[f*m1,f*m2,f*m3]
shift=max(mr)


xmax=max(xamax)+2*shift
xmin=min(xamin)-2*shift
ymax=max(yamax)+2*shift
ymin=min(yamin)-2*shift

v1=rth[:,1]
w1=rth[:,3]
v2=rth[:,5]
w2=rth[:,7]
v3=rth[:,9]
w3=rth[:,11]

ke1=0.5*m1*((v1**2)+((r1*w1)**2))
ke2=0.5*m2*((v2**2)+((r2*w2)**2))
ke3=0.5*m3*((v3**2)+((r3*w3)**2))
ke=ke1+ke2+ke3
rr12=np.sqrt((r1**2)+(r2**2)-2*r1*r2*np.cos(th1-th2))
rr13=np.sqrt((r1**2)+(r3**2)-2*r1*r3*np.cos(th1-th3))
rr23=np.sqrt((r2**2)+(r3**2)-2*r2*r3*np.cos(th2-th3))
pe=-(2*G*m1*m2/rr12)-(2*G*m1*m3/rr13)-(2*G*m2*m3/rr23)
E=ke+pe
Emax=abs(max(E))
ke/=Emax
pe/=Emax
E/=Emax
Emax=max(E)
ke-=Emax
pe-=Emax
E-=Emax

fig, a=plt.subplots()
fig.tight_layout()

def run(frame):
	plt.clf()
	plt.subplot(211)
	circle=plt.Circle((x1[frame],y1[frame]),radius=mr[0],fc='r')
	plt.gca().add_patch(circle)
	circle=plt.Circle((x2[frame],y2[frame]),radius=mr[1],fc='r')
	plt.gca().add_patch(circle)
	circle=plt.Circle((x3[frame],y3[frame]),radius=mr[2],fc='r')
	plt.gca().add_patch(circle)
	plt.title("3 Body Orbital Dynamics")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(t[0:frame],ke[0:frame],'r',lw=1)
	plt.plot(t[0:frame],pe[0:frame],'b',lw=1)
	plt.plot(t[0:frame],E[0:frame],'g',lw=1)
	plt.xlim([0,tf])
	plt.title("Energy (Rescaled and Shifted)")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')


ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('gravity_3body_ode_wgphs.mp4', writer=writervideo)

plt.show()


