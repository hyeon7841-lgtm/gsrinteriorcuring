import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.path import Path

st.set_page_config(layout="wide")

# =====================
# 상수
# =====================
TEMP_MIN, TEMP_MAX = -10, 40
HEATER_KCAL = 17600
HEATER_WATT = HEATER_KCAL * 1.163
INFLUENCE_RADIUS = 10.0
DT = 60
SIM_HOURS = 6
ALPHA = 0.05
MIXING = 0.15

WALL_U = {
    "조적벽": 2.0,
    "콘크리트벽": 1.7,
    "샌드위치판넬": 0.5
}

# =====================
# 초기화
# =====================
def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]

# =====================
# 자동 열풍기 배치
# =====================
def auto_place_heaters(space_pts, n):
    pts = np.array(space_pts)
    cx, cy = pts[:,0].mean(), pts[:,1].mean()

    heaters = []

    if n == 1:
        heaters.append({"x": cx, "y": cy, "angle": 0.0})
    else:
        minx, miny = pts.min(axis=0)
        maxx, maxy = pts.max(axis=0)

        xs = [minx + (maxx-minx)*0.3, minx + (maxx-minx)*0.7]

        for x in xs:
            angle = np.arctan2(cy - cy, cx - x)
            heaters.append({"x": x, "y": cy, "angle": angle})

    return heaters

# =====================
# 시뮬레이션
# =====================
def run_simulation(space_pts, heaters, wall, height, T0, Tout):
    pts = np.array(space_pts)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)

    nx = ny = 60
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    poly = Path(space_pts)
    mask = poly.contains_points(
        np.vstack((X.flatten(), Y.flatten())).T
    ).reshape(X.shape)

    T = np.full_like(X, T0)
    history = []

    area = (xmax-xmin)*(ymax-ymin)
    wall_area = 2*((xmax-xmin)+(ymax-ymin))*height
    rho, cp = 1.2, 1000
    C = rho*cp*area*height
    U = WALL_U[wall]

    steps = int(SIM_HOURS*3600/DT)

    for step in range(steps):
        Tn = T.copy()

        # 확산
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                if mask[j,i]:
                    Tn[j,i] += ALPHA*(T[j+1,i]+T[j-1,i]+T[j,i+1]+T[j,i-1]-4*T[j,i])

        # 열풍기
        for h in heaters:
            hx, hy, a = h["x"], h["y"], h["angle"]
            ca, sa = np.cos(a), np.sin(a)

            for i in range(nx):
                for j in range(ny):
                    if not mask[j,i]:
                        continue
                    dx, dy = X[j,i]-hx, Y[j,i]-hy
                    r = np.hypot(dx,dy)
                    if r==0 or r>INFLUENCE_RADIUS:
                        continue
                    proj = dx*ca + dy*sa
                    if proj <= 0:
                        continue
                    w = np.exp(-r/3)*(proj/r)
                    Tn[j,i] += (HEATER_WATT*DT/C)*w

        # 공기 혼합
        meanT = np.mean(Tn[mask])
        Tn[mask] += MIXING*(meanT-Tn[mask])

        # 벽 손실
        Qloss = U*wall_area*(meanT-Tout)*DT
        Tn[mask] -= Qloss/C

        T = np.clip(Tn, TEMP_MIN, TEMP_MAX)

        if step%30==0:
            history.append(T.copy())

    return history, x, y, mask

# =====================
# UI
# =====================
st.title("🔥 난방 열 시뮬레이터")

if st.button("🔄 전체 초기화"):
    reset_all()
    st.rerun()

# ---------------------
# 1단계 공간 정의
# ---------------------
st.header("1️⃣ 공간 정의")

if "space" not in st.session_state:
    st.session_state.space = []

c1,c2 = st.columns(2)
px = c1.number_input("X", format="%.2f")
py = c2.number_input("Y", format="%.2f")

if st.button("좌표 추가"):
    st.session_state.space.append((px,py))

if len(st.session_state.space)>=1:
    fig = go.Figure()
    xs,ys = zip(*st.session_state.space)
    fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines+markers"))
    if len(xs)>=3:
        fig.add_trace(go.Scatter(
            x=list(xs)+[xs[0]],
            y=list(ys)+[ys[0]],
            mode="lines",
            line=dict(dash="dot")
        ))
    fig.update_yaxes(scaleanchor="x")
    st.plotly_chart(fig,use_container_width=True)

# ---------------------
# 2단계 열풍기 배치
# ---------------------
st.header("2️⃣ 열풍기 배치")

heater_n = st.radio("열풍기 수량",[1,2],horizontal=True)
manual_heaters = []

for i in range(heater_n):
    st.subheader(f"열풍기 {i+1}")
    c1,c2,c3 = st.columns(3)
    hx = c1.number_input("X", key=f"x{i}")
    hy = c2.number_input("Y", key=f"y{i}")
    ang = c3.slider("풍향(°)", -180,180,0, key=f"a{i}")
    manual_heaters.append({"x":hx,"y":hy,"angle":np.deg2rad(ang)})

auto_heaters = auto_place_heaters(st.session_state.space, heater_n)

def draw_layout(title, heaters):
    fig = go.Figure()
    xs,ys = zip(*(st.session_state.space+[st.session_state.space[0]]))
    fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines",line=dict(color="black")))
    for h in heaters:
        hx,hy,a = h["x"],h["y"],h["angle"]
        fig.add_trace(go.Scatter(
            x=[hx],y=[hy],
            mode="markers",
            marker=dict(size=14,symbol="triangle-up",color="red")
        ))
        L = INFLUENCE_RADIUS*0.3
        fig.add_trace(go.Scatter(
            x=[hx,hx+L*np.cos(a)],
            y=[hy,hy+L*np.sin(a)],
            mode="lines",
            line=dict(width=3,color="orange")
        ))
    fig.update_yaxes(scaleanchor="x")
    fig.update_layout(title=title,height=400)
    return fig

if len(st.session_state.space)>=3:
    colA,colB = st.columns(2)
    colA.plotly_chart(draw_layout("👤 수동 배치",manual_heaters),use_container_width=True)
    colB.plotly_chart(draw_layout("🤖 자동 배치",auto_heaters),use_container_width=True)

# ---------------------
# 3단계 시뮬레이션
# ---------------------
st.header("3️⃣ 시뮬레이션")

wall = st.selectbox("벽체", list(WALL_U.keys()))
height = st.number_input("천장 높이(m)",value=3.0)

c1,c2 = st.columns(2)
T0 = c1.number_input("초기 내부온도",value=10.0)
Tout = c2.number_input("외부온도",value=0.0)

if st.button("🔥 시뮬레이션 실행"):
    st.session_state.manual = run_simulation(
        st.session_state.space, manual_heaters, wall, height, T0, Tout
    )
    st.session_state.auto = run_simulation(
        st.session_state.space, auto_heaters, wall, height, T0, Tout
    )

# ---------------------
# 결과
# ---------------------
if "manual" in st.session_state:
    for title,key in [("👤 수동 배치","manual"),("🤖 자동 배치","auto")]:
        hist,x,y,mask = st.session_state[key]
        frames = [
            go.Frame(data=[go.Heatmap(
                z=T,x=x,y=y,
                zmin=TEMP_MIN,zmax=TEMP_MAX,
                colorscale="Turbo",
                hovertemplate="X %{x:.1f}m<br>Y %{y:.1f}m<br>온도 %{z:.1f}°C"
            )])
            for T in hist
        ]
        fig = go.Figure(data=frames[0].data,frames=frames)
        fig.update_layout(
            title=title,
            updatemenus=[{
                "type":"buttons",
                "buttons":[{"label":"▶ 재생","method":"animate","args":[None]}]
            }]
        )
        fig.update_yaxes(scaleanchor="x")
        st.plotly_chart(fig,use_container_width=True)
