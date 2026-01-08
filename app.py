import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.path import Path
import io

st.set_page_config(layout="wide")

# =========================
# 상수
# =========================
TEMP_MIN, TEMP_MAX = -10, 40

HEATER_KCAL = 17600
HEATER_WATT = HEATER_KCAL * 1.163
INFLUENCE_RADIUS = 10.0

DT = 60
SIM_HOURS = 9
ALPHA = 0.03
MIXING = 0.08

WALL_U = {
    "조적벽": 2.0,
    "콘크리트벽": 1.7,
    "샌드위치판넬": 0.5
}

# =========================
# 초기화
# =========================
def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]

# =========================
# 시뮬레이션
# =========================
def run_simulation(space_pts, heaters, wall_type, height, init_temp, ext_temp):
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

    T = np.full_like(X, init_temp)
    T_hist = []

    area = (xmax - xmin) * (ymax - ymin)
    perimeter = 2 * ((xmax - xmin) + (ymax - ymin))
    wall_area = perimeter * height

    rho, cp = 1.2, 1000
    C = rho * cp * area * height
    U = WALL_U[wall_type]

    steps = int(SIM_HOURS * 3600 / DT)

    for step in range(steps):
        Tn = T.copy()

        # 확산
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                if mask[j, i]:
                    Tn[j, i] += ALPHA * (
                        T[j+1,i] + T[j-1,i] + T[j,i+1] + T[j,i-1] - 4*T[j,i]
                    )

        # 열풍기
        for h in heaters:
            hx, hy, angle = h["x"], h["y"], h["angle"]
            ca, sa = np.cos(angle), np.sin(angle)

            for i in range(nx):
                for j in range(ny):
                    if not mask[j, i]:
                        continue
                    dx = X[j, i] - hx
                    dy = Y[j, i] - hy
                    r = np.hypot(dx, dy)
                    if r == 0 or r > INFLUENCE_RADIUS:
                        continue
                    proj = dx*ca + dy*sa
                    if proj <= 0:
                        continue
                    w = np.exp(-r/3) * (proj / r)
                    Tn[j, i] += (HEATER_WATT * DT / C) * w

        # 공기 혼합 (폐쇄공간)
        T_mean = np.mean(Tn[mask])
        Tn[mask] += MIXING * (T_mean - Tn[mask])

        # 벽체 열손실
        Q_loss = U * wall_area * (T_mean - ext_temp) * DT
        Tn[mask] -= Q_loss / C

        T = np.clip(Tn, TEMP_MIN, TEMP_MAX)

        if step % 30 == 0:
            T_hist.append(T.copy())

    return T_hist, x, y, mask

# =========================
# UI
# =========================
st.title("🔥 난방 열 시뮬레이터")

if st.button("🔄 전체 초기화"):
    reset_all()
    st.rerun()

# ---------- 1단계 ----------
st.header("1️⃣ 공간 정의")
if "space" not in st.session_state:
    st.session_state.space = []

c1, c2 = st.columns(2)
px = c1.number_input("X (m)", format="%.3f")
py = c2.number_input("Y (m)", format="%.3f")

if st.button("좌표 추가"):
    st.session_state.space.append((px, py))

if len(st.session_state.space) >= 3:
    xs, ys = zip(*(st.session_state.space + [st.session_state.space[0]]))
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers"))
    fig.update_yaxes(scaleanchor="x")
    st.plotly_chart(fig, use_container_width=True)

# ---------- 2단계 ----------
st.header("2️⃣ 열풍기 & 온도 조건")
heater_n = st.radio("열풍기 수량", [1, 2], horizontal=True)

heaters = []
for i in range(heater_n):
    st.markdown(f"### 🔥 열풍기 {i+1}")
    c1, c2, c3 = st.columns(3)
    hx = c1.number_input("X (m)", key=f"hx{i}", format="%.3f")
    hy = c2.number_input("Y (m)", key=f"hy{i}", format="%.3f")
    ang = c3.slider("풍향 (°)", -180, 180, 20, key=f"ang{i}")
    heaters.append({"x": hx, "y": hy, "angle": np.deg2rad(ang)})

st.subheader("🌡️ 온도 조건 설정")
c1, c2 = st.columns(2)
init_temp = c1.number_input("시작 내부 온도 (°C)", value=10.0, step=0.5)
ext_temp = c2.number_input("외부 온도 (°C)", value=0.0, step=0.5)

# ---------- 3단계 ----------
st.header("3️⃣ 시뮬레이션")
wall = st.selectbox("벽체 재질", list(WALL_U.keys()))
height = st.number_input("천장 높이 (m)", value=3.0)

if st.button("🔥 시뮬레이션 실행"):
    st.session_state.result = run_simulation(
        st.session_state.space,
        heaters,
        wall,
        height,
        init_temp,
        ext_temp
    )

# ---------- 결과 ----------
if "result" in st.session_state:
    T_hist, x, y, mask = st.session_state.result
    idx = st.slider("시간 (30분 간격)", 0, len(T_hist)-1)

    fig = go.Figure(go.Heatmap(
        z=T_hist[idx],
        x=x, y=y,
        zmin=TEMP_MIN, zmax=TEMP_MAX,
        colorscale="Turbo"
    ))
    fig.update_yaxes(scaleanchor="x")
    st.plotly_chart(fig, use_container_width=True)
