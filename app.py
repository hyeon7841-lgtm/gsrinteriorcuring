import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.path import Path

st.set_page_config(layout="wide")

# =========================
# 기본 상수
# =========================
TEMP_MIN, TEMP_MAX = -10, 45
INFLUENCE_RADIUS = 10.0
PREVIEW_RADIUS = INFLUENCE_RADIUS * 0.3
HEATER_POWER = 18000  # W
SPREAD_ANGLE = np.deg2rad(40)

WALL_U = {
    "조적벽": 2.0,
    "콘크리트벽": 1.7,
    "샌드위치판넬": 0.5
}

# =========================
# 유틸
# =========================
def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]

# =========================
# 시뮬레이션
# =========================
def run_simulation(space, heaters, wall_u, height, t_init, t_ext):
    pts = np.array(space)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)

    nx = ny = 60
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    poly = Path(space)
    mask = poly.contains_points(
        np.vstack((X.flatten(), Y.flatten())).T
    ).reshape(X.shape)

    T = np.full_like(X, t_init)
    T_hist = []

    area = (xmax - xmin) * (ymax - ymin)
    perimeter = 2 * ((xmax - xmin) + (ymax - ymin))
    wall_area = perimeter * height

    rho, cp = 1.2, 1000
    C = rho * cp * area * height

    dt = 60
    steps = int(6 * 3600 / dt)
    alpha = 0.12

    for step in range(steps):
        Tn = T.copy()

        # 내부 확산 (균질화)
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                if mask[j, i]:
                    Tn[j, i] += alpha * (
                        T[j+1,i] + T[j-1,i] +
                        T[j,i+1] + T[j,i-1] - 4*T[j,i]
                    )

        # 열풍기
        for h in heaters:
            hx, hy, ang = h
            for i in range(nx):
                for j in range(ny):
                    if not mask[j, i]:
                        continue
                    dx = X[j,i] - hx
                    dy = Y[j,i] - hy
                    r = np.sqrt(dx*dx + dy*dy)
                    if r > INFLUENCE_RADIUS or r == 0:
                        continue
                    a = np.arctan2(dy, dx)
                    if abs((a - ang + np.pi) % (2*np.pi) - np.pi) < SPREAD_ANGLE/2:
                        gain = (HEATER_POWER * dt / C) * np.exp(-r/4)
                        Tn[j,i] += gain

        # 벽체 손실
        Tm = np.mean(Tn[mask])
        loss = wall_u * wall_area * (Tm - t_ext) * dt / C
        Tn[mask] -= loss

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
with c1:
    px = st.number_input("X 좌표", format="%.2f")
with c2:
    py = st.number_input("Y 좌표", format="%.2f")

if st.button("좌표 추가"):
    st.session_state.space.append((px, py))

if len(st.session_state.space) >= 1:
    xs, ys = zip(*st.session_state.space)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers"
    ))
    if len(xs) >= 3:
        fig.add_trace(go.Scatter(
            x=list(xs)+[xs[0]],
            y=list(ys)+[ys[0]],
            line=dict(dash="dot")
        ))
    fig.update_yaxes(scaleanchor="x")
    st.plotly_chart(fig, use_container_width=True)

# ---------- 2단계 ----------
st.header("2️⃣ 열풍기 설정")

heater_n = st.radio("열풍기 수량", [1, 2], horizontal=True)
heaters = []

for i in range(heater_n):
    st.subheader(f"열풍기 {i+1}")
    c1, c2, c3 = st.columns(3)
    with c1:
        hx = st.number_input("X", key=f"x{i}")
    with c2:
        hy = st.number_input("Y", key=f"y{i}")
    with c3:
        ang = np.deg2rad(st.slider("풍향", -180, 180, 0, key=f"a{i}"))
    heaters.append((hx, hy, ang))

# 미리보기
if len(st.session_state.space) >= 3:
    fig = go.Figure()
    xs, ys = zip(*(st.session_state.space+[st.session_state.space[0]]))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines"))

    for hx, hy, a in heaters:
        fig.add_trace(go.Scatter(
            x=[hx], y=[hy],
            marker=dict(size=14, symbol="triangle-up"),
            mode="markers"
        ))
        angles = np.linspace(a-SPREAD_ANGLE/2, a+SPREAD_ANGLE/2, 40)
        fx = [hx] + [hx + PREVIEW_RADIUS*np.cos(t) for t in angles] + [hx]
        fy = [hy] + [hy + PREVIEW_RADIUS*np.sin(t) for t in angles] + [hy]
        fig.add_trace(go.Scatter(
            x=fx, y=fy,
            fill="toself",
            opacity=0.3,
            showlegend=False
        ))
        L = PREVIEW_RADIUS*0.6
        fig.add_trace(go.Scatter(
            x=[hx, hx+L*np.cos(a)],
            y=[hy, hy+L*np.sin(a)],
            mode="lines"
        ))

    fig.update_yaxes(scaleanchor="x")
    st.plotly_chart(fig, use_container_width=True)

# ---------- 3단계 ----------
st.header("3️⃣ 시뮬레이션 설정")
t_init = st.number_input("초기 내부 온도 (°C)", value=10.0)
t_ext = st.number_input("외부 온도 (°C)", value=0.0)
wall = st.selectbox("벽체", list(WALL_U.keys()))
height = st.number_input("천장 높이 (m)", value=3.0)

if st.button("🔥 시뮬레이션 실행"):
    st.session_state.result = run_simulation(
        st.session_state.space,
        heaters,
        WALL_U[wall],
        height,
        t_init,
        t_ext
    )

# ---------- 결과 ----------
if "result" in st.session_state:
    T_hist, x, y, mask = st.session_state.result
    idx = st.slider("시간 (30분)", 0, len(T_hist)-1)
    fig = go.Figure(go.Heatmap(
        z=T_hist[idx],
        x=x, y=y,
        zmin=TEMP_MIN, zmax=TEMP_MAX,
        hovertemplate="X: %{x:.1f}<br>Y: %{y:.1f}<br>온도: %{z:.1f}°C"
    ))
    for hx, hy, _ in heaters:
        fig.add_trace(go.Scatter(
            x=[hx], y=[hy],
            marker=dict(size=12, symbol="triangle-up"),
            mode="markers"
        ))
    fig.update_yaxes(scaleanchor="x")
    st.plotly_chart(fig, use_container_width=True)
