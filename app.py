import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.path import Path

st.set_page_config(layout="wide")

# =========================
# 상수
# =========================
TEMP_MIN, TEMP_MAX = -10, 40
HEATER_KCAL = 17600
HEATER_WATT = HEATER_KCAL * 1.163
INFLUENCE_RADIUS = 10.0
DISPLAY_RADIUS = INFLUENCE_RADIUS * 0.3  # 🔥 시각화용 30%

DT = 60
SIM_HOURS = 9
ALPHA = 0.03
MIXING = 0.12

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
# 자동 배치 알고리즘
# =========================
def auto_place_heaters(space_pts, heater_n):
    pts = np.array(space_pts)
    center = pts.mean(axis=0)

    heaters = []

    if heater_n == 1:
        pos = center
        angle = np.arctan2(center[1]-pos[1], center[0]-pos[0])
        heaters.append({"x": pos[0], "y": pos[1], "angle": angle})

    else:
        offset = (pts.max(axis=0) - pts.min(axis=0)) * 0.2
        p1 = center - offset
        p2 = center + offset

        for p in [p1, p2]:
            angle = np.arctan2(center[1]-p[1], center[0]-p[0])
            heaters.append({"x": p[0], "y": p[1], "angle": angle})

    return heaters

# =========================
# 열 시뮬레이션
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
                        T[j+1,i] + T[j-1,i] +
                        T[j,i+1] + T[j,i-1] -
                        4*T[j,i]
                    )

        # 열풍기
        for h in heaters:
            hx, hy, angle = h["x"], h["y"], h["angle"]
            ca, sa = np.cos(angle), np.sin(angle)

            for i in range(nx):
                for j in range(ny):
                    if not mask[j, i]:
                        continue
                    dx = X[j,i] - hx
                    dy = Y[j,i] - hy
                    r = np.hypot(dx, dy)
                    if r == 0 or r > INFLUENCE_RADIUS:
                        continue
                    proj = dx*ca + dy*sa
                    if proj <= 0:
                        continue
                    w = np.exp(-r/3) * (proj / r)
                    Tn[j,i] += (HEATER_WATT * DT / C) * w

        # 공기 혼합
        T_mean = np.mean(Tn[mask])
        Tn[mask] += MIXING * (T_mean - Tn[mask])

        # 벽체 손실
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
    st.rerun()

# 공간 미리보기
if len(st.session_state.space) >= 1:
    fig = go.Figure()
    xs, ys = zip(*st.session_state.space)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers"))
    if len(xs) >= 3:
        fig.add_trace(go.Scatter(x=list(xs)+[xs[0]], y=list(ys)+[ys[0]], mode="lines", line=dict(dash="dot")))
    fig.update_yaxes(scaleanchor="x")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# ---------- 2단계 ----------
st.header("2️⃣ 열풍기 배치")
heater_n = st.radio("열풍기 수량", [1, 2], horizontal=True)

manual_heaters = []
for i in range(heater_n):
    c1, c2, c3 = st.columns(3)
    hx = c1.number_input("X (m)", key=f"hx{i}", format="%.3f")
    hy = c2.number_input("Y (m)", key=f"hy{i}", format="%.3f")
    ang = c3.slider("풍향 (°)", -180, 180, 20, key=f"ang{i}")
    manual_heaters.append({"x": hx, "y": hy, "angle": np.deg2rad(ang)})

auto_heaters = auto_place_heaters(st.session_state.space, heater_n)

# ---------- 배치 비교 ----------
if len(st.session_state.space) >= 3:
    st.subheader("🔥 열풍기 배치 비교")

    colA, colB = st.columns(2)

    def draw_layout(title, heaters):
        fig = go.Figure()
        xs, ys = zip(*(st.session_state.space + [st.session_state.space[0]]))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="black")))

        for h in heaters:
            hx, hy, a = h["x"], h["y"], h["angle"]

            # 열풍기 아이콘
            fig.add_trace(go.Scatter(
                x=[hx], y=[hy],
                mode="markers",
                marker=dict(size=16, symbol="triangle-up", color="red")
            ))

            # 🔥 부채꼴
            theta = np.linspace(-np.pi/6, np.pi/6, 30)
            fx = hx + DISPLAY_RADIUS * np.cos(theta + a)
            fy = hy + DISPLAY_RADIUS * np.sin(theta + a)

            fig.add_trace(go.Scatter(
                x=[hx]+list(fx)+[hx],
                y=[hy]+list(fy)+[hy],
                fill="toself",
                fillcolor="rgba(255,0,0,0.2)",
                line=dict(color="rgba(255,0,0,0.3)"),
                showlegend=False
            ))

        fig.update_yaxes(scaleanchor="x")
        fig.update_layout(title=title, height=350)
        return fig

    colA.plotly_chart(draw_layout("🔧 수동 배치", manual_heaters), use_container_width=True)
    colB.plotly_chart(draw_layout("🤖 자동 배치", auto_heaters), use_container_width=True)
