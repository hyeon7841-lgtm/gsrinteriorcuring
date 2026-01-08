import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.path import Path

st.set_page_config(layout="wide")

# =========================
# 기본 설정
# =========================
INIT_TEMP = 10.0
T_EXT = 0.0
TEMP_MIN, TEMP_MAX = -10, 40

HEATER_KCAL = 17600
HEATER_WATT = HEATER_KCAL * 1.163  # kcal/h → W
HEATER_ANGLE = np.deg2rad(20)

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
# 열 시뮬레이션
# =========================
def run_simulation(space_pts, heaters, wall_type, height):
    pts = np.array(space_pts)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)

    nx, ny = 60, 60
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    poly = Path(space_pts)
    mask = poly.contains_points(
        np.vstack((X.flatten(), Y.flatten())).T
    ).reshape(X.shape)

    T = np.full_like(X, INIT_TEMP)
    T_hist = []

    dx = (xmax - xmin) / nx
    area = (xmax - xmin) * (ymax - ymin)
    perimeter = 2 * ((xmax - xmin) + (ymax - ymin))
    wall_area = perimeter * height

    rho = 1.2
    cp = 1000
    V = area * height
    C = rho * cp * V

    dt = 60
    steps = int(9 * 3600 / dt)
    alpha = 0.12
    U = WALL_U[wall_type]

    for step in range(steps):
        Tn = T.copy()

        # 확산
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if mask[j, i]:
                    Tn[j, i] += alpha * (
                        T[j+1,i] + T[j-1,i] + T[j,i+1] + T[j,i-1] - 4*T[j,i]
                    )

        # 열풍기 (부채꼴)
        for hx, hy in heaters:
            for i in range(nx):
                for j in range(ny):
                    if not mask[j, i]:
                        continue
                    dxh = X[j, i] - hx
                    dyh = Y[j, i] - hy
                    r = np.sqrt(dxh**2 + dyh**2)
                    if r == 0 or r > 3:
                        continue
                    angle = np.arctan2(dyh, dxh)
                    if abs(angle) <= HEATER_ANGLE:
                        gain = (HEATER_WATT * dt / C) * np.exp(-r)
                        Tn[j, i] += gain

        # 벽체 열손실
        T_mean = np.mean(Tn[mask])
        Q_loss = U * wall_area * (T_mean - T_EXT) * dt
        dT_loss = Q_loss / C
        Tn[mask] -= dT_loss

        Tn = np.clip(Tn, TEMP_MIN, TEMP_MAX)
        T = Tn

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
st.header("1️⃣ 공간 정의 (m)")
if "space" not in st.session_state:
    st.session_state.space = [(0.0, 0.0)]

col1, col2 = st.columns(2)
with col1:
    x = st.number_input("X 좌표", format="%.3f")
with col2:
    y = st.number_input("Y 좌표", format="%.3f")

if st.button("좌표 추가"):
    st.session_state.space.append((x, y))

st.subheader("📐 현재 공간 미리보기")

if len(st.session_state.space) >= 1:
    fig = go.Figure()

    xs, ys = zip(*st.session_state.space)

    # 점 & 진행 중인 선
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name="공간 경계",
            line=dict(width=2),
            marker=dict(size=6)
        )
    )

    # 3개 이상이면 닫힌 폴리곤도 표시
    if len(st.session_state.space) >= 3:
        fig.add_trace(
            go.Scatter(
                x=list(xs) + [xs[0]],
                y=list(ys) + [ys[0]],
                mode="lines",
                line=dict(dash="dot"),
                name="완성 예상"
            )
        )

    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    fig.update_yaxes(scaleanchor="x")  # ✅ 1:1 비율 유지

    st.plotly_chart(fig, use_container_width=True)


# ---------- 2단계 ----------
st.subheader("🌬️ 열풍기 풍향 설정")

wind_angle_deg = st.slider(
    "풍향 각도 (도, +X축 기준)",
    min_value=-180,
    max_value=180,
    value=20,
    step=1
)
wind_angle = np.deg2rad(wind_angle_deg)
influence_radius = st.slider(
    "열풍기 영향 반경 표시 (m)",
    min_value=0.5,
    max_value=5.0,
    value=2.5,
    step=0.1
)

st.header("2️⃣ 열풍기 배치 (m)")

heater_n = st.radio("열풍기 수량", [1, 2], horizontal=True)

heaters = []
for i in range(heater_n):
    col1, col2 = st.columns(2)
    with col1:
        hx = st.number_input(
            f"열풍기 {i+1} X 좌표 (m)",
            format="%.3f",
            key=f"heater_x_{i}"
        )
    with col2:
        hy = st.number_input(
            f"열풍기 {i+1} Y 좌표 (m)",
            format="%.3f",
            key=f"heater_y_{i}"
        )
    heaters.append((hx, hy))

# ---------- 미리보기 ----------
st.subheader("🔥 열풍기 배치 미리보기")

fig = go.Figure()

# --- 공간 경계 ---
xs, ys = zip(*(st.session_state.space + [st.session_state.space[0]]))
fig.add_trace(
    go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(width=2, color="black"),
        name="공간"
    )
)

# --- 열풍기 표시 ---
for i, (hx, hy) in enumerate(heaters):
    # 열풍기 위치
    fig.add_trace(
        go.Scatter(
            x=[hx],
            y=[hy],
            mode="markers",
            marker=dict(
                size=14,
                color="red",
                symbol="triangle-up"
            ),
            name=f"열풍기 {i+1}"
        )
    )

    # 🔥 영향 반경 (중첩 시각화 핵심)
    theta = np.linspace(0, 2*np.pi, 60)
    rx = hx + influence_radius * np.cos(theta)
    ry = hy + influence_radius * np.sin(theta)

    fig.add_trace(
        go.Scatter(
            x=rx,
            y=ry,
            fill="toself",
            mode="lines",
            line=dict(color="rgba(255,0,0,0.2)"),
            fillcolor="rgba(255,0,0,0.15)",
            showlegend=False
        )
    )

    # 🌬️ 풍향 벡터
    L = influence_radius
    dx = L * np.cos(wind_angle)
    dy = L * np.sin(wind_angle)

    fig.add_trace(
        go.Scatter(
            x=[hx, hx + dx],
            y=[hy, hy + dy],
            mode="lines",
            line=dict(width=3, color="orange"),
            showlegend=False
        )
    )

fig.update_layout(
    height=480,
    margin=dict(l=20, r=20, t=20, b=20),
    showlegend=False
)
fig.update_yaxes(scaleanchor="x")

st.plotly_chart(fig, use_container_width=True)


# ---------- 3단계 ----------
st.header("3️⃣ 시뮬레이션 설정")
wall = st.selectbox("벽체 재질", list(WALL_U.keys()))
height = st.number_input("천장 높이 (m)", value=3.0)

if st.button("🔥 시뮬레이션 실행"):
    with st.spinner("계산 중..."):
        result = run_simulation(
            st.session_state.space, heaters, wall, height
        )
        st.session_state.result = result

# ---------- 결과 ----------
if "result" in st.session_state:
    T_hist, x, y, mask = st.session_state.result
    idx = st.slider("시간 (30분 간격)", 0, len(T_hist)-1)

    fig = go.Figure(
        data=go.Heatmap(
            z=T_hist[idx],
            x=x, y=y,
            zmin=TEMP_MIN, zmax=TEMP_MAX,
            colorscale="Turbo"
        )
    )

    for hx, hy in heaters:
        fig.add_trace(go.Scatter(
            x=[hx], y=[hy],
            mode="markers",
            marker=dict(size=12, color="red", symbol="triangle-up")
        ))

    fig.update_yaxes(scaleanchor="x")
    st.plotly_chart(fig, use_container_width=True)
