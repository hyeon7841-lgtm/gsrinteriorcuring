import streamlit as st
import numpy as np
import plotly.graph_objects as go
from matplotlib.path import Path

st.set_page_config(layout="wide")

# =========================
# 상수
# =========================
INIT_TEMP = 10.0
T_EXT = 0.0
TEMP_MIN, TEMP_MAX = -10, 40

HEATER_KCAL = 17600
HEATER_WATT = HEATER_KCAL * 1.163
INFLUENCE_RADIUS = 10.0  # 🔒 고정
DT = 60
SIM_HOURS = 9
ALPHA = 0.03

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
# 열 시뮬레이션 (이동 열원)
# =========================
def run_simulation(space_pts, heaters, wall_type, height):
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

    T = np.full_like(X, INIT_TEMP)
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

        # 열풍기 (이동 열원)
        for h in heaters:
            hx, hy, angle = h["x"], h["y"], h["angle"]
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            for i in range(nx):
                for j in range(ny):
                    if not mask[j, i]:
                        continue

                    dx = X[j, i] - hx
                    dy = Y[j, i] - hy
                    r = np.hypot(dx, dy)

                    if r == 0 or r > INFLUENCE_RADIUS:
                        continue

                    proj = dx*cos_a + dy*sin_a
                    if proj <= 0:
                        continue

                    weight = np.exp(-r/3) * (proj / r)
                    dT = (HEATER_WATT * DT / C) * weight
                    Tn[j, i] += dT

        # 벽체 손실
        T_mean = np.mean(Tn[mask])
        Q_loss = U * wall_area * (T_mean - T_EXT) * DT
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
with c1:
    px = st.number_input("X 좌표 (m)", format="%.3f")
with c2:
    py = st.number_input("Y 좌표 (m)", format="%.3f")

if st.button("좌표 추가"):
    st.session_state.space.append((px, py))

if len(st.session_state.space) >= 1:
    st.subheader("📐 공간 미리보기")
    xs, ys = zip(*(st.session_state.space + [st.session_state.space[0]]))
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers"))
    fig.update_yaxes(scaleanchor="x")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# ---------- 2단계 ----------
st.header("2️⃣ 열풍기 배치")

heater_n = st.radio("열풍기 수량", [1, 2], horizontal=True)
heaters = []

for i in range(heater_n):
    st.markdown(f"### 🔥 열풍기 {i+1}")
    c1, c2, c3 = st.columns(3)

    with c1:
        hx = st.number_input("X (m)", key=f"hx{i}", format="%.3f")
    with c2:
        hy = st.number_input("Y (m)", key=f"hy{i}", format="%.3f")
    with c3:
        ang = st.slider("풍향 (°)", -180, 180, 20, key=f"ang{i}")

    heaters.append({
        "x": hx,
        "y": hy,
        "angle": np.deg2rad(ang)
    })

# ---------- 미리보기 ----------
if len(st.session_state.space) >= 3:
    st.subheader("🔥 열풍기 배치 미리보기")
    fig = go.Figure()

    xs, ys = zip(*(st.session_state.space + [st.session_state.space[0]]))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="black")))

    for h in heaters:
        hx, hy, ang = h["x"], h["y"], h["angle"]

        fig.add_trace(go.Scatter(
            x=[hx], y=[hy],
            mode="markers",
            marker=dict(size=14, color="red", symbol="triangle-up")
        ))

        theta = np.linspace(0, 2*np.pi, 80)
        fig.add_trace(go.Scatter(
            x=hx + INFLUENCE_RADIUS*np.cos(theta),
            y=hy + INFLUENCE_RADIUS*np.sin(theta),
            fill="toself",
            fillcolor="rgba(255,0,0,0.12)",
            line=dict(color="rgba(255,0,0,0.25)"),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[hx, hx + INFLUENCE_RADIUS*np.cos(ang)],
            y=[hy, hy + INFLUENCE_RADIUS*np.sin(ang)],
            mode="lines",
            line=dict(width=3, color="orange"),
            showlegend=False
        ))

    fig.update_yaxes(scaleanchor="x")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

# ---------- 3단계 ----------
st.header("3️⃣ 시뮬레이션")
wall = st.selectbox("벽체 재질", list(WALL_U.keys()))
height = st.number_input("천장 높이 (m)", value=3.0)

if st.button("🔥 시뮬레이션 실행"):
    with st.spinner("계산 중..."):
        st.session_state.result = run_simulation(
            st.session_state.space, heaters, wall, height
        )

# ---------- 결과 ----------
if "result" in st.session_state:
    T_hist, x, y, mask = st.session_state.result
    idx = st.slider("시간 (30분 간격)", 0, len(T_hist)-1)

    fig = go.Figure(go.Heatmap(
        z=T_hist[idx], x=x, y=y,
        zmin=TEMP_MIN, zmax=TEMP_MAX,
        colorscale="Turbo"
    ))

    for h in heaters:
        fig.add_trace(go.Scatter(
            x=[h["x"]], y=[h["y"]],
            mode="markers",
            marker=dict(size=12, color="red", symbol="triangle-up")
        ))

    fig.update_yaxes(scaleanchor="x")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
