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

DT = 60
SIM_HOURS = 9
ALPHA = 0.03
MIXING = 0.12   # 🔥 열대류/공기 혼합 강화

# 🔒 벽체는 샌드위치 판넬로 고정
WALL_U = 1.0   # W/m²K

# =========================
# 초기화
# =========================
def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]

# =========================
# 열 시뮬레이션
# =========================
def run_simulation(space_pts, heaters, height, init_temp, ext_temp):
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
    U = WALL_U

    steps = int(SIM_HOURS * 3600 / DT)

    for step in range(steps):
        Tn = T.copy()

        # 🔁 열 확산
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if mask[j, i]:
                    Tn[j, i] += ALPHA * (
                        T[j+1, i] + T[j-1, i] +
                        T[j, i+1] + T[j, i-1] -
                        4 * T[j, i]
                    )

        # 🔥 열풍기 영향
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

                    proj = dx * ca + dy * sa
                    if proj <= 0:
                        continue

                    w = np.exp(-r / 3) * (proj / r)
                    Tn[j, i] += (HEATER_WATT * DT / C) * w

        # 🌪️ 공기 혼합 (열대류)
        T_mean = np.mean(Tn[mask])
        Tn[mask] += MIXING * (T_mean - Tn[mask])

        # 🧱 벽체 열손실 (샌드위치 판넬 고정)
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
px = c1.number_input("X 좌표 (m)", format="%.3f")
py = c2.number_input("Y 좌표 (m)", format="%.3f")

if st.button("좌표 추가"):
    st.session_state.space.append((px, py))
    st.rerun()

if len(st.session_state.space) >= 1:
    fig = go.Figure()
    xs, ys = zip(*st.session_state.space)

    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines+markers",
        line=dict(width=2),
        marker=dict(size=6)
    ))

    if len(st.session_state.space) >= 3:
        fig.add_trace(go.Scatter(
            x=list(xs) + [xs[0]],
            y=list(ys) + [ys[0]],
            mode="lines",
            line=dict(dash="dot")
        ))

    fig.update_yaxes(scaleanchor="x")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ---------- 2단계 ----------
st.header("2️⃣ 열풍기 배치")

heater_n = st.radio("열풍기 수량", [1, 2], horizontal=True)
heaters = []

for i in range(heater_n):
    st.markdown(f"### 🔥 열풍기 {i+1}")
    c1, c2, c3 = st.columns(3)
    hx = c1.number_input("X (m)", format="%.3f", key=f"hx{i}")
    hy = c2.number_input("Y (m)", format="%.3f", key=f"hy{i}")
    ang = c3.slider("풍향 (°)", -180, 180, 20, key=f"ang{i}")
    heaters.append({"x": hx, "y": hy, "angle": np.deg2rad(ang)})

# 🔍 배치 미리보기 
if len(st.session_state.space) >= 3:
    fig = go.Figure()
    xs, ys = zip(*(st.session_state.space + [st.session_state.space[0]]))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="black")))

    for h in heaters:
        hx, hy, a = h["x"], h["y"], h["angle"]

        fig.add_trace(go.Scatter(
            x=[hx], y=[hy],
            mode="markers",
            marker=dict(size=18, color="red", symbol="triangle-up")
        ))

        spread = np.deg2rad(20)
        r = INFLUENCE_RADIUS * 0.3
        theta = np.linspace(a - spread, a + spread, 30)

        fx = [hx] + list(hx + r * np.cos(theta)) + [hx]
        fy = [hy] + list(hy + r * np.sin(theta)) + [hy]

        fig.add_trace(go.Scatter(
            x=fx, y=fy,
            fill="toself",
            mode="lines",
            fillcolor="rgba(255,0,0,0.2)",
            line=dict(color="rgba(255,0,0,0.4)"),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[hx, hx + r * np.cos(a)],
            y=[hy, hy + r * np.sin(a)],
            mode="lines",
            line=dict(width=3, color="orange")
        ))

    fig.update_yaxes(scaleanchor="x")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

# ---------- 3단계 ----------
st.header("3️⃣ 시뮬레이션 설정")

height = st.number_input("천장 높이 (m)", value=3.0)

c1, c2 = st.columns(2)
init_temp = c1.number_input("시작 내부 온도 (°C)", value=10.0)
ext_temp = c2.number_input("외부 온도 (°C)", value=0.0)

if st.button("🔥 시뮬레이션 실행"):
    st.session_state.result = run_simulation(
        st.session_state.space,
        heaters,
        height,
        init_temp,
        ext_temp
    )

# ---------- 결과 ----------
if "result" in st.session_state:
    T_hist, x, y, mask = st.session_state.result

    frames = []
    for T in T_hist:
        frames.append(go.Frame(
            data=[go.Heatmap(
                z=T,
                x=x, y=y,
                zmin=TEMP_MIN, zmax=TEMP_MAX,
                colorscale="Turbo",
                hovertemplate=(
                    "X: %{x:.1f} m<br>"
                    "Y: %{y:.1f} m<br>"
                    "온도: %{z:.1f} °C"
                )
            )]
        ))

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_yaxes(scaleanchor="x")
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "buttons": [{
                "label": "▶ 재생",
                "method": "animate",
                "args": [None, {"frame": {"duration": 300}}]
            }]
        }]
    )

    st.plotly_chart(fig, use_container_width=True)

    rows = []
    for t, T in enumerate(T_hist):
        for i in range(len(x)):
            for j in range(len(y)):
                if mask[j, i]:
                    rows.append([t * 0.5, x[i], y[j], T[j, i]])

    df = pd.DataFrame(rows, columns=["시간(h)", "X(m)", "Y(m)", "온도(°C)"])
    csv = df.to_csv(index=False).encode()
    st.download_button("📥 CSV 다운로드", csv, "heat_simulation.csv")
