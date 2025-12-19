import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import numpy as np

# ======================================================
# 기본 설정
# ======================================================
st.set_page_config(layout="wide")
st.title("🔥 내부공간 열풍기 배치 및 열해석 시뮬레이터 (산업용 v1)")

# ======================================================
# 세션 상태
# ======================================================
if "space_points" not in st.session_state:
    st.session_state.space_points = [(0.0, 0.0)]

if "space_closed" not in st.session_state:
    st.session_state.space_closed = False

if "heater_points" not in st.session_state:
    st.session_state.heater_points = []

if "temp_heater" not in st.session_state:
    st.session_state.temp_heater = None

if "heat_result" not in st.session_state:
    st.session_state.heat_result = None  # (T_history, x, y, mask)

# ======================================================
# 사이드바
# ======================================================
st.sidebar.header("설정")

heater_count = st.sidebar.selectbox("열풍기 개수", [1, 2])

inside_temp = st.sidebar.number_input("초기 내부온도 (°C)", value=10.0)
total_hours = 9

if st.sidebar.button("❌ 전체 초기화"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# ======================================================
# 1단계: 내부공간 정의
# ======================================================
st.subheader("🧱 1단계: 내부공간 정의")

if not st.session_state.space_closed:
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        x = st.number_input("X 좌표", value=0.0, step=0.5)
    with col2:
        y = st.number_input("Y 좌표", value=0.0, step=0.5)
    with col3:
        if st.button("➕ 선 추가"):
            st.session_state.space_points.append((x, y))
            st.rerun()
        if len(st.session_state.space_points) > 1:
            if st.button("⬅ 이전 단계"):
                st.session_state.space_points.pop()
                st.rerun()
        if len(st.session_state.space_points) >= 3:
            if st.button("✅ 공간 완성"):
                st.session_state.space_points.append((0.0, 0.0))
                st.session_state.space_closed = True
                st.rerun()

# ======================================================
# 공간 시각화
# ======================================================
fig = go.Figure()

if st.session_state.space_points:
    xs, ys = zip(*st.session_state.space_points)
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines+markers",
        line=dict(color="blue", width=3),
        marker=dict(size=8),
        name="내부 공간"
    ))

if st.session_state.heater_points:
    hx, hy = zip(*st.session_state.heater_points)
    fig.add_trace(go.Scatter(
        x=hx, y=hy,
        mode="markers",
        marker=dict(color="red", size=14),
        name="열풍기"
    ))

fig.update_layout(
    width=750, height=450,
    dragmode=False,
    clickmode="event",
    xaxis=dict(fixedrange=True),
    yaxis=dict(scaleanchor="x", scaleratio=1, fixedrange=True),
    title="공간 및 열풍기 배치"
)

st.plotly_chart(fig, use_container_width=True)

# ======================================================
# 공간 내부 판별
# ======================================================
def point_in_polygon(x, y, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside

# ======================================================
# 2단계: 열풍기 배치
# ======================================================
if st.session_state.space_closed:
    st.subheader("🔥 2단계: 열풍기 배치")

    if st.button("⬅ 이전 열풍기"):
        if st.session_state.heater_points:
            st.session_state.heater_points.pop()
            st.rerun()

    clicked = plotly_events(fig, click_event=True)
    if clicked:
        st.session_state.temp_heater = (
            float(clicked[0]["x"]),
            float(clicked[0]["y"])
        )

    if st.session_state.temp_heater:
        hx, hy = st.session_state.temp_heater
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            hx = st.number_input("X", value=float(hx), step=0.1)
        with col2:
            hy = st.number_input("Y", value=float(hy), step=0.1)
        with col3:
            if st.button("🔥 위치 확정"):
                if point_in_polygon(hx, hy, st.session_state.space_points):
                    if len(st.session_state.heater_points) < heater_count:
                        st.session_state.heater_points.append((hx, hy))
                        st.session_state.temp_heater = None
                        st.session_state.heat_result = None
                        st.rerun()

# ======================================================
# 🔥 실제 열해석 엔진
# ======================================================
def run_heat_simulation(space_points, heater_points, inside_temp, total_hours):
    alpha = 1.0e-6
    rho, cp = 2400, 900
    heater_power = 20461  # W

    xs, ys = zip(*space_points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    nx, ny = 60, 40
    dx = (max_x - min_x) / nx
    dy = (max_y - min_y) / ny
    dt = 3600

    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    X, Y = np.meshgrid(x, y)

    mask = np.zeros((ny, nx), dtype=bool)
    for i in range(nx):
        for j in range(ny):
            mask[j, i] = point_in_polygon(X[j, i], Y[j, i], space_points)

    T = np.ones((ny, nx)) * inside_temp
    history = [T.copy()]

    for _ in range(total_hours):
        Tn = T.copy()
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                if not mask[j, i]:
                    continue
                lap = (
                    (T[j, i+1] - 2*T[j, i] + T[j, i-1]) / dx**2 +
                    (T[j+1, i] - 2*T[j, i] + T[j-1, i]) / dy**2
                )
                Tn[j, i] += alpha * dt * lap

        for hx, hy in heater_points:
            ix = np.argmin(np.abs(x - hx))
            iy = np.argmin(np.abs(y - hy))
            if mask[iy, ix]:
                Tn[iy, ix] += heater_power * dt / (rho * cp * dx * dy)

        T = Tn
        history.append(T.copy())

    return history, x, y, mask

# ======================================================
# 3단계: 계산 실행 + 시각화
# ======================================================
if st.session_state.heater_points:
    st.subheader("🌡️ 3단계: 열해석 결과")

    if st.button("🧮 열해석 계산 실행"):
        with st.spinner("열해석 계산 중..."):
            st.session_state.heat_result = run_heat_simulation(
                st.session_state.space_points,
                st.session_state.heater_points,
                inside_temp,
                total_hours
            )

    if st.session_state.heat_result:
        T_hist, x, y, mask = st.session_state.heat_result
        t = st.slider("시간 (h)", 0, total_hours, 0)

        T = T_hist[t]
        T[~mask] = np.nan

        figT = go.Figure(
            data=go.Heatmap(
                z=T, x=x, y=y,
                colorscale="Turbo",
                colorbar=dict(title="온도 (°C)")
            )
        )
        figT.update_layout(
            width=750, height=450,
            yaxis=dict(scaleanchor="x", scaleratio=1),
            title=f"{t}시간 후 온도 분포"
        )
        st.plotly_chart(figT, use_container_width=True)
