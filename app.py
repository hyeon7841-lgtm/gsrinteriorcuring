import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import numpy as np

# ======================================================
# 기본 설정
# ======================================================
st.set_page_config(layout="wide")
st.title("🔥 내부공간 열풍기 배치 및 온도 시각화 시뮬레이터 (v1)")

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

# ======================================================
# 사이드바
# ======================================================
st.sidebar.header("설정")

heater_count = st.sidebar.selectbox("열풍기 개수", [1, 2])

if st.sidebar.button("❌ 전체 초기화"):
    st.session_state.space_points = [(0.0, 0.0)]
    st.session_state.space_closed = False
    st.session_state.heater_points = []
    st.session_state.temp_heater = None
    st.rerun()

# ======================================================
# 1단계: 내부공간 정의
# ======================================================
st.subheader("🧱 1단계: 내부공간 정의 (기준점: 0,0)")

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
            if st.button("⬅ 이전 단계로 되돌리기"):
                st.session_state.space_points.pop()
                st.rerun()

        if len(st.session_state.space_points) >= 3:
            if st.button("✅ 공간 완성 (0,0으로 닫기)"):
                st.session_state.space_points.append((0.0, 0.0))
                st.session_state.space_closed = True
                st.rerun()

# ======================================================
# 공간 및 열풍기 배치 시각화
# ======================================================
fig = go.Figure()

if len(st.session_state.space_points) >= 1:
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
    width=750,
    height=450,
    dragmode=False,
    clickmode="event",
    xaxis=dict(title="X (m)", fixedrange=True),
    yaxis=dict(
        title="Y (m)",
        fixedrange=True,
        scaleanchor="x",
        scaleratio=1
    ),
    title="공간 정의 및 열풍기 배치"
)

st.plotly_chart(fig, use_container_width=True)

# ======================================================
# 공간 내부 판별 함수
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

    colu1, _ = st.columns([1, 5])
    with colu1:
        if st.button("⬅ 이전 열풍기 되돌리기"):
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

        col1, col2, col3, col4 = st.columns([1, 1, 2, 2])
        with col1:
            hx = st.number_input("열풍기 X 좌표", value=float(hx), step=0.1)
        with col2:
            hy = st.number_input("열풍기 Y 좌표", value=float(hy), step=0.1)

        with col3:
            if st.button("🔥 위치 확정"):
                if point_in_polygon(hx, hy, st.session_state.space_points):
                    if len(st.session_state.heater_points) < heater_count:
                        st.session_state.heater_points.append((hx, hy))
                        st.session_state.temp_heater = None
                        st.rerun()
                else:
                    st.warning("공간 내부에만 배치할 수 있습니다.")

        with col4:
            if st.button("❌ 임시 위치 취소"):
                st.session_state.temp_heater = None
                st.rerun()

# ======================================================
# 3단계: 온도 분포 시각화 (v1)
# ======================================================
if st.session_state.heater_points:
    st.subheader("🌡️ 3단계: 온도 분포 시각화")

    time_hour = st.slider("경과 시간 (시간)", 0, 9, 1)

    xs, ys = zip(*st.session_state.space_points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    nx, ny = 80, 50
    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    X, Y = np.meshgrid(x, y)

    mask = np.zeros_like(X, dtype=bool)
    for i in range(nx):
        for j in range(ny):
            mask[j, i] = point_in_polygon(
                X[j, i], Y[j, i],
                st.session_state.space_points
            )

    T = np.ones_like(X) * 10.0  # 초기온도

    for hx, hy in st.session_state.heater_points:
        dist = np.sqrt((X - hx)**2 + (Y - hy)**2)
        T += 18 * np.exp(-dist / 2.5) * (time_hour / 9)

    T[~mask] = np.nan

    fig2 = go.Figure(
        data=go.Heatmap(
            z=T,
            x=x,
            y=y,
            colorscale="Turbo",
            colorbar=dict(title="온도 (°C)")
        )
    )

    fig2.update_layout(
        width=750,
        height=450,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        title=f"{time_hour}시간 경과 후 온도 분포"
    )

    st.plotly_chart(fig2, use_container_width=True)
