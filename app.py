import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# =====================================
# 기본 설정
# =====================================
st.set_page_config(layout="wide")
st.title("🔥 좌표 입력 기반 공간 열전달 시뮬레이션")

# =====================================
# 세션 상태
# =====================================
if "space_points" not in st.session_state:
    st.session_state.space_points = [(0, 0)]

if "space_closed" not in st.session_state:
    st.session_state.space_closed = False

if "heater_points" not in st.session_state:
    st.session_state.heater_points = []

# =====================================
# 사이드바
# =====================================
st.sidebar.header("환경 조건")

outside_temp = st.sidebar.number_input("외부 온도 (°C)", value=0.0)
inside_temp = st.sidebar.number_input("초기 내부 온도 (°C)", value=10.0)
heater_count = st.sidebar.selectbox("열풍기 개수", [1, 2])

if st.sidebar.button("❌ 전체 초기화"):
    st.session_state.space_points = [(0, 0)]
    st.session_state.space_closed = False
    st.session_state.heater_points = []
    st.rerun()

run_btn = st.sidebar.button("▶ 시뮬레이션 실행")

# =====================================
# 공간 좌표 입력
# =====================================
st.subheader("🧱 1단계: 내부공간 좌표 입력 (기준점: 0,0)")

if not st.session_state.space_closed:
    col1, col2 = st.columns(2)
    with col1:
        x = st.number_input("X 좌표", value=0.0)
    with col2:
        y = st.number_input("Y 좌표", value=0.0)

    if st.button("➕ 선 추가"):
        st.session_state.space_points.append((x, y))
        st.rerun()

    if len(st.session_state.space_points) >= 3:
        if st.button("✅ 공간 완성 (0,0으로 닫기)"):
            st.session_state.space_points.append((0, 0))
            st.session_state.space_closed = True
            st.rerun()

# =====================================
# 공간 시각화
# =====================================
NX, NY = 100, 60

fig = go.Figure()

xs, ys = zip(*st.session_state.space_points)

fig.add_trace(
    go.Scatter(
        x=xs,
        y=ys,
        mode="lines+markers",
        line=dict(color="blue", width=3),
        marker=dict(size=8),
        name="내부 공간"
    )
)

if st.session_state.heater_points:
    hx, hy = zip(*st.session_state.heater_points)
    fig.add_trace(
        go.Scatter(
            x=hx,
            y=hy,
            mode="markers",
            marker=dict(color="red", size=14),
            name="열풍기"
        )
    )

fig.update_layout(
    width=700,
    height=400,
    dragmode=False,
    clickmode="event",
    xaxis=dict(range=[-1, NX], fixedrange=True),
    yaxis=dict(range=[-1, NY], fixedrange=True),
    title="공간 및 열풍기 배치"
)

st.plotly_chart(fig, use_container_width=True)

# =====================================
# 공간 내부 판별
# =====================================
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

# =====================================
# 2단계: 열풍기 배치
# =====================================
if st.session_state.space_closed:
    st.subheader("🔥 2단계: 공간 내부에 열풍기 배치")

    clicked = plotly_events(fig, click_event=True)

    if clicked:
        hx = int(clicked[0]["x"])
        hy = int(clicked[0]["y"])

        if point_in_polygon(hx, hy, st.session_state.space_points):
            if len(st.session_state.heater_points) < heater_count:
                st.session_state.heater_points.append((hx, hy))
                st.rerun()
        else:
            st.warning("열풍기는 내부 공간에만 배치할 수 있습니다.")
