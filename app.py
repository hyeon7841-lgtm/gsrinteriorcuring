import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# ======================================================
# 기본 설정
# ======================================================
st.set_page_config(layout="wide")
st.title("🔥 내부공간 열풍기 난방 시뮬레이터")

# ======================================================
# 세션 상태 초기화
# ======================================================
def reset_all():
    st.session_state.step = 1
    st.session_state.space_points = [(0.0, 0.0)]
    st.session_state.space_closed = False
    st.session_state.heater_points = []
    st.session_state.heat_result = None

if "step" not in st.session_state:
    reset_all()

# ======================================================
# 🔁 전체 초기화 버튼 (항상 표시)
# ======================================================
st.sidebar.header("공통 설정")

if st.sidebar.button("🔄 전체 초기화"):
    reset_all()
    st.rerun()

# ======================================================
# 사이드바 입력
# ======================================================
heater_count = st.sidebar.selectbox("열풍기 개수", [1, 2])

ceiling_height = st.sidebar.number_input(
    "천장 높이 (m)", 2.0, 15.0, 4.0, step=0.1
)

wall_type = st.sidebar.selectbox(
    "벽체 재질",
    ["조적벽", "콘크리트벽", "샌드위치판넬"]
)

U_map = {
    "조적벽": 1.2,
    "콘크리트벽": 1.7,
    "샌드위치판넬": 0.25
}
U = U_map[wall_type]

T_inside0 = 10.0
T_outside = -5.0

# ======================================================
# 내부 판별 함수
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
# 1단계: 공간 정의
# ======================================================
if st.session_state.step == 1:
    st.subheader("🧱 1단계: 내부공간 정의 (단위: m)")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        x = st.number_input("X 좌표", step=0.001, format="%.3f")
    with c2:
        y = st.number_input("Y 좌표", step=0.001, format="%.3f")
    with c3:
        if st.button("➕ 선 추가"):
            st.session_state.space_points.append((x, y))
            st.rerun()

        if len(st.session_state.space_points) > 1:
            if st.button("⬅ 이전 점 삭제"):
                st.session_state.space_points.pop()
                st.rerun()

        if len(st.session_state.space_points) >= 3:
            if st.button("✅ 공간 완성"):
                st.session_state.space_points.append((0.0, 0.0))
                st.session_state.space_closed = True
                st.session_state.step = 2
                st.rerun()

    xs, ys = zip(*st.session_state.space_points)
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers"))
    fig.update_layout(
        height=400,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        title="내부공간 형상"
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# 2단계: 열풍기 좌표 입력 + 임시 시각화
# ======================================================
if st.session_state.step == 2:
    st.subheader("🔥 2단계: 열풍기 좌표 입력 (단위: m)")

    if st.button("⬅ 1단계로 돌아가기"):
        st.session_state.step = 1
        st.session_state.heater_points = []
        st.session_state.heat_result = None
        st.rerun()

    heaters = []

    for i in range(heater_count):
        st.markdown(f"### 🔥 열풍기 #{i+1}")
        hx = st.number_input(
            f"X 좌표 (m) - 열풍기 {i+1}",
            step=0.001,
            format="%.3f",
            key=f"hx_{i}"
        )
        hy = st.number_input(
            f"Y 좌표 (m) - 열풍기 {i+1}",
            step=0.001,
            format="%.3f",
            key=f"hy_{i}"
        )
        heaters.append((hx, hy))

    # 🔍 임시 시각화
    xs, ys = zip(*st.session_state.space_points)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers", name="공간"
    ))

    if heaters:
        hx, hy = zip(*heaters)
        fig2.add_trace(go.Scatter(
            x=hx, y=hy,
            mode="markers",
            marker=dict(size=12, color="red"),
            name="열풍기 (임시)"
        ))

    fig2.update_layout(
        height=400,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        title="열풍기 임시 배치 미리보기"
    )
    st.plotly_chart(fig2, use_container_width=True)

    if st.button("🔥 열풍기 위치 확정"):
        invalid = False
        for hx, hy in heaters:
            if not point_in_polygon(hx, hy, st.session_state.space_points):
                invalid = True
                break

        if invalid:
            st.error("❌ 모든 열풍기는 내부공간 안에 있어야 합니다.")
        else:
            st.session_state.heater_points = heaters
            st.session_state.step = 3
            st.rerun()

# ======================================================
# 3단계 이후 (열해석/시각화)
# → 이전에 준 코드와 동일, 변경 없음
# ======================================================
st.info("ℹ️ 3단계 열해석 및 시각화는 이전 최종본과 동일하게 이어서 사용하면 됩니다.")
