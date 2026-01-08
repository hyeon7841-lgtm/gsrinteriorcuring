import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon

# =====================
# 기본 설정
# =====================
st.set_page_config(layout="wide")
st.title("🔥 실내 난방 열 시뮬레이터 (열대류 적용)")

GRID = 0.5
ALPHA = 0.15
MIXING_BASE = 0.02
LOSS = 0.01
BUOYANCY = 0.03          # 🔥 자연대류 강도
HEATER_RADIUS = 10.0     # 고정 영향반경
FAN_ANGLE = 60           # 부채꼴 각도

# =====================
# 세션 초기화
# =====================
def reset():
    st.session_state.step = 1
    st.session_state.space = []
    st.session_state.heaters = []
    st.session_state.df_result = None

if "step" not in st.session_state:
    reset()

# =====================
# STEP 1 공간 정의
# =====================
if st.session_state.step == 1:
    st.subheader("1️⃣ 공간 정의 (좌표 입력)")

    cols = st.columns(2)

    with cols[0]:
        x = st.number_input("X 좌표", value=0.0, step=1.0)
        y = st.number_input("Y 좌표", value=0.0, step=1.0)

        if st.button("좌표 추가"):
            st.session_state.space.append((x, y))

        if st.button("다음 단계"):
            if len(st.session_state.space) >= 3:
                st.session_state.step = 2
                st.experimental_rerun()

    with cols[1]:
        if len(st.session_state.space) >= 2:
            poly = Polygon(st.session_state.space)
            xs, ys = zip(*(st.session_state.space + [st.session_state.space[0]]))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=xs, y=ys, fill="toself",
                mode="lines+markers"
            ))
            fig.update_layout(title="공간 미리보기", yaxis_scaleanchor="x")
            st.plotly_chart(fig, use_container_width=True)

    st.button("전체 초기화", on_click=reset)

# =====================
# STEP 2 열풍기 배치
# =====================
if st.session_state.step == 2:
    st.subheader("2️⃣ 열풍기 배치")

    heater_count = st.number_input("열풍기 수량", min_value=1, max_value=2, value=1)

    st.session_state.heaters = []

    for i in range(heater_count):
        st.markdown(f"### 🔥 열풍기 {i+1}")
        hx = st.number_input(f"X 위치 {i+1}", key=f"x{i}")
        hy = st.number_input(f"Y 위치 {i+1}", key=f"y{i}")
        angle = st.slider(f"풍향 각도 {i+1}", 0, 360, 0)
        st.session_state.heaters.append((hx, hy, angle))

    # 🔍 미리보기
    poly = Polygon(st.session_state.space)
    xs, ys = zip(*(st.session_state.space + [st.session_state.space[0]]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself"))

    for hx, hy, ang in st.session_state.heaters:
        fig.add_trace(go.Scatter(
            x=[hx], y=[hy],
            mode="markers+text",
            text=["🔥"],
            textposition="top center"
        ))

        rad = np.deg2rad(ang)
        fig.add_trace(go.Scatter(
            x=[hx, hx + np.cos(rad)*3],
            y=[hy, hy + np.sin(rad)*3],
            mode="lines"
        ))

    fig.update_layout(title="열풍기 배치 미리보기", yaxis_scaleanchor="x")
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(3)
    cols[0].button("⬅ 이전 단계", on_click=lambda: setattr(st.session_state, "step", 1))
    cols[1].button("▶ 시뮬레이션", on_click=lambda: setattr(st.session_state, "step", 3))
    cols[2].button("전체 초기화", on_click=reset)

# =====================
# STEP 3 시뮬레이션
# =====================
if st.session_state.step == 3:
    st.subheader("3️⃣ 열 시뮬레이션")

    poly = Polygon(st.session_state.space)
    minx, miny, maxx, maxy = poly.bounds

    xs = np.arange(minx, maxx, GRID)
    ys = np.arange(miny, maxy, GRID)
    nx, ny = len(xs), len(ys)

    T = np.ones((ny, nx)) * 5.0  # 초기 내부온도
    mask = np.zeros_like(T, dtype=bool)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if poly.contains(Point(x, y)):
                mask[j, i] = True

    steps = 60
    history = []

    for step in range(steps):
        Tn = T.copy()

        # 확산
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                if mask[j, i]:
                    Tn[j, i] += ALPHA * (
                        T[j+1,i]+T[j-1,i]+T[j,i+1]+T[j,i-1]-4*T[j,i]
                    )

        # 열풍기
        for hx, hy, ang in st.session_state.heaters:
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    if not mask[j,i]:
                        continue
                    dx, dy = x-hx, y-hy
                    d = np.hypot(dx, dy)
                    if d > HEATER_RADIUS:
                        continue
                    theta = (np.degrees(np.arctan2(dy, dx)) - ang + 360) % 360
                    if theta < FAN_ANGLE/2 or theta > 360-FAN_ANGLE/2:
                        Tn[j,i] += 0.6 * (1-d/HEATER_RADIUS)

        # 🔥 열대류
        Tm = np.mean(Tn[mask])
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                if mask[j,i]:
                    buoy = BUOYANCY * (Tn[j,i]-Tm)
                    if j > 0 and mask[j-1,i]:
                        Tn[j-1,i] += buoy
                        Tn[j,i] -= buoy

        # 혼합
        mix = MIXING_BASE + 0.15*(step/steps)
        Tn[mask] += mix*(Tm - Tn[mask])

        # 손실
        Tn[mask] -= LOSS

        T = Tn
        history.append(T.copy())

    # 결과 시각화
    fig = go.Figure(
        data=[go.Heatmap(
            z=history[-1],
            x=xs, y=ys,
            hovertemplate="x=%{x:.1f}<br>y=%{y:.1f}<br>온도=%{z:.1f}℃"
        )]
    )
    fig.update_layout(title="최종 온도 분포", yaxis_scaleanchor="x")
    st.plotly_chart(fig, use_container_width=True)

    # CSV
    df = pd.DataFrame([
        {"x": xs[i], "y": ys[j], "temp": history[-1][j,i]}
        for i in range(nx) for j in range(ny) if mask[j,i]
    ])
    st.download_button("CSV 다운로드", df.to_csv(index=False), "result.csv")

    cols = st.columns(3)
    cols[0].button("⬅ 2단계", on_click=lambda: setattr(st.session_state,"step",2))
    cols[1].button("⬅ 1단계", on_click=lambda: setattr(st.session_state,"step",1))
    cols[2].button("전체 초기화", on_click=reset)
