import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.path import Path

# ======================================================
# 기본 설정
# ======================================================
st.set_page_config(page_title="난방 시뮬레이터", layout="wide")

# ======================================================
# 세션 상태 초기화
# ======================================================
def reset_all():
    st.session_state.step = 1
    st.session_state.space_points = [(0.0, 0.0)]
    st.session_state.heater_points = []
    st.session_state.heater_count = 1
    st.session_state.heat_result = None
    st.session_state.df_result = None
    st.session_state.html_result = None

def clear_simulation_result():
    st.session_state.heat_result = None
    st.session_state.df_result = None
    st.session_state.html_result = None

if "step" not in st.session_state:
    reset_all()

# ======================================================
# 열해석 함수 (단순 확산 모델)
# ======================================================
def run_heat_simulation(space_pts, heater_pts):
    pts = np.array(space_pts)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)

    nx, ny = 50, 50
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    poly = Path(space_pts)
    mask = poly.contains_points(
        np.vstack((X.flatten(), Y.flatten())).T
    ).reshape(X.shape)

    T = np.full_like(X, 10.0)  # 초기 내부온도 10℃
    T_hist = []

    alpha = 0.15
    dt = 1.0
    hours = 9

    for _ in range(hours + 1):
        Tn = T.copy()

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if not mask[j, i]:
                    continue
                Tn[j, i] += alpha * (
                    T[j+1, i] + T[j-1, i] +
                    T[j, i+1] + T[j, i-1] -
                    4*T[j, i]
                )

        for hx, hy in heater_pts:
            ix = np.argmin(np.abs(x - hx))
            iy = np.argmin(np.abs(y - hy))
            Tn[iy, ix] += 5.0

        T = np.clip(Tn, -10, 40)
        T_hist.append(T.copy())

    return T_hist, x, y, X, Y, mask

# ======================================================
# UI 시작
# ======================================================
st.title("🔥 난방 시뮬레이터")

st.button("🔄 전체 초기화", on_click=reset_all)

# ======================================================
# 1단계: 공간 정의
# ======================================================
if st.session_state.step == 1:
    st.subheader("1️⃣ 공간 좌표 입력 (단위: m)")

    x = st.number_input("다음 X 좌표 (m)", step=0.001, format="%.3f")
    y = st.number_input("다음 Y 좌표 (m)", step=0.001, format="%.3f")

    if st.button("➕ 좌표 추가"):
        st.session_state.space_points.append((x, y))

    if len(st.session_state.space_points) > 2:
        if st.button("✔ 공간 완성"):
            st.session_state.step = 2
            st.rerun()

    fig = go.Figure()
    xs, ys = zip(*st.session_state.space_points)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers"))
    fig.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1),
        title="현재 공간 형상"
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# 2단계: 열풍기 배치
# ======================================================
if st.session_state.step == 2:
    st.subheader("2️⃣ 열풍기 배치")

    if st.button("⬅ 1단계로 돌아가기"):
        clear_simulation_result()
        st.session_state.heater_points = []
        st.session_state.step = 1
        st.rerun()

    st.session_state.heater_count = st.radio(
        "열풍기 개수", [1, 2], horizontal=True
    )

    heaters = []
    for i in range(st.session_state.heater_count):
        st.markdown(f"**열풍기 {i+1} 좌표**")
        hx = st.number_input(
            f"X{i+1} (m)", step=0.001, format="%.3f", key=f"hx{i}"
        )
        hy = st.number_input(
            f"Y{i+1} (m)", step=0.001, format="%.3f", key=f"hy{i}"
        )
        heaters.append((hx, hy))

    fig = go.Figure()
    xs, ys = zip(*st.session_state.space_points)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines"))
    if heaters:
        hx, hy = zip(*heaters)
        fig.add_trace(go.Scatter(
            x=hx, y=hy,
            mode="markers+text",
            marker=dict(size=12, color="red"),
            text=[f"🔥{i+1}" for i in range(len(hx))],
            textposition="top center"
        ))
    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
    st.plotly_chart(fig, use_container_width=True)

    if st.button("➡ 3단계로 이동"):
        st.session_state.heater_points = heaters
        st.session_state.step = 3
        st.rerun()

# ======================================================
# 3단계: 열해석 결과
# ======================================================
if st.session_state.step == 3:
    st.subheader("3️⃣ 시뮬레이션 결과")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ 2단계로 돌아가기"):
            clear_simulation_result()
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("⬅⬅ 1단계로 돌아가기"):
            clear_simulation_result()
            st.session_state.heater_points = []
            st.session_state.step = 1
            st.rerun()

    if st.button("🧮 열해석 실행"):
        with st.spinner("계산 중..."):
            result = run_heat_simulation(
                st.session_state.space_points,
                st.session_state.heater_points
            )
            st.session_state.heat_result = result

    if st.session_state.heat_result is not None:
        T_hist, x, y, X, Y, mask = st.session_state.heat_result

        rows = []
        for t, T in enumerate(T_hist):
            T2 = T.copy()
            T2[~mask] = np.nan
            rows.append({
                "시간(h)": t,
                "중심부 평균온도(°C)": np.nanmean(T2),
                "모서리 평균온도(°C)": np.nanmean([
                    T2[0,0], T2[0,-1], T2[-1,0], T2[-1,-1]
                ])
            })

        df = pd.DataFrame(rows)
        st.session_state.df_result = df

        st.line_chart(df.set_index("시간(h)"))

        frames = []
        wind = np.deg2rad(20)
        arrow = 0.2*(x.max()-x.min())

        for t, T in enumerate(T_hist):
            T2 = T.copy()
            T2[~mask] = np.nan

            data = [
                go.Heatmap(
                    z=T2, x=x, y=y,
                    zmin=-10, zmax=40,
                    colorscale="Turbo"
                )
            ]

            hx, hy = zip(*st.session_state.heater_points)
            data.append(go.Scatter(
                x=hx, y=hy,
                mode="markers+text",
                marker=dict(size=14, color="red"),
                text=["🔥"]*len(hx)
            ))

            for px, py in st.session_state.heater_points:
                data.append(go.Scatter(
                    x=[px, px+arrow*np.cos(wind)],
                    y=[py, py+arrow*np.sin(wind)],
                    mode="lines",
                    line=dict(width=3, color="black"),
                    showlegend=False
                ))

            frames.append(go.Frame(data=data, name=str(t)))

        fig = go.Figure(data=frames[0].data, frames=frames)
        fig.update_layout(
            yaxis=dict(scaleanchor="x", scaleratio=1),
            updatemenus=[{
                "type": "buttons",
                "buttons": [{"label": "▶ 재생", "method": "animate", "args": [None]}]
            }]
        )

        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "⬇ CSV 다운로드",
            df.to_csv(index=False),
            file_name="simulation_result.csv"
        )

        st.download_button(
            "⬇ HTML 다운로드",
            fig.to_html(),
            file_name="heatmap_animation.html"
        )
