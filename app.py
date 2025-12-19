import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio

# ======================================================
# 기본 설정
# ======================================================
st.set_page_config(layout="wide")
st.title("🔥 내부공간 열풍기 난방 시뮬레이터")

# ======================================================
# 세션 초기화
# ======================================================
def reset_all():
    st.session_state.step = 1
    st.session_state.space_points = [(0.0, 0.0)]
    st.session_state.heater_points = []
    st.session_state.heat_result = None
    st.session_state.df_result = None
    st.session_state.html_result = None

if "step" not in st.session_state:
    reset_all()
if "df_result" not in st.session_state:
    st.session_state.df_result = None

if "html_result" not in st.session_state:
    st.session_state.html_result = None
# ======================================================
# 사이드바
# ======================================================
st.sidebar.header("공통 설정")

if st.sidebar.button("🔄 전체 초기화"):
    reset_all()
    st.rerun()

heater_count = st.sidebar.selectbox("열풍기 개수", [1, 2])
ceiling_height = st.sidebar.number_input("천장 높이 (m)", 2.0, 15.0, 4.0)
wall_type = st.sidebar.selectbox("벽체 재질", ["조적벽", "콘크리트벽", "샌드위치판넬"])

U_map = {"조적벽": 1.2, "콘크리트벽": 1.7, "샌드위치판넬": 0.25}
U = U_map[wall_type]

T_inside0, T_outside = 10.0, -5.0

# ======================================================
# 내부 판별
# ======================================================
def point_in_polygon(x, y, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi+1e-9)+xi):
            inside = not inside
        j = i
    return inside

# ======================================================
# 1단계: 공간 정의
# ======================================================
if st.session_state.step == 1:
    st.subheader("🧱 1단계: 내부공간 정의 (m)")
    x = st.number_input("X 좌표", step=0.001, format="%.3f")
    y = st.number_input("Y 좌표", step=0.001, format="%.3f")

    if st.button("➕ 선 추가"):
        st.session_state.space_points.append((x, y))
        st.rerun()

    if len(st.session_state.space_points) > 1 and st.button("⬅ 이전 점 삭제"):
        st.session_state.space_points.pop()
        st.rerun()

    if len(st.session_state.space_points) >= 3 and st.button("✅ 공간 완성"):
        st.session_state.space_points.append((0.0, 0.0))
        st.session_state.step = 2
        st.rerun()

    xs, ys = zip(*st.session_state.space_points)
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers"))
    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# 2단계: 열풍기 입력
# ======================================================
if st.session_state.step == 2:
    st.subheader("🔥 2단계: 열풍기 좌표 입력")

    if st.button("⬅ 1단계로 돌아가기"):
        st.session_state.step = 1
        st.rerun()

    heaters = []
    for i in range(heater_count):
        hx = st.number_input(f"열풍기{i+1} X", step=0.001, format="%.3f", key=f"hx{i}")
        hy = st.number_input(f"열풍기{i+1} Y", step=0.001, format="%.3f", key=f"hy{i}")
        heaters.append((hx, hy))

    xs, ys = zip(*st.session_state.space_points)
    fig2 = go.Figure(go.Scatter(x=xs, y=ys, mode="lines"))
    if heaters:
        hx, hy = zip(*heaters)
        fig2.add_trace(go.Scatter(x=hx, y=hy, mode="markers", marker=dict(size=12)))
    fig2.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
    st.plotly_chart(fig2, use_container_width=True)

    if st.button("🔥 위치 확정"):
        for hx, hy in heaters:
            if not point_in_polygon(hx, hy, st.session_state.space_points):
                st.error("❌ 열풍기가 내부 공간 밖에 있습니다.")
                break
        else:
            st.session_state.heater_points = heaters
            st.session_state.step = 3
            st.rerun()

# ======================================================
# 열해석
# ======================================================
def run_heat_sim(space, heaters):
    nx, ny = 60, 40
    x = np.linspace(min(p[0] for p in space), max(p[0] for p in space), nx)
    y = np.linspace(min(p[1] for p in space), max(p[1] for p in space), ny)
    X, Y = np.meshgrid(x, y)

    mask = np.zeros((ny, nx), bool)
    for i in range(nx):
        for j in range(ny):
            mask[j, i] = point_in_polygon(X[j, i], Y[j, i], space)

    T = np.ones((ny, nx)) * T_inside0
    hist = [T.copy()]

    for _ in range(9):
        Tn = T.copy()
        for hx, hy in heaters:
            ix = np.argmin(abs(x-hx))
            iy = np.argmin(abs(y-hy))
            Tn[iy, ix] += 5
        T = Tn
        hist.append(T.copy())

    return hist, x, y, X, Y, mask

# ======================================================
# 3단계: 결과 + 다운로드
# ======================================================
if st.session_state.step == 3:
    st.subheader("🌡️ 3단계: 결과")

    if st.button("🧮 시뮬레이션 실행"):
        T_hist, x, y, X, Y, mask = run_heat_sim(
            st.session_state.space_points,
            st.session_state.heater_points
        )

        rows = []
        for t, Th in enumerate(T_hist):
            Th2 = Th.copy()
            Th2[~mask] = np.nan
            rows.append({
                "시간(h)": t,
                "중심부 평균온도": np.nanmean(Th2),
                "모서리 평균온도": np.nanmean([Th2[0,0], Th2[0,-1], Th2[-1,0], Th2[-1,-1]])
            })

        df = pd.DataFrame(rows)
        st.session_state.df_result = df

        frames = []
        for t, Th in enumerate(T_hist):
            Th2 = Th.copy()
            Th2[~mask] = np.nan
            frames.append(go.Frame(
                data=[go.Heatmap(z=Th2, x=x, y=y, zmin=-10, zmax=40)],
                name=str(t)
            ))

        fig = go.Figure(data=frames[0].data, frames=frames)
        fig.update_layout(
            yaxis=dict(scaleanchor="x", scaleratio=1),
            updatemenus=[{"type": "buttons",
                          "buttons": [{"label": "▶ 재생", "method": "animate", "args": [None]}]}]
        )

        st.plotly_chart(fig, use_container_width=True)

        st.session_state.html_result = pio.to_html(fig, full_html=True)

    # -------------------------
    # 다운로드 영역
    # -------------------------
    if st.session_state.df_result is not None:
        st.download_button(
            "📥 CSV 다운로드",
            st.session_state.df_result.to_csv(index=False).encode("utf-8-sig"),
            file_name="heating_result.csv",
            mime="text/csv"
        )

    if st.session_state.html_result is not None:
        st.download_button(
            "📥 시뮬레이션 HTML 다운로드",
            st.session_state.html_result.encode("utf-8"),
            file_name="heating_simulation.html",
            mime="text/html"
        )
