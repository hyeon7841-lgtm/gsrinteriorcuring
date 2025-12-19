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
# 세션 초기화
# ======================================================
def reset_all():
    st.session_state.step = 1
    st.session_state.space_points = [(0.0, 0.0)]
    st.session_state.heater_points = []
    st.session_state.heat_result = None

if "step" not in st.session_state:
    reset_all()

# ======================================================
# 사이드바 (공통)
# ======================================================
st.sidebar.header("공통 설정")

if st.sidebar.button("🔄 전체 초기화"):
    reset_all()
    st.rerun()

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
# 2단계: 열풍기 좌표 입력
# ======================================================
if st.session_state.step == 2:
    st.subheader("🔥 2단계: 열풍기 좌표 입력 (단위: m)")

    if st.button("⬅ 1단계로 돌아가기"):
        st.session_state.step = 1
        st.session_state.heater_points = []
        st.rerun()

    heaters = []

    for i in range(heater_count):
        st.markdown(f"### 🔥 열풍기 #{i+1}")
        hx = st.number_input(
            f"X 좌표 (m)", step=0.001, format="%.3f", key=f"hx{i}"
        )
        hy = st.number_input(
            f"Y 좌표 (m)", step=0.001, format="%.3f", key=f"hy{i}"
        )
        heaters.append((hx, hy))

    xs, ys = zip(*st.session_state.space_points)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="공간"))

    if heaters:
        hx, hy = zip(*heaters)
        fig2.add_trace(go.Scatter(
            x=hx, y=hy, mode="markers",
            marker=dict(size=12, color="red"),
            name="열풍기(임시)"
        ))

    fig2.update_layout(
        height=400,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        title="열풍기 배치 미리보기"
    )
    st.plotly_chart(fig2, use_container_width=True)

    if st.button("🔥 열풍기 위치 확정"):
        for hx, hy in heaters:
            if not point_in_polygon(hx, hy, st.session_state.space_points):
                st.error("❌ 모든 열풍기는 내부공간 안에 있어야 합니다.")
                break
        else:
            st.session_state.heater_points = heaters
            st.session_state.step = 3
            st.rerun()

# ======================================================
# 열해석 계산
# ======================================================
def run_heat_simulation(space, heaters):
    nx, ny = 60, 40
    alpha = 1e-6
    rho, cp = 1.2, 1005
    heater_power = 20461
    total_hours = 9
    dt = 3600

    xs, ys = zip(*space)
    x = np.linspace(min(xs), max(xs), nx)
    y = np.linspace(min(ys), max(ys), ny)
    X, Y = np.meshgrid(x, y)

    dx = (x.max() - x.min()) / nx
    dy = (y.max() - y.min()) / ny

    mask = np.zeros((ny, nx), bool)
    for i in range(nx):
        for j in range(ny):
            mask[j, i] = point_in_polygon(X[j, i], Y[j, i], space)

    T = np.ones((ny, nx)) * T_inside0
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
                loss = U * (T[j, i] - T_outside) / (rho * cp)
                Tn[j, i] += dt * (alpha * lap - loss)

        for hx, hy in heaters:
            ix = np.argmin(np.abs(x - hx))
            iy = np.argmin(np.abs(y - hy))
            Tn[iy, ix] += heater_power * dt / (rho * cp * dx * dy)

        T = Tn
        history.append(T.copy())

    return history, x, y, X, Y, mask

# ======================================================
# 3단계: 결과 시각화
# ======================================================
if st.session_state.step == 3:
    st.subheader("🌡️ 3단계: 열해석 결과")

    if st.button("⬅ 2단계로 돌아가기"):
        st.session_state.step = 2
        st.session_state.heat_result = None
        st.rerun()

    if st.button("🧮 열해석 계산 실행"):
        with st.spinner("계산 중..."):
            st.session_state.heat_result = run_heat_simulation(
                st.session_state.space_points,
                st.session_state.heater_points
            )

    if st.session_state.heat_result:
        T_hist, x, y, X, Y, mask = st.session_state.heat_result

        rows = []
        cx, cy = (x.min()+x.max())/2, (y.min()+y.max())/2
        rx, ry = 0.1*(x.max()-x.min()), 0.1*(y.max()-y.min())

        for t, Th in enumerate(T_hist):
            Th2 = Th.copy()
            Th2[~mask] = np.nan
            center = np.nanmean(
                Th2[(X>=cx-rx)&(X<=cx+rx)&(Y>=cy-ry)&(Y<=cy+ry)]
            )
            corners = [Th2[0,0], Th2[0,-1], Th2[-1,-1], Th2[-1,0]]
            rows.append({
                "시간(h)": t,
                "중심부 평균온도": center,
                "모서리 평균온도": np.nanmean(corners)
            })

        df = pd.DataFrame(rows)

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=df["시간(h)"], y=df["중심부 평균온도"], name="중심부 평균"
        ))
        fig_line.add_trace(go.Scatter(
            x=df["시간(h)"], y=df["모서리 평균온도"], name="모서리 평균"
        ))
        fig_line.update_layout(yaxis_title="°C")
        st.plotly_chart(fig_line, use_container_width=True)

        wind_angle = np.deg2rad(20)
        arrow_len = 0.3 * (x.max() - x.min())

        frames = []
        for t, Th in enumerate(T_hist):
            Th2 = Th.copy()
            Th2[~mask] = np.nan
            data = [
                go.Heatmap(
                    z=Th2, x=x, y=y,
                    zmin=-10, zmax=40,
                    colorscale="Turbo"
                )
            ]

            hx, hy = zip(*st.session_state.heater_points)
            data.append(go.Scatter(
                x=hx, y=hy,
                mode="markers+text",
                marker=dict(size=14, color="red"),
                text=[f"🔥{i+1}" for i in range(len(hx))],
                textposition="top center"
            ))

            for hx_i, hy_i in st.session_state.heater_points:
                data.append(go.Scatter(
                    x=[hx_i, hx_i + arrow_len*np.cos(wind_angle)],
                    y=[hy_i, hy_i + arrow_len*np.sin(wind_angle)],
                    mode="lines",
                    line=dict(width=3, color="black"),
                    showlegend=False
                ))

            frames.append(go.Frame(data=data, name=str(t)))

        fig_anim = go.Figure(data=frames[0].data, frames=frames)
        fig_anim.update_layout(
            yaxis=dict(scaleanchor="x", scaleratio=1),
            updatemenus=[{
                "type": "buttons",
                "buttons": [{
                    "label": "▶ 재생",
                    "method": "animate",
                    "args": [None]
                }]
            }]
        )
        st.plotly_chart(fig_anim, use_container_width=True)
