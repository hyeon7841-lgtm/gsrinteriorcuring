import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import numpy as np
import pandas as pd
import io

# ======================================================
# 기본 설정
# ======================================================
st.set_page_config(layout="wide")
st.title("🔥 내부공간 열풍기 난방 시뮬레이터")

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
    st.session_state.heat_result = None

# ======================================================
# 사이드바
# ======================================================
st.sidebar.header("환경 설정")

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

if st.sidebar.button("❌ 전체 초기화"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# ======================================================
# 내부 판별
# ======================================================
def point_in_polygon(x, y, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside

# ======================================================
# 1단계: 공간 정의
# ======================================================
st.subheader("🧱 1단계: 내부공간 정의 (단위: m)")

if not st.session_state.space_closed:
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        x = st.number_input("X (m)", 0.000, step=0.001, format="%.3f")
    with c2:
        y = st.number_input("Y (m)", 0.000, step=0.001, format="%.3f")
    with c3:
        if st.button("➕ 선 추가"):
            st.session_state.space_points.append((x,y))
            st.rerun()
        if len(st.session_state.space_points) > 1:
            if st.button("⬅ 이전 단계"):
                st.session_state.space_points.pop()
                st.rerun()
        if len(st.session_state.space_points) >= 3:
            if st.button("✅ 공간 완성"):
                st.session_state.space_points.append((0.0,0.0))
                st.session_state.space_closed = True
                st.rerun()

# ======================================================
# 공간 시각화
# ======================================================
fig = go.Figure()

if st.session_state.space_points:
    xs, ys = zip(*st.session_state.space_points)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers"))

if st.session_state.heater_points:
    hx, hy = zip(*st.session_state.heater_points)
    fig.add_trace(go.Scatter(
        x=hx, y=hy,
        mode="markers",
        marker=dict(size=14, color="red")
    ))

fig.update_layout(
    height=420,
    clickmode="event",
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(fixedrange=True),
    yaxis_fixedrange=True
)

st.plotly_chart(fig, use_container_width=True)

# ======================================================
# 2단계: 열풍기 배치
# ======================================================
if st.session_state.space_closed:
    st.subheader("🔥 2단계: 열풍기 배치")

    clicked = plotly_events(fig, click_event=True)
    if clicked:
        st.session_state.temp_heater = (
            float(clicked[0]["x"]),
            float(clicked[0]["y"])
        )

    if st.session_state.temp_heater:
        hx, hy = st.session_state.temp_heater
        if st.button("🔥 위치 확정"):
            if point_in_polygon(hx, hy, st.session_state.space_points):
                if len(st.session_state.heater_points) < heater_count:
                    st.session_state.heater_points.append((hx,hy))
                    st.session_state.temp_heater = None
                    st.rerun()

# ======================================================
# 열해석
# ======================================================
def run_heat_simulation(space, heaters):
    alpha = 1e-6
    rho, cp = 1.2, 1005
    heater_power = 20461
    total_hours = 9

    theta = np.deg2rad(20)
    wind_speed = 0.3
    u = wind_speed * np.cos(theta)
    v = wind_speed * np.sin(theta)

    xs, ys = zip(*space)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    nx, ny = 60, 40
    dx = (max_x-min_x)/nx
    dy = (max_y-min_y)/ny
    dt = 3600

    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    X, Y = np.meshgrid(x,y)

    mask = np.zeros((ny,nx), bool)
    for i in range(nx):
        for j in range(ny):
            mask[j,i] = point_in_polygon(X[j,i], Y[j,i], space)

    T = np.ones((ny,nx)) * T_inside0
    history = [T.copy()]

    for _ in range(total_hours):
        Tn = T.copy()
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                if not mask[j,i]:
                    continue

                lap = (
                    (T[j,i+1]-2*T[j,i]+T[j,i-1])/dx**2 +
                    (T[j+1,i]-2*T[j,i]+T[j-1,i])/dy**2
                )

                adv = -(
                    u*(T[j,i]-T[j,i-1])/dx +
                    v*(T[j,i]-T[j-1,i])/dy
                )

                loss = U*(T[j,i]-T_outside)/(rho*cp)

                Tn[j,i] += dt*(alpha*lap + adv - loss)

        for hx,hy in heaters:
            ix = np.argmin(np.abs(x-hx))
            iy = np.argmin(np.abs(y-hy))
            Tn[iy,ix] += heater_power*dt/(rho*cp*dx*dy)

        T = Tn
        history.append(T.copy())

    return history, x, y, X, Y, mask

# ======================================================
# 3단계: 결과
# ======================================================
if st.session_state.heater_points:
    st.subheader("🌡️ 3단계: 열해석 결과")

    if st.button("🧮 열해석 계산 실행"):
        with st.spinner("계산 중..."):
            st.session_state.heat_result = run_heat_simulation(
                st.session_state.space_points,
                st.session_state.heater_points
            )

    if st.session_state.heat_result:
        T_hist, x, y, X, Y, mask = st.session_state.heat_result

        rows = []
        cx = (x.min()+x.max())/2
        cy = (y.min()+y.max())/2
        rx = 0.1*(x.max()-x.min())
        ry = 0.1*(y.max()-y.min())

        for t, Th in enumerate(T_hist):
            Tm = Th.copy()
            Tm[~mask] = np.nan

            center_mask = (
                (X >= cx-rx) & (X <= cx+rx) &
                (Y >= cy-ry) & (Y <= cy+ry)
            )

            center_avg = np.nanmean(Tm[center_mask])

            corners = [
                (x.min(), y.min()), (x.min(), y.max()),
                (x.max(), y.max()), (x.max(), y.min())
            ]

            corner_vals = []
            for px,py in corners:
                ix = np.argmin(np.abs(x-px))
                iy = np.argmin(np.abs(y-py))
                corner_vals.append(Tm[iy,ix])

            corner_avg = np.nanmean(corner_vals)

            rows.append({
                "시간(h)": t,
                "중심부 평균온도": center_avg,
                "모서리 평균온도": corner_avg
            })

        df = pd.DataFrame(rows)

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=df["시간(h)"], y=df["중심부 평균온도"], name="중심부 평균온도"
        ))
        fig_line.add_trace(go.Scatter(
            x=df["시간(h)"], y=df["모서리 평균온도"], name="모서리 평균온도"
        ))

        fig_line.update_layout(
            title="시간별 온도 변화",
            xaxis_title="시간 (h)",
            yaxis_title="온도 (°C)"
        )

        st.plotly_chart(fig_line, use_container_width=True)

        frames = []
        for t, Th in enumerate(T_hist):
            Th2 = Th.copy()
            Th2[~mask] = np.nan
            frames.append(go.Frame(
                data=[go.Heatmap(
                    z=Th2, x=x, y=y,
                    zmin=-10, zmax=40,
                    colorscale="Turbo"
                )],
                name=str(t)
            ))

        fig_anim = go.Figure(data=frames[0].data, frames=frames)
        fig_anim.update_layout(
            title="시간 경과 Heatmap (바닥면)",
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
