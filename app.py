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
st.title("🔥 내부공간 열풍기 배치 및 열해석 시뮬레이터")

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
inside_temp = st.sidebar.number_input("초기 내부온도 (°C)", value=10.0, step=0.5)

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
        x = st.number_input("X 좌표 (m)", 0.000, step=0.001, format="%.3f")
    with c2:
        y = st.number_input("Y 좌표 (m)", 0.000, step=0.001, format="%.3f")
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
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers",
        line=dict(width=3), marker=dict(size=7),
        name="공간"
    ))

if st.session_state.heater_points:
    hx, hy = zip(*st.session_state.heater_points)
    fig.add_trace(go.Scatter(
        x=hx, y=hy, mode="markers",
        marker=dict(size=14, color="red"),
        name="열풍기"
    ))

fig.update_layout(
    height=450,
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

    if st.button("⬅ 이전 열풍기"):
        if st.session_state.heater_points:
            st.session_state.heater_points.pop()
            st.session_state.heat_result = None
            st.rerun()

    clicked = plotly_events(fig, click_event=True)

    if clicked:
        st.session_state.temp_heater = (
            float(clicked[0]["x"]),
            float(clicked[0]["y"])
        )

    if st.session_state.temp_heater:
        hx, hy = st.session_state.temp_heater
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            hx = st.number_input("X (m)", hx, step=0.001, format="%.3f")
        with c2:
            hy = st.number_input("Y (m)", hy, step=0.001, format="%.3f")
        with c3:
            if st.button("🔥 위치 확정"):
                if point_in_polygon(hx, hy, st.session_state.space_points):
                    if len(st.session_state.heater_points) < heater_count:
                        st.session_state.heater_points.append((hx,hy))
                        st.session_state.temp_heater = None
                        st.session_state.heat_result = None
                        st.rerun()

# ======================================================
# 열해석 함수
# ======================================================
def run_heat_simulation(space, heaters, T0):
    alpha = 1e-6
    rho, cp = 2400, 900
    heater_power = 20461
    total_hours = 9

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

    T = np.ones((ny,nx))*T0
    history = [T.copy()]

    for _ in range(total_hours):
        Tn = T.copy()
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                if not mask[j,i]: continue
                lap = (
                    (T[j,i+1]-2*T[j,i]+T[j,i-1])/dx**2 +
                    (T[j+1,i]-2*T[j,i]+T[j-1,i])/dy**2
                )
                Tn[j,i] += alpha*dt*lap

        for hx,hy in heaters:
            ix = np.argmin(np.abs(x-hx))
            iy = np.argmin(np.abs(y-hy))
            if mask[iy,ix]:
                Tn[iy,ix] += heater_power*dt/(rho*cp*dx*dy)

        T = Tn
        history.append(T.copy())

    return history, x, y, mask

# ======================================================
# 3단계: 결과 + 지표 + 저장
# ======================================================
if st.session_state.heater_points:
    st.subheader("🌡️ 3단계: 열해석 결과")

    if st.button("🧮 열해석 계산 실행"):
        with st.spinner("계산 중..."):
            st.session_state.heat_result = run_heat_simulation(
                st.session_state.space_points,
                st.session_state.heater_points,
                inside_temp
            )

    if st.session_state.heat_result:
        T_hist, x, y, mask = st.session_state.heat_result
        t = st.slider("경과 시간 (h)", 0, 9, 0)

        T = T_hist[t].copy()
        T[~mask] = np.nan

        # ---- 지표 계산 ----
        avg_temp = np.nanmean(T)

        cx, cy = (x.min()+x.max())/2, (y.min()+y.max())/2
        ix = np.argmin(np.abs(x-cx))
        iy = np.argmin(np.abs(y-cy))
        center_temp = T[iy,ix]

        corners = [
            (x.min(), y.min()),
            (x.min(), y.max()),
            (x.max(), y.max()),
            (x.max(), y.min())
        ]
        corner_vals = []
        for px,py in corners:
            ix = np.argmin(np.abs(x-px))
            iy = np.argmin(np.abs(y-py))
            corner_vals.append(T[iy,ix])
        corner_avg = np.nanmean(corner_vals)

        st.markdown(f"""
        **평균 온도:** {avg_temp:.2f} °C  
        **중앙 온도:** {center_temp:.2f} °C  
        **꼭지점 평균 온도:** {corner_avg:.2f} °C
        """)

        # ---- Heatmap ----
        figT = go.Figure(
            data=go.Heatmap(
                z=T, x=x, y=y,
                colorscale="Turbo",
                zmin=-10, zmax=40,
                colorbar=dict(title="온도 (°C)")
            )
        )
        figT.update_layout(
            height=450,
            yaxis=dict(scaleanchor="x", scaleratio=1),
            title=f"{t}시간 후 온도 분포"
        )

        st.plotly_chart(figT, use_container_width=True)

        # ---- CSV 저장 ----
        data = []
        for i,Th in enumerate(T_hist):
            Th2 = Th.copy()
            Th2[~mask] = np.nan
            data.append({
                "시간(h)": i,
                "평균온도": np.nanmean(Th2)
            })
        df = pd.DataFrame(data)

        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📄 결과 CSV 다운로드",
            csv,
            "temperature_result.csv",
            "text/csv"
        )

        # ---- 이미지 저장 ----
        html_buf = io.StringIO()
figT.write_html(html_buf, include_plotlyjs="cdn")

st.download_button(
    "🌐 Heatmap HTML 다운로드",
    html_buf.getvalue(),
    "heatmap.html",
    "text/html"
)
