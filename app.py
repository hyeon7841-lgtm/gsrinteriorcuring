import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# ==================================================
# 기본 설정
# ==================================================
st.set_page_config(layout="wide")
st.title("🔥 공간 기반 2D 열전달 시뮬레이션")

# ==================================================
# 세션 상태 초기화
# ==================================================
if "space_points" not in st.session_state:
    st.session_state.space_points = []

if "space_closed" not in st.session_state:
    st.session_state.space_closed = False

if "heater_points" not in st.session_state:
    st.session_state.heater_points = []

# ==================================================
# 사이드바
# ==================================================
st.sidebar.header("환경 조건")

outside_temp = st.sidebar.number_input("외부 온도 (°C)", value=0.0)
inside_temp = st.sidebar.number_input("초기 내부 온도 (°C)", value=10.0)

heater_count = st.sidebar.selectbox("열풍기 개수", [1, 2])

if st.sidebar.button("❌ 전체 초기화"):
    st.session_state.space_points = []
    st.session_state.space_closed = False
    st.session_state.heater_points = []
    st.rerun()

run_btn = st.sidebar.button("▶ 시뮬레이션 실행")

# ==================================================
# 격자 및 물성
# ==================================================
NX, NY = 100, 60
DX = DY = 0.1

ALPHA = 2.1e-5
RHO = 1.225
CP = 1005

HEATER_POWER_W = 17600 * 1.163
HEATER_RADIUS = 2

DT = 1.0
TOTAL_TIME = 9 * 3600
OUTPUT_INTERVAL = 3600

# ==================================================
# 기하 함수
# ==================================================
def point_in_polygon(x, y, poly):
    inside = False
    n = len(poly)
    px, py = zip(*poly)
    j = n - 1
    for i in range(n):
        if ((py[i] > y) != (py[j] > y)) and \
           (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i] + 1e-9) + px[i]):
            inside = not inside
        j = i
    return inside

# ==================================================
# 1단계: 공간 그리기
# ==================================================
st.subheader("🧱 1단계: 내부 공간 그리기 (직선 클릭)")

fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        z=np.zeros((NY, NX)),
        colorscale="Greys",
        showscale=False
    )
)

if st.session_state.space_points:
    xs, ys = zip(*st.session_state.space_points)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+lines",
            marker=dict(color="blue", size=10),
            line=dict(color="blue", width=2),
            name="공간 경계"
        )
    )

if st.session_state.space_closed:
    xs, ys = zip(*(st.session_state.space_points + [st.session_state.space_points[0]]))
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="blue", width=3),
            name="완성된 공간"
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
    width=720,
    height=420,
    xaxis=dict(range=[0, NX]),
    yaxis=dict(range=[0, NY]),
    title="공간 꼭지점을 순서대로 클릭하세요"
)

clicked = plotly_events(fig, click_event=True)

if clicked:
    x = int(clicked[0]["x"])
    y = int(clicked[0]["y"])

    if not st.session_state.space_closed:
        st.session_state.space_points.append((x, y))
        st.rerun()
    else:
        if point_in_polygon(x, y, st.session_state.space_points):
            if len(st.session_state.heater_points) < heater_count:
                st.session_state.heater_points.append((x, y))
                st.rerun()
        else:
            st.warning("열풍기는 내부 공간에만 배치할 수 있습니다.")

if len(st.session_state.space_points) >= 3 and not st.session_state.space_closed:
    if st.button("✅ 공간 완성"):
        st.session_state.space_closed = True
        st.rerun()

# ==================================================
# 열전달 계산 함수
# ==================================================
def apply_boundary(T, mask):
    T[~mask] = outside_temp
    return T

def add_heaters(T):
    for hx, hy in st.session_state.heater_points:
        T[hx-HEATER_RADIUS:hx+HEATER_RADIUS,
          hy-HEATER_RADIUS:hy+HEATER_RADIUS] += (HEATER_POWER_W / (RHO * CP)) * DT
    return T

def step_temperature(T, mask):
    Tn = T.copy()
    for i in range(1, NX-1):
        for j in range(1, NY-1):
            if mask[i, j]:
                Tn[i, j] = T[i, j] + ALPHA * DT * (
                    (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / DX**2 +
                    (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / DY**2
                )
    return Tn

# ==================================================
# 시뮬레이션 실행
# ==================================================
if run_btn:
    if not st.session_state.space_closed:
        st.error("공간을 먼저 완성하세요.")
    elif len(st.session_state.heater_points) < heater_count:
        st.error("열풍기 위치를 모두 배치하세요.")
    else:
        with st.spinner("열전달 계산 중..."):
            mask = np.zeros((NX, NY), dtype=bool)
            for i in range(NX):
                for j in range(NY):
                    if point_in_polygon(i, j, st.session_state.space_points):
                        mask[i, j] = True

            T = np.ones((NX, NY)) * inside_temp
            T = apply_boundary(T, mask)

            snapshots = []
            results = []

            time = 0
            next_out = 0

            while time <= TOTAL_TIME:
                T = add_heaters(T)
                T = step_temperature(T, mask)
                T = apply_boundary(T, mask)

                if time >= next_out:
                    snapshots.append(T.copy())
                    results.append({
                        "시간(h)": int(time / 3600),
                        "평균온도": np.mean(T[mask])
                    })
                    next_out += OUTPUT_INTERVAL

                time += DT

        st.success("시뮬레이션 완료")

        hour = st.slider("시간 선택 (h)", 0, 9, 0)
        idx = min(hour, len(snapshots)-1)

        fig2, ax = plt.subplots()
        im = ax.imshow(snapshots[idx].T, origin="lower", cmap="hot")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{hour}시간 후 열 분포")
        st.pyplot(fig2)

        st.table(results[idx])

else:
    st.info("① 공간 그리기 → ② 공간 완성 → ③ 열풍기 배치 → ④ 시뮬레이션 실행")
