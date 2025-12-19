import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# =====================================
# Streamlit 기본 설정
# =====================================
st.set_page_config(layout="wide")
st.title("🔥 2D 공간 열전달 시뮬레이션")

# =====================================
# 사이드바 설정
# =====================================
st.sidebar.header("환경 조건")

outside_temp = st.sidebar.number_input("외부 온도 (°C)", value=0.0)
inside_temp = st.sidebar.number_input("초기 내부 온도 (°C)", value=10.0)

heater_count = st.sidebar.selectbox("열풍기 개수", [1, 2])

run_btn = st.sidebar.button("▶ 시뮬레이션 실행")
reset_btn = st.sidebar.button("❌ 열풍기 위치 초기화")

# =====================================
# 세션 상태 초기화
# =====================================
if "heater_points" not in st.session_state:
    st.session_state.heater_points = []

if "click_count" not in st.session_state:
    st.session_state.click_count = 0

if reset_btn:
    st.session_state.heater_points = []
    st.session_state.click_count = 0
    st.experimental_rerun()

# =====================================
# 격자 & 물리 상수
# =====================================
NX, NY = 100, 60
DX = DY = 0.1  # m

ALPHA = 2.1e-5  # m²/s
RHO = 1.225
CP = 1005

HEATER_POWER_W = 17600 * 1.163  # kcal/h → W
HEATER_RADIUS = 2

DT = 1.0
TOTAL_TIME = 9 * 3600
OUTPUT_INTERVAL = 3600

# =====================================
# 열원 배치 UI
# =====================================
st.subheader("🖱 공간을 클릭해서 열풍기 배치")

fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        z=np.zeros((NY, NX)),
        colorscale="Greys",
        showscale=False
    )
)

if st.session_state.heater_points:
    xs, ys = zip(*st.session_state.heater_points)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(color="red", size=14),
            name="열풍기"
        )
    )

fig.update_layout(
    width=700,
    height=420,
    xaxis=dict(range=[0, NX], showgrid=False),
    yaxis=dict(range=[0, NY], showgrid=False),
    title="열풍기 위치를 클릭하세요"
)

clicked = plotly_events(fig, click_event=True)

if clicked and st.session_state.click_count < heater_count:
    x = int(clicked[0]["x"])
    y = int(clicked[0]["y"])
    st.session_state.heater_points.append((x, y))
    st.session_state.click_count += 1
    st.experimental_rerun()

st.info(f"선택된 열풍기 위치: {st.session_state.heater_points}")

# =====================================
# 열전달 계산 함수
# =====================================
def apply_boundary(T):
    T[0, :] = outside_temp
    T[-1, :] = outside_temp
    T[:, 0] = outside_temp
    T[:, -1] = outside_temp
    return T

def add_heaters(T):
    for hx, hy in st.session_state.heater_points:
        power_term = HEATER_POWER_W / (RHO * CP)
        T[hx-HEATER_RADIUS:hx+HEATER_RADIUS,
          hy-HEATER_RADIUS:hy+HEATER_RADIUS] += power_term * DT
    return T

def step_temperature(T):
    Tn = T.copy()
    for i in range(1, NX-1):
        for j in range(1, NY-1):
            Tn[i, j] = T[i, j] + ALPHA * DT * (
                (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / DX**2 +
                (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / DY**2
            )
    return Tn

def measure_points(T):
    return {
        "중앙": T[NX//2, NY//2],
        "좌상": T[5, NY-5],
        "우상": T[NX-5, NY-5],
        "좌하": T[5, 5],
        "우하": T[NX-5, 5]
    }

# =====================================
# 시뮬레이션 실행
# =====================================
if run_btn:
    if len(st.session_state.heater_points) < heater_count:
        st.error("열풍기 개수만큼 위치를 먼저 클릭하세요.")
    else:
        with st.spinner("열전달 계산 중..."):
            T = np.ones((NX, NY)) * inside_temp
            T = apply_boundary(T)

            snapshots = []
            results = []

            time = 0
            next_output = 0

            while time <= TOTAL_TIME:
                T = add_heaters(T)
                T = step_temperature(T)
                T = apply_boundary(T)

                if time >= next_output:
                    points = measure_points(T)
                    results.append({
                        "시간(h)": int(time / 3600),
                        "평균온도": np.mean(T),
                        **points
                    })
                    snapshots.append(T.copy())
                    next_output += OUTPUT_INTERVAL

                time += DT

        st.success("시뮬레이션 완료")

        # =====================================
        # 결과 시각화
        # =====================================
        hour = st.slider("시간 선택 (h)", 0, 9, 0)
        idx = min(hour, len(snapshots)-1)

        col1, col2 = st.columns(2)

        with col1:
            fig2, ax = plt.subplots()
            im = ax.imshow(
                snapshots[idx].T,
                origin="lower",
                cmap="hot"
            )
            plt.colorbar(im, ax=ax, label="Temperature (°C)")
            ax.set_title(f"{hour}시간 후 열 분포")
            st.pyplot(fig2)

        with col2:
            st.subheader("📊 평균 온도 (°C)")
            st.table(results[idx])

        st.subheader("⏱ 시간별 평균 온도 변화")
        hours = [r["시간(h)"] for r in results]
        avgs = [r["평균온도"] for r in results]

        fig3, ax3 = plt.subplots()
        ax3.plot(hours, avgs, marker="o")
        ax3.set_xlabel("시간 (h)")
        ax3.set_ylabel("평균 온도 (°C)")
        ax3.grid(True)
        st.pyplot(fig3)

else:
    st.info("좌측에서 조건 설정 → 공간 클릭 → 시뮬레이션 실행")
