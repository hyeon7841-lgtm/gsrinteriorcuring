import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 기본 설정
# ===============================
st.set_page_config(layout="wide")
st.title("🔥 2D 공간 열전달 시뮬레이션 (Streamlit)")

# ===============================
# 사이드바 입력
# ===============================
st.sidebar.header("환경 설정")

outside_temp = st.sidebar.number_input("외부 온도 (°C)", value=0.0)
inside_temp = st.sidebar.number_input("초기 내부 온도 (°C)", value=10.0)

heater_count = st.sidebar.selectbox("열풍기 개수", [1, 2])

st.sidebar.markdown("---")
st.sidebar.subheader("열풍기 위치 (비율)")

heater_positions = []
for i in range(heater_count):
    x = st.sidebar.slider(f"열풍기 {i+1} X 위치", 0.1, 0.9, 0.5)
    y = st.sidebar.slider(f"열풍기 {i+1} Y 위치", 0.1, 0.9, 0.5)
    heater_positions.append((x, y))

run_btn = st.sidebar.button("▶ 시뮬레이션 실행")

# ===============================
# 물리 상수
# ===============================
# 격자
NX, NY = 100, 60
DX = DY = 0.1  # m

# 열 물성 (공기)
ALPHA = 2.1e-5  # 열확산계수 (m^2/s)
RHO = 1.225
CP = 1005

# 열풍기
HEATER_POWER_W = 17600 * 1.163  # kcal/h → W
HEATER_RADIUS = 2  # grid cell

# 시간
DT = 1.0  # s
TOTAL_TIME = 9 * 3600
OUTPUT_INTERVAL = 3600  # 1시간

# ===============================
# 함수 정의
# ===============================
def apply_boundary(T):
    T[0, :] = outside_temp
    T[-1, :] = outside_temp
    T[:, 0] = outside_temp
    T[:, -1] = outside_temp
    return T

def add_heaters(T):
    for hx_ratio, hy_ratio in heater_positions:
        hx = int(hx_ratio * NX)
        hy = int(hy_ratio * NY)
        power_term = HEATER_POWER_W / (RHO * CP)
        T[hx-HEATER_RADIUS:hx+HEATER_RADIUS,
          hy-HEATER_RADIUS:hy+HEATER_RADIUS] += power_term * DT
    return T

def step_temperature(T):
    Tn = T.copy()
    # 2D Finite Difference Method (Vectorized)
    Tn[1:-1, 1:-1] = T[1:-1, 1:-1] + ALPHA * DT * (
        (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1]) / DX**2 +
        (T[1:-1, 2:] - 2*T[1:-1, 1:-1] + T[1:-1, :-2]) / DY**2
    )
    return Tn

def measure_points(T):
    points = {
        "중앙": T[NX//2, NY//2],
        "좌상": T[5, NY-5],
        "우상": T[NX-5, NY-5],
        "좌하": T[5, 5],
        "우하": T[NX-5, 5]
    }
    return points

# ===============================
# 시뮬레이션 실행
# ===============================
if run_btn:
    with st.spinner("열전달 계산 중..."):
        T = np.ones((NX, NY)) * inside_temp
        T = apply_boundary(T)

        results = []
        snapshots = []

        time = 0
        next_output = OUTPUT_INTERVAL

        while time <= TOTAL_TIME:
            T = add_heaters(T)
            T = step_temperature(T)
            T = apply_boundary(T)

            if time >= next_output or time == 0:
                points = measure_points(T)
                avg_temp = np.mean(T)

                results.append({
                    "시간(h)": int(time / 3600),
                    "평균온도": avg_temp,
                    **points
                })
                snapshots.append(T.copy())
                next_output += OUTPUT_INTERVAL

            time += DT

    # ===============================
    # 결과 출력
    # ===============================
    st.success("시뮬레이션 완료")

    # 시간 선택
    hour = st.slider("시간 선택 (h)", 0, 9, 0)
    idx = min(hour, len(snapshots)-1)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        im = ax.imshow(
            snapshots[idx].T,
            origin="lower",
            cmap="hot"
        )
        plt.colorbar(im, ax=ax, label="Temperature (°C)")
        ax.set_title(f"{hour} 시간 후 열 분포")
        st.pyplot(fig)

    with col2:
        st.subheader("📊 평균 온도 (°C)")
        st.table(results[idx])

    # ===============================
    # 시간별 변화 그래프
    # ===============================
    st.subheader("⏱ 시간별 평균 온도 변화")

    hours = [r["시간(h)"] for r in results]
    avg_temps = [r["평균온도"] for r in results]

    fig2, ax2 = plt.subplots()
    ax2.plot(hours, avg_temps, marker="o")
    ax2.set_xlabel("시간 (h)")
    ax2.set_ylabel("평균 온도 (°C)")
    ax2.grid(True)

    st.pyplot(fig2)

else:
    st.info("좌측에서 설정 후 ▶ 시뮬레이션 실행을 눌러주세요.")
