import streamlit as st
import plotly.graph_objects as go
import numpy as np
import math
import csv
import sys
import os

# --- modeFRONTIER Response Surface ----------------
# Code Created by modeFRONTIER 2025R4
# --------------------------------------------------

class t4_NN_0_pv:
    def __init__(self):
        self.n_input = 10
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(current_dir, 't4_NN_0_pv.csv')
            
            with open(csv_path) as csvfile:
                filereader = csv.reader(csvfile)
                next(filereader)
                next(filereader)
                self.x_range = [[0 for _ in range(2)] for _ in range(10)]
                for i in range(10):
                    self.x_range[i] = [float(value) for value in next(filereader)]
                next(filereader)
                self.y_range = [0 for _ in range(2)]
                for i in range(2):
                    self.y_range[i] = float(next(filereader)[0])
                next(filereader)
                self.out_range = [0 for _ in range(2)]
                for i in range(2):
                    self.out_range[i] = float(next(filereader)[0])
                next(filereader)
                self.w1 = [[0 for _ in range(10)] for _ in range(8)]
                for i in range(8):
                    self.w1[i] = [float(value) for value in next(filereader)]
                next(filereader)
                self.b1 = [0 for _ in range(8)]
                for i in range(8):
                    self.b1[i] = float(next(filereader)[0])
                next(filereader)
                self.w2 = [[0 for _ in range(8)] for _ in range(1)]
                for i in range(1):
                    self.w2[i] = [float(value) for value in next(filereader)]
                next(filereader)
                self.b2 = [0 for _ in range(1)]
                for i in range(1):
                    self.b2[i] = float(next(filereader)[0])
                next(filereader)
                csvfile.close()
        except OSError:
            raise FileNotFoundError(f"データファイルが見つかりません: {csv_path}")
        except StopIteration:
            pass

    def evaluate(self, x):
        if len(x) != 12:
            print("ERROR - Wrong Input Vector Length")
            return math.nan
        
        # モデルが使用する10個の重要変数を抽出
        # x[2](R1) と x[5](R5) は無視される
        xx = [x[0], x[1], x[3], x[4], x[6], x[7], x[8], x[9], x[10], x[11]]

        # 入力の正規化
        xn = [0 for _ in range(self.n_input)]
        for i in range(self.n_input):
            xn[i] = (2 * xx[i] - self.x_range[i][0] - self.x_range[i][1]) / (self.x_range[i][1] - self.x_range[i][0])

        # ニューラルネットワーク計算
        n1 = [0 for _ in range(len(self.w1))]
        for i in range(len(self.w1)):
            n1[i] = self.b1[i]
            for j in range(len(self.w1[0])):
                n1[i] += self.w1[i][j] * xn[j]
        y1 = [0 for _ in range(len(self.w1))]
        for i in range(len(self.w1)):
            exp = math.exp(-2.0 * n1[i])
            if exp == math.inf:
                y1[i] = -1.0
            else:
                y1[i] = (1.0 - exp)/(1.0 + exp)
        n2 = [0 for _ in range(len(self.w2))]
        for i in range(len(self.w2)):
            n2[i] = self.b2[i]
            for j in range(len(self.w2[0])):
                n2[i] += self.w2[i][j] * y1[j]
        yn = [0 for _ in range(len(self.w2))]
        for i in range(len(self.w2)):
            yn[i] = n2[i]
            
        # 出力のスケール戻し
        y = self.y_range[0] + (self.y_range[1] - self.y_range[0])/(self.out_range[1] - self.out_range[0]) * (yn[0] - self.out_range[0])
        return y

    def get_input_variable_names(self):
        # 6番目がal、7番目がas
        return ["J", "Je", "R1", "R2", "R4", "R5", "al", "as", "e1", "e2", "ti", "to"]

    def get_output_variable_name(self):
        return "t4"


# ==========================================
# 2. Streamlit ダッシュボード UI
# ==========================================
def main():
    st.set_page_config(layout="wide", page_title="modeFRONTIER Dashboard")
    st.title("📊 modeFRONTIER RSM Dashboard (t4)")

    try:
        model = t4_NN_0_pv()
    except Exception as e:
        st.error(str(e))
        st.info("実行フォルダに 't4_NN_0_pv.csv' が存在するか確認してください。")
        return

    st.sidebar.header("Input Parameters")
    
    # --- 重要変数のインデックス順に基づき、model.x_range から正しく範囲を取得 ---
    v_J   = st.sidebar.slider("J (Active)", float(model.x_range[0][0]), float(model.x_range[0][1]), float((model.x_range[0][0] + model.x_range[0][1])/2))
    v_Je  = st.sidebar.slider("Je (Active)", float(model.x_range[1][0]), float(model.x_range[1][1]), float((model.x_range[1][0] + model.x_range[1][1])/2))
    
    v_R1  = st.sidebar.number_input("R1 (Ignored)", value=0.0)
    v_R2  = st.sidebar.slider("R2 (Active)", float(model.x_range[2][0]), float(model.x_range[2][1]), float((model.x_range[2][0] + model.x_range[2][1])/2))
    v_R4  = st.sidebar.slider("R4 (Active)", float(model.x_range[3][0]), float(model.x_range[3][1]), float((model.x_range[3][0] + model.x_range[3][1])/2))
    v_R5  = st.sidebar.number_input("R5 (Ignored)", value=0.0)
    
    # インデックス4がal、5がas
    v_al  = st.sidebar.slider("al (Active)", float(model.x_range[4][0]), float(model.x_range[4][1]), float((model.x_range[4][0] + model.x_range[4][1])/2))
    v_as  = st.sidebar.slider("as (Active)", float(model.x_range[5][0]), float(model.x_range[5][1]), float((model.x_range[5][0] + model.x_range[5][1])/2))
    
    v_ε1  = st.sidebar.slider("ε1 (Active)", float(model.x_range[6][0]), float(model.x_range[6][1]), float((model.x_range[6][0] + model.x_range[6][1])/2))
    v_ε2  = st.sidebar.slider("ε2 (Active)", float(model.x_range[7][0]), float(model.x_range[7][1]), float((model.x_range[7][0] + model.x_range[7][1])/2))
    v_ti  = st.sidebar.slider("ti (Active)", float(model.x_range[8][0]), float(model.x_range[8][1]), float((model.x_range[8][0] + model.x_range[8][1])/2))
    v_to  = st.sidebar.slider("to (Active)", float(model.x_range[9][0]), float(model.x_range[9][1]), float((model.x_range[9][0] + model.x_range[9][1])/2))

    # --- 数式による派生変数の計算 ---
    st.sidebar.markdown("---")
    st.sidebar.header("Computed Parameters")
    
    # 1. SATの計算
    v_SAT = v_to + (1.0/23.0) * (v_as * v_J - v_al * v_Je)
    st.sidebar.metric("SAT (Calculated)", f"{v_SAT:.4f}")

    # 2. R3の計算
    try:
        rad_term = (1.0 / (max(v_ε1, 1e-10)**-1 + max(v_ε2, 1e-10)**-1 - 1.0)) * 4.0 * (293.0**3) * 5.67e-8
        v_R3 = 1.0 / (25.0 + rad_term)
    except:
        v_R3 = 0.0
    st.sidebar.metric("R3 (Calculated)", f"{v_R3:.6f}")

    # --- モデル評価の実行 (正しい変数順序 [..., al, as, ...] で渡す) ---
    input_vec = [v_J, v_Je, v_R1, v_R2, v_R4, v_R5, v_al, v_as, v_ε1, v_ε2, v_ti, v_to]
    t4_val = model.evaluate(input_vec)

    # --- メインパネル表示 ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Predicted Output (t4)", value=f"{t4_val:.6f}")
        st.caption("Active Inputs: 10 variables")
        st.write("---")
        st.write("**Equation-based derived values:**")
        st.write(f"- **SAT:** {v_SAT:.4f}")
        st.write(f"- **R3:** {v_R3:.6f}")

    # --- 3Dグラフ表示 (X軸: as, Y軸: R4) ---
    with col2:
        res = 25
        # X軸に as (インデックス5)、Y軸に R4 (インデックス3) を割り当て
        x_axis_grid = np.linspace(model.x_range[5][0], model.x_range[5][1], res)
        y_axis_grid = np.linspace(model.x_range[3][0], model.x_range[3][1], res)
        
        Z = np.zeros((res, res))
        for i in range(res):     # 行 (Y軸: R4)
            for j in range(res): # 列 (X軸: as)
                # メッシュの値を適切な位置に代入して evaluate を実行
                temp_input = [v_J, v_Je, v_R1, v_R2, y_axis_grid[i], v_R5, v_al, x_axis_grid[j], v_ε1, v_ε2, v_ti, v_to]
                Z[i, j] = model.evaluate(temp_input)

        # 軸反転を防ぐため1次元配列のままPlotlyに渡す
        fig = go.Figure(data=[go.Surface(z=Z, x=x_axis_grid, y=y_axis_grid, colorscale='Viridis')])
        fig.update_layout(
            scene=dict(xaxis_title='as', yaxis_title='R4', zaxis_title='t4'),
            margin=dict(l=0, r=0, b=0, t=0),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()