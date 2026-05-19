import streamlit as st
import plotly.graph_objects as go
import numpy as np
import math
import csv
import sys
import os

# 
# --- modeFRONTIER Response Surface ----------------
# Code Created by
# modeFRONTIER  - (c) ESTECO S.p.A.
# modeFRONTIER Version modeFRONTIER 2025R4 19.7.3 b20251021
# Date Tue Apr 21 12:09:04 JST 2026
# Project Name mf結果4.prj
# Operating System Windows 11 10.0 amd64
# Java (SDK/JRE) Version 21.0.7
# Java Vendor Eclipse Adoptium
# Java Vendor URL https://adoptium.net/
# User Name nakay
# 
# 
# --- DISCLAIMER - Please do not erase -------------
# NO WARRANTY ON RSM CODE
# --------------------------------------------------

class t4_NN_notpv:
    def __init__(self):
        self.n_input = 7
        # load data from file
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(current_dir, 't4_NN_notpv.csv')
            
            with open(csv_path) as csvfile:
                filereader = csv.reader(csvfile)
                next(filereader)
                next(filereader)
                self.x_range = [[0 for _ in range(2)] for _ in range(7)]
                for i in range(7):
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
                self.w1 = [[0 for _ in range(7)] for _ in range(100)]
                for i in range(100):
                    self.w1[i] = [float(value) for value in next(filereader)]
                next(filereader)
                self.b1 = [0 for _ in range(100)]
                for i in range(100):
                    self.b1[i] = float(next(filereader)[0])
                next(filereader)
                self.w2 = [[0 for _ in range(100)] for _ in range(1)]
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
        if len(x) != 9:
            print("ERROR - Wrong Input Vector Length")
            return math.nan
            
        # keep only important input variables
        # 0:J, 1:Je, 3:R4, 5:as, 6:al, 7:ti, 8:to
        xx = [x[0], x[1], x[3], x[5], x[6], x[7], x[8]]

        # normalize input
        xn = [0 for _ in range(self.n_input)]
        for i in range(self.n_input):
            xn[i] = (2 * xx[i] - self.x_range[i][0] - self.x_range[i][1]) / (self.x_range[i][1] - self.x_range[i][0])

        # perform computations
        n1 = [0 for _ in range(len(self.w1))]
        for i in range(len(self.w1)):
            n1[i] = self.b1[i]
            for j in range(len(self.w1[0])):
                n1[i] += self.w1[i][j] * xn[j]
        y1 = [0 for _ in range(len(self.w1))]
        for i in range(len(self.w1)):
            try:
                exp = math.exp(-2.0 * n1[i])
            except OverflowError:
                exp = math.inf
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
        # scale output
        y = self.y_range[0] + (self.y_range[1] - self.y_range[0])/(self.out_range[1] - self.out_range[0]) * (yn[0] - self.out_range[0])
        return y

    def get_input_variable_names(self):
        return ["J", "Je", "R1", "R4", "R5", "as", "al", "ti", "to"]

    def get_output_variable_name(self):
        return "t4"


# ==========================================
# 2. Streamlit ダッシュボード UI
# ==========================================
def main():
    st.set_page_config(layout="wide", page_title="modeFRONTIER Dashboard")
    st.title("📊 modeFRONTIER RSM Dashboard (t4_notpv)")

    try:
        model = t4_NN_notpv()
    except Exception as e:
        st.error(str(e))
        st.info("実行フォルダまたはGitHub上に 't4_NN_notpv.csv' が存在するか確認してください。")
        return

    st.sidebar.header("Input Parameters")
   
    # --- スライダー入力の設定 ---
    # x_rangeのインデックス： 0:J, 1:Je, 2:R4, 3:as, 4:al, 5:ti, 6:to
    v_J   = st.sidebar.slider("J (Active)", float(model.x_range[0][0]), float(model.x_range[0][1]), float((model.x_range[0][0] + model.x_range[0][1])/2))
    v_Je  = st.sidebar.slider("Je (Active)", float(model.x_range[1][0]), float(model.x_range[1][1]), float((model.x_range[1][0] + model.x_range[1][1])/2))
    
    v_R1  = st.sidebar.number_input("R1 (Ignored)", value=0.04)
    v_R4  = st.sidebar.slider("R4 (Active)", float(model.x_range[2][0]), float(model.x_range[2][1]), float((model.x_range[2][0] + model.x_range[2][1])/2))
    v_R5  = st.sidebar.number_input("R5 (Ignored)", value=0.11)
    
    v_as  = st.sidebar.slider("as (Active)", float(model.x_range[3][0]), float(model.x_range[3][1]), float((model.x_range[3][0] + model.x_range[3][1])/2))
    v_al  = st.sidebar.slider("al (Active)", float(model.x_range[4][0]), float(model.x_range[4][1]), float((model.x_range[4][0] + model.x_range[4][1])/2))
    
    # 【変更】ここに配置されていた ε1, ε2 のスライダーを削除しました
    
    v_ti  = st.sidebar.slider("ti (Active)", float(model.x_range[5][0]), float(model.x_range[5][1]), float((model.x_range[5][0] + model.x_range[5][1])/2))
    v_to  = st.sidebar.slider("to (Active)", float(model.x_range[6][0]), float(model.x_range[6][1]), float((model.x_range[6][0] + model.x_range[6][1])/2))


    # --- 数式による派生変数の計算 ---
    st.sidebar.markdown("---")
    st.sidebar.header("Computed Parameters")
   
    v_SAT = v_to + (1.0/23.0) * (v_as * v_J - v_al * v_Je)
    st.sidebar.metric("SAT (Calculated)", f"{v_SAT:.4f}")

    # 【変更】ここに配置されていた R3 の計算とサイドバー表示を削除しました


    # --- モデル評価の実行 ---
    input_vec = [v_J, v_Je, v_R1, v_R4, v_R5, v_as, v_al, v_ti, v_to]
    t4_val = model.evaluate(input_vec)


    # --- メインパネル表示 ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Predicted Output (t4_notpv)", value=f"{t4_val:.6f}")
        st.caption("Active Inputs: 7 variables")
        st.write("---")
        st.write("**Equation-based derived values:**")
        st.write(f"- **SAT:** {v_SAT:.4f}")
        # 【変更】ここに配置されていた R3 のテキスト表示を削除しました


    # --- 3Dグラフ表示 ---
    with col2:
        res = 25
        # X軸: as の範囲（インデックス3）
        x_axis_grid = np.linspace(model.x_range[3][0], model.x_range[3][1], res)
        # Y軸: R4 の範囲（インデックス2）
        y_axis_grid = np.linspace(model.x_range[2][0], model.x_range[2][1], res)
        
        X_MESH, Y_MESH = np.meshgrid(x_axis_grid, y_axis_grid)
        
        Z = np.zeros((res, res))
        for i in range(res):
            for j in range(res):
                temp_input = [v_J, v_Je, v_R1, Y_MESH[i, j], v_R5, X_MESH[i, j], v_al, v_ti, v_to]
                Z[i, j] = model.evaluate(temp_input)

        fig = go.Figure(data=[go.Surface(z=Z, x=x_axis_grid, y=y_axis_grid, colorscale='Viridis')])
        fig.update_layout(
            scene=dict(xaxis_title='as', yaxis_title='R4', zaxis_title='t4_notpv'),
            margin=dict(l=0, r=0, b=0, t=0),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
