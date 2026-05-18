import streamlit as st
import plotly.graph_objects as go
import numpy as np
import math
import csv
import sys
import os

# 
# --- modeFRONTIER Response Surface ----------------
# (コメント中略)
# --------------------------------------------------

class t4_NN_0_notpv:
    def __init__(self):
        self.n_input = 7
        # load data from file
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(current_dir, 't4_NN_0_notpv.csv')
            
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
                self.w1 = [[0 for _ in range(7)] for _ in range(11)]
                for i in range(11):
                    self.w1[i] = [float(value) for value in next(filereader)]
                next(filereader)
                self.b1 = [0 for _ in range(11)]
                for i in range(11):
                    self.b1[i] = float(next(filereader)[0])
                next(filereader)
                self.w2 = [[0 for _ in range(11)] for _ in range(1)]
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
        # check input (t4モデルは長さ9の入力を期待しています)
        if len(x) != 9:
            print("ERROR - Wrong Input Vector Length")
            return math.nan
            
        # keep only important input variables
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
            # 【修正】OverflowError対策
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
        return ["J", "Je", "R1", "R4", "R5", "al", "as", "ti", "to"]

    def get_output_variable_name(self):
        return "t4"


# ==========================================
# 2. Streamlit ダッシュボード UI
# ==========================================
def main():
    st.set_page_config(layout="wide", page_title="modeFRONTIER Dashboard")
    st.title("📊 modeFRONTIER RSM Dashboard (t4_pv無し)")

    try:
        model = t4_NN_0_notpv()
    except Exception as e:
        st.error(str(e))
        st.info("実行フォルダに 't4_NN_0_notpv.csv' が存在するか確認してください。")
        return

    st.sidebar.header("Input Parameters")
   
    # --- スライダー入力の設定 ---
    # t4モデルの active variables (7個): J(0), Je(1), R4(2), al(3), as(4), ti(5), to(6)
    
    v_J   = st.sidebar.slider("J (Active)", float(model.x_range[0][0]), float(model.x_range[0][1]), float((model.x_range[0][0] + model.x_range[0][1])/2))
    v_Je  = st.sidebar.slider("Je (Active)", 0.0, 150.0, 75.0)
    v_R1  = st.sidebar.number_input("R1 (Ignored)", value=0.04)
   
    # R2はモデルから削除されたため、入力から外しています
    
    v_R4  = st.sidebar.slider("R4 (Active)", 0.0, 7.0, 3.5)
    v_R5  = st.sidebar.number_input("R5 (Ignored)", value=0.11)
   
    v_as  = st.sidebar.slider("as (Active)", 0.02, 1.0, 0.5)
    # 【修正】alのインデックスを3に変更
    v_al  = st.sidebar.slider("al (Active)", float(model.x_range[3][0]), float(model.x_range[3][1]), float((model.x_range[3][0] + model.x_range[3][1])/2))
    
    # ε1, ε2はt4モデルには使われませんが、R3の計算用にスライダーを残します（独立した範囲を指定）
    v_ε1  = st.sidebar.slider("ε1 (For R3 Calc)", 0.01, 1.0, 0.5)
    v_ε2  = st.sidebar.slider("ε2 (For R3 Calc)", 0.01, 1.0, 0.5)
    
    # 【修正】ti, toのインデックスを5, 6に変更
    v_ti  = st.sidebar.slider("ti (Active)", float(model.x_range[5][0]), float(model.x_range[5][1]), float((model.x_range[5][0] + model.x_range[5][1])/2))
    v_to  = st.sidebar.slider("to (Active)", float(model.x_range[6][0]), float(model.x_range[6][1]), float((model.x_range[6][0] + model.x_range[6][1])/2))


    # --- 数式による派生変数の計算 ---
    st.sidebar.markdown("---")
    st.sidebar.header("Computed Parameters")
   
    v_SAT = v_to + (1.0/23.0) * (v_as * v_J - v_al * v_Je)
    st.sidebar.metric("SAT (Calculated)", f"{v_SAT:.4f}")

    try:
        rad_term = (1.0 / (max(v_ε1, 1e-10)**-1 + max(v_ε2, 1e-10)**-1 - 1.0)) * 4.0 * (293.0**3) * 5.67e-8
        v_R3 = 1.0 / (25.0 + rad_term)
    except:
        v_R3 = 0.0
    st.sidebar.metric("R3 (Calculated)", f"{v_R3:.6f}")


    # --- モデル評価の実行 ---
    # 【修正】t4モデルに合わせて入力変数を9個に厳選
    input_vec = [v_J, v_Je, v_R1, v_R4, v_R5, v_al, v_as, v_ti, v_to]
    t4_val = model.evaluate(input_vec)


    # --- メインパネル表示 ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Predicted Output (t4)", value=f"{t4_val:.6f}")
        st.caption("Active Inputs: 7 variables")
        st.write("---")
        st.write("**Equation-based derived values:**")
        st.write(f"- **SAT:** {v_SAT:.4f}")
        st.write(f"- **R3:** {v_R3:.6f}")


    # --- 3Dグラフ表示 ---
    with col2:
        res = 25
        # 【変更】X軸を 'as' に変更
        # t4モデルの内部変数（7個）のうち、as は5番目（インデックス4）に該当します
        x_axis_grid = np.linspace(model.x_range[4][0], model.x_range[4][1], res) # as軸
        y_axis_grid = np.linspace(0.0, 7.0, res) # R4軸
        X_MESH, Y_MESH = np.meshgrid(x_axis_grid, y_axis_grid)
       
        Z = np.zeros((res, res))
        for i in range(res):
            for j in range(res):
                # 【変更】J (0番目) はスライダーの値 v_J を使い、as (6番目) に X_MESH を割り当てます
                temp_input = [v_J, v_Je, v_R1, Y_MESH[i, j], v_R5, v_al, X_MESH[i, j], v_ti, v_to]
                Z[i, j] = model.evaluate(temp_input)

        fig = go.Figure(data=[go.Surface(z=Z, x=x_axis_grid, y=y_axis_grid, colorscale='Viridis')])
        fig.update_layout(
            # 【変更】X軸のラベルを 'as' に変更
            scene=dict(xaxis_title='as', yaxis_title='R4', zaxis_title='t4'),
            margin=dict(l=0, r=0, b=0, t=0),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()